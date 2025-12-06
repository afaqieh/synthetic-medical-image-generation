# Import libraries

import os
import pandas as pd
import torch
from metadata_conditioning import MetadataConditionEncoder
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from PIL import Image

#Generate a dataset of metadata-conditioned images using the trained LoRA model. These images are used for FID and MS-SSIM evaluation.
# Reuses the same LoRA and attention-patching functions defined in the training pipeline

class LoRALinear(torch.nn.Module):
    def __init__(self, original, r=4, alpha=1.0):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha

        self.down = torch.nn.Linear(original.in_features, r, bias=False)
        self.up = torch.nn.Linear(r, original.out_features, bias=False)

        self.scaling = alpha / r

        torch.nn.init.normal_(self.down.weight, std=0.01)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.original(x) + self.up(self.down(x)) * self.scaling


def inject_metadata_into_attention(unet, device):
    def make_replacement(attn2):
        original_call = attn2.processor.__call__

        def new_call(attn, hidden_states, encoder_hidden_states=None, **kwargs):
            encoder_hidden_states = attn2.metadata_context
            return original_call(attn, hidden_states, encoder_hidden_states, **kwargs)

        return new_call

    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = torch.zeros(1, 77, 768, device=device)
            module.attn2.processor.__call__ = make_replacement(module.attn2)

def apply_lora_to_unet(unet, r=4):
    lora_layers = []
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["to_q", "to_k", "to_v", "to_out"]):
                parent = unet
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                last = parts[-1]

                orig = getattr(parent, last)
                lora = LoRALinear(orig, r=r)
                setattr(parent, last, lora)
                lora_layers.append(lora)
    return lora_layers

def generate_image(
    cond_encoder,
    unet,
    vae,
    metadata,
    device="cuda",
    steps=30,
):
    cond = cond_encoder(metadata)               
    cond = cond.unsqueeze(1).repeat(1, 77, 1)   
    
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = cond

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)

    latents = torch.randn(1, 4, 16, 16, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t, encoder_hidden_states=cond ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        image = vae.decode(latents).sample

    image = (image.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype("uint8")

    return Image.fromarray(image)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    
    model_name = "runwayml/stable-diffusion-v1-5"
    lora_path = "lora_numeric_full/lora_numeric_condition.pth"   # <-- full run
    csv_path = "data/HAM10000_metadata.csv"
    out_dir = "fid_eval/methodB"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading metadata CSV...")
    df = pd.read_csv(csv_path).dropna(subset=["dx", "localization", "sex", "age"]).reset_index(drop=True)

    dx2idx   = {k: i for i, k in enumerate(sorted(df["dx"].unique()))}
    site2idx = {k: i for i, k in enumerate(sorted(df["localization"].unique()))}
    sex2idx  = {k: i for i, k in enumerate(sorted(df["sex"].unique()))}

    # Choose how many images to generate for FID
    
    N = 500
    df = df.sample(n=N, random_state=0).reset_index(drop=True)

    print("Loading VAE & UNet...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    # apply LoRA + inject metadata
     
    lora_layers = apply_lora_to_unet(unet, r=16)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)

    # load checkpoint
    
    ckpt = torch.load(lora_path, map_location="cpu")
    state = ckpt["cond_encoder"]
    num_dx   = state["dx_emb.weight"].shape[0]
    num_sites = state["site_emb.weight"].shape[0]
    num_sex  = state["sex_emb.weight"].shape[0]

    cond_encoder = MetadataConditionEncoder(
        num_dx=num_dx,
        num_sites=num_sites,
        num_sex=num_sex,
        hidden_dim=256,
        final_dim=768
    )
    cond_encoder.load_state_dict(state)
    cond_encoder.to(device).eval()

    for layer, layer_state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(layer_state)

    print(f"Generating {N} images for FID...")
    for i, row in df.iterrows():
        meta = {
            "dx":   torch.tensor([dx2idx[row["dx"]]], device=device),
            "site": torch.tensor([site2idx[row["localization"]]], device=device),
            "sex":  torch.tensor([sex2idx[row["sex"]]], device=device),
            "age":  torch.tensor([float(row["age"]) / 100.0], device=device),
        }

        img = generate_image(cond_encoder, unet, vae, meta, device=device, steps=30)
        img.save(os.path.join(out_dir, f"{i:04d}.png"))

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{N} images done")

    print("Done. Images saved in", out_dir)


if __name__ == "__main__":
    main()

