# Import libraries

import os
import torch
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from metadata_conditioning import MetadataConditionEncoder


# LoRA class & helper (same as in train)

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


# Inject metadata into attention (same pattern as train)

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


# Image sampling from metadata-conditioned Stable Diffusion

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

    # Inject conditioning into all attention layers
    
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = cond
    
    # DDPM scheduler setup

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    latents = torch.randn(1, 4, 16, 16, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
	
	# Decode latents using Stable Diffusion's VAE 
	
        latents = latents / 0.18215
        image = vae.decode(latents).sample
    
    # Postprocess to uint8 image
    
    image = (image.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype("uint8")

    return Image.fromarray(image)

# Load Stable Diffusion components, restore LoRA + metadata encoder weights and generate a sample conditioned image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_path = "lora_numeric_128/lora_numeric_condition.pth"
    model_name = "runwayml/stable-diffusion-v1-5"

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(device)
    vae.eval()

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    # Apply LoRA structure and inject metadata into attention
    
    print("Applying LoRA structure...")
    lora_layers = apply_lora_to_unet(unet, r=4)
    inject_metadata_into_attention(unet, device)

    # Load checkpoint and LoRA weights
    
    print("Loading LoRA & cond encoder...")
    ckpt = torch.load(lora_path, map_location="cpu")
    state = ckpt["cond_encoder"]
    num_dx = state["dx_emb.weight"].shape[0]
    num_sites = state["site_emb.weight"].shape[0]
    num_sex = state["sex_emb.weight"].shape[0]

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

    sample_metadata = {
        "dx": torch.tensor([0], device=device),
        "site": torch.tensor([0], device=device),
        "sex": torch.tensor([0], device=device),
        "age": torch.tensor([0.55], device=device),
    }

    print("Generating image...")
    out = generate_image(cond_encoder, unet, vae, sample_metadata, device=device)
    out.save("numeric_condition_sample.png")
    print("Saved: numeric_condition_sample.png")


if __name__ == "__main__":
    main()

