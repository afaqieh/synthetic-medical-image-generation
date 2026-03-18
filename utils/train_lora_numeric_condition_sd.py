# Import libraries

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from dataset_ham10000 import HAM10000Dataset
from metadata_conditioning import MetadataConditionEncoder


# dx mapping from dataset

DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# LoRA wrapper for replacing Linear layers in the UNet's attention blocks.

class LoRALinear(nn.Module):
    def __init__(self, original, r=4, alpha=1.0):
        super().__init__()
        self.original = original
        self.r = r
        self.alpha = alpha

        self.down = nn.Linear(original.in_features, r, bias=False)
        self.up = nn.Linear(r, original.out_features, bias=False)

        self.scaling = alpha / r

        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.original(x) + self.up(self.down(x)) * self.scaling

# Replace UNet's attention linear layers  with LoRA-wrapped versions

def apply_lora_to_unet(unet, r=4):
    lora_layers = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            if any(keyword in name for keyword in ["to_q", "to_k", "to_v", "to_out"]):
                parent = unet
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                last = parts[-1]

                orig_layer = getattr(parent, last)
                lora_layer = LoRALinear(orig_layer, r=r)

                setattr(parent, last, lora_layer)
                lora_layers.append(lora_layer)
    return lora_layers


# Override all cross attention layers so they use metadata embedding instead of the original text encoder

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

# Generate and save a single example image using fixed metadata to monitor training progress over time

def save_sample(unet, vae, cond_encoder, output_dir, device, step):

    meta = {
        "dx": torch.tensor([5], device=device),
        "site": torch.tensor([0], device=device),
        "sex": torch.tensor([0], device=device),
        "age": torch.tensor([0.45], device=device),
    }

    cond = cond_encoder(meta)
    cond = cond.unsqueeze(1).repeat(1, 77, 1)

    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = cond

    model_name = "runwayml/stable-diffusion-v1-5"  # o pásalo como argumento
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(70)

    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t,encoder_hidden_states=cond).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        img = vae.decode(latents).sample

    img = ((img.clamp(-1, 1) + 1) / 2)[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")

    filename = os.path.join(output_dir, f"sample_step_{step}.png")
    Image.fromarray(img).save(filename)
    print(f"Saved general sample → {filename}")

# Generate and save one sample image per disease class to track class-specific model performance during training

def save_all_classes(unet, vae, cond_encoder, output_dir, device, step):

    num_classes = len(DX_LABELS)

    folder = os.path.join(output_dir, "samples_by_class", f"step_{step}")
    os.makedirs(folder, exist_ok=True)

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(70)

    for cls in range(num_classes):

        meta = {
            "dx": torch.tensor([cls], device=device),
            "site": torch.tensor([0], device=device),
            "sex": torch.tensor([0], device=device),
            "age": torch.tensor([0.50], device=device),
        }

        cond = cond_encoder(meta)
        cond = cond.unsqueeze(1).repeat(1, 77, 1)

        for _, module in unet.named_modules():
            if hasattr(module, "attn2"):
                module.attn2.metadata_context = cond

        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = unet(latents, t,encoder_hidden_states=cond).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / 0.18215
            img = vae.decode(latents).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        filename = os.path.join(folder, f"{DX_LABELS[cls]}.png")
        Image.fromarray(img).save(filename)

    print(f"Saved per-class samples → {folder}")

# Train the metadata-conditioned Stable Diffusion LoRA model by updating LoRA attention 
# adapters and the metadata encoder while keeping the base UNet and VAE frozen.

def train(
    csv="../data/HAM10000_metadata.csv",
    img_dir="../data/HAM10000_images",
    outdir="lora_numeric_full",
    batch_size=4,
    lr=3e-5,
    steps=20000,
    r=16,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    ).to(device)
    vae.eval()

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)

    for p in unet.parameters():
        p.requires_grad = False

    print("Building dataset...")
    dataset = HAM10000Dataset(csv, img_dir, vae, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Building metadata encoder...")
    cond_encoder = MetadataConditionEncoder(
        num_dx=len(dataset.dx2idx),
        num_sites=len(dataset.site2idx),
        num_sex=len(dataset.sex2idx),
        hidden_dim=256,
        final_dim=768
    ).to(device)

    print("Injecting metadata into attention...")
    inject_metadata_into_attention(unet, device)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=r)
    for layer in lora_layers:
        layer.to(device)

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) +
        [p for layer in lora_layers for p in layer.parameters()],
        lr=lr
    )

    print("Training...")
    step = 0

    while step < steps:
        for latents, noise, noisy_latents, t, meta in dataloader:
            if step >= steps:
                break

            latents = latents.to(device)
            noise = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t = t.squeeze().to(device)

            meta_batch = {
                "dx": meta["dx"].to(device),
                "site": meta["site"].to(device),
                "sex": meta["sex"].to(device),
                "age": meta["age"].to(device),
            }

            cond = cond_encoder(meta_batch)
            cond = cond.unsqueeze(1).repeat(1, 77, 1)

            for _, module in unet.named_modules():
                if hasattr(module, "attn2"):
                    module.attn2.metadata_context = cond

            noise_pred = unet(noisy_latents, t,encoder_hidden_states=cond).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{steps}  Loss: {loss.item():.6f}")

            if step % 500 == 0:
                save_sample(unet, vae, cond_encoder, outdir, device, step)
                save_all_classes(unet, vae, cond_encoder, outdir, device, step)

    print("Saving weights...")
    torch.save(
        {
            "lora_layers": [l.state_dict() for l in lora_layers],
            "cond_encoder": cond_encoder.state_dict(),
        },
        os.path.join(outdir, "lora_numeric_condition.pth")
    )

    print("Training complete.")



if __name__ == "__main__":
    train()

