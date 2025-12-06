import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoPipelineForText2Image
import torch

device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)
    
def sample_base(prompt, num_inference_steps, guidance_scale):
    pipeline_base = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
    ).to(device)

    pipeline_base.safety_checker=None
    
    out = pipeline_base(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image_base = out.images[0]
    plt.imsave('./results/prompt based/base_model.png', image_base)
    return image_base

def sample_finetuned(prompt, LoRA_weights, num_inference_steps, guidance_scale):
    pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    pipeline.load_lora_weights(
        LoRA_weights,
        weight_name="pytorch_lora_weights.safetensors"
    )
    

    image = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    plt.imsave('./results/prompt based/finetuned_model.png', image)
    return image

def display_results(base_image, finetuned_image):
    print('\nimages saved to results/prompt based')
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].imshow(finetuned_image)
    axes[0].set_title('After Fine-Tuning')
    axes[0].axis('off')
    axes[1].imshow(base_image)
    axes[1].set_title('Before Fine-Tuning')
    axes[1].axis('off')
    plt.suptitle('Sampling Comparison', fontweight = 'bold', fontsize=14)
    plt.tight_layout()
    plt.show()