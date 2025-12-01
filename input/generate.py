#!/usr/bin/env python3
import random
import torch
from diffusers import ZImagePipeline

# Load the pipeline with float16 for better memory efficiency, using 'dtype' instead of 'torch_dtype'
print("Loading Z-Image-Turbo model...")
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    dtype=torch.float16,  # Corrected parameter name
)

# Use SEQUENTIAL offloading
# Do not move the entire pipeline to CUDA before this call
pipe.enable_sequential_cpu_offload(gpu_id=0)

# FORCE VAE to run in float32 to prevent black images
# This is the essential part of the fix.
print("Forcing VAE to float32 precision...")
pipe.vae.to(torch.float32)

print("Model loaded with sequential CPU offloading and fp32 VAE")

# Example prompt
prompt = "A Modern Nvidia GPU with 3 fans engulfed in flames."


print("Generating image...")
# Ensure the generator is using 'cuda'
seed = random.randint(0, 2**32 - 1)

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]  # Changed back to [0] for saving a single image file

print(f"Saving image to /workspace/output/output-seed({seed}).png")
image.save(f"/workspace/output/output-seed({seed}).png")
print("Done!")
