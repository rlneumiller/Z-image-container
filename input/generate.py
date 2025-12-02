#!/usr/bin/env python3
import os
import inspect
import random
from pathlib import Path
import diffusers
import torch
from diffusers import ZImagePipeline

# Print diffusers version for debug
print("diffusers version:", diffusers.__version__)

print("Loading Z-Image-Turbo model...")

# Choose the correct keyword arg for dtype based on the installed diffusers signature.
# Prefer 'dtype' when available (newer versions) and fall back to 'torch_dtype' when needed.
def _get_dtype_kwarg(dtype_val):
    sig = inspect.signature(ZImagePipeline.from_pretrained)
    if "dtype" in sig.parameters:
        return {"dtype": dtype_val}
    if "torch_dtype" in sig.parameters:
        return {"torch_dtype": dtype_val}
    return {}

dtype_kw = _get_dtype_kwarg(torch.float16)
if dtype_kw:
    print("Using from_pretrained keyword:", list(dtype_kw.keys())[0])
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", **dtype_kw)
else:
    print("from_pretrained does not accept dtype/torch_dtype kwargs; loading without dtype and will cast modules manually")
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo")

# Use SEQUENTIAL offloading
# Do not move the entire pipeline to CUDA before this call
pipe.enable_sequential_cpu_offload(gpu_id=0)

# FORCE VAE to run in float32 to prevent black images
# This is the essential part of the fix.
print("Forcing VAE to float32 precision (to avoid artifacts)...")
pipe.vae.to(torch.float32)

# If from_pretrained did not allow dtype selection, cast modules we want to relieve GPU memory usage
if not dtype_kw:
    try:
        print("Casting unet/text_encoder to float16 to reduce memory (manual fallback)...")
        if getattr(pipe, "unet", None) is not None:
            pipe.unet.to(torch.float16)
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(torch.float16)
    except Exception as e:
        print("Manual casting failed:", e)

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
def get_next_filename(out_dir: Path, seed: int, padding: int = 3, max_attempts: int = 9999) -> Path:
    """Return the next available filename path in out_dir for a given seed.

    Filenames take the format: <seed>-<index>.png where index is zero-padded to 'padding' digits.
    The function checks existence and returns a path for the first available index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{seed}"
    for i in range(1, max_attempts + 1):
        filename = f"{base_name}-{i:0{padding}d}.png"
        candidate = out_dir / filename
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"All {max_attempts} filenames exist for seed {seed} in {out_dir}")

out_dir = Path("/workspace/output")
save_path = get_next_filename(out_dir, seed)
print(f"Saving image to {save_path}")
image.save(save_path)
print("Done!")

# podman run --device nvidia.com/gpu=all -it -v ./cache:/root/.cache/huggingface/hub -v ./input:/workspace/input -v ./output:/workspace/output z-image-cuda:12.1
