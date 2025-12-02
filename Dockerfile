# Use NVIDIA CUDA 12.1 base image with Ubuntu
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from source (required for Z-Image)
RUN pip3 install git+https://github.com/huggingface/diffusers

# Example to install a specific version of diffusers for other use cases (not Z-image compatible)
#RUN pip3 install diffusers==0.31.0

# Install additional dependencies with numpy<2 for compatibility
RUN pip3 install transformers accelerate safetensors pillow "numpy<2" huggingface-hub

# Set working directory
WORKDIR /workspace

# Clone Z-Image repository (optional)
# RUN git clone https://github.com/Tongyi-MAI/Z-Image.git /workspace/Z-Image

# Create a directory for model cache
RUN mkdir -p /root/.cache/huggingface

# When not using volume mounts copy the local inference script into the container
# COPY input/generate.py /workspace/generate.py

# And ensure the script is executable
#RUN chmod +x /workspace/generate.py

# Default command
CMD ["bash"]

####################################################################3
# Instructions to build and run the Docker container with Podman
####################################################################3

# Build
# podman build -t z-image-cuda:12.1 .

# Run
# podman run --device nvidia.com/gpu=all -it \
#   -v ./cache:/root/.cache/huggingface/hub \
#   -v ./input:/workspace/input \
#   -v ./output:/workspace/output \
#   z-image-cuda:12.1

# Inside container, test GPU:
# nvidia-smi

# Run your script:
# python3 /workspace/generate.py