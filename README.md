
# Containerized Z-image inference with Nvidia GPU 12GB VRAM using sequential CPU offloading and fp32 VAE

## Instructions to build and run the Docker container with Podman

## Build

```bash
podman build -t z-image-cuda:12.1 .
``

## Run

```bash
podman run --device nvidia.com/gpu=all -it -v ./cache:/root/.cache/huggingface/hub -v ./input:/workspace/input -v ./output:/workspace/output z-image-cuda:12.1
```

## Inside container, test GPU

```bash
nvidia-smi
```

## Run your script in the container (this is how I typically play around with it)

```bash
cp /workspace/input/generate.py .

python3 generate.py
```
