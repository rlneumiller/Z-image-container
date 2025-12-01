# Containerized Z-image inference with Nvidia GPU 12GB VRAM using sequential CPU offloading and fp32 VAE
Built with podman on debian (13) trixie with a Nvidia 4070 Super with 12GB VRAM and 64GB RAM
You'll need to have the nvidia container toolkit installed your host to be able to utilize your GPU from within the container

* When using the GPU I generate a 1024x1024 image in ~10s
* When using CPU I generate the same image in ~10m (useful to test before getting GPU working)

## Instructions to build and run the `Docker` podman container with Podman

## Build

```bash
# In the folder with Dockerfile
podman build -t z-image-cuda:12.1 .
```

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
