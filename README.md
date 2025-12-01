# Build a variety of z-image inference containers - last one might even work

```bash
podman build -t z-image-cpu .
```

```bash
podman build -t z-image-cuda:12.1 .
```

## Verify gpu access

```bash
docker run --gpus all -it z-image-cuda:12.1 nvidia-smi
podman run --device nvidia.com/gpu=all -it z-image-cuda:12.1 nvidia-smi
podman run --device nvidia.com/gpu=all -it z-image-cuda:12.1-with-flash-attn-with-repo
```

## Run with GPU access in Podman

```bash
podman run --device nvidia.com/gpu=all -v $(pwd)/outputs:/workspace z-image-cuda:12.1
```

## Or interactive mode

```bash
podman run --device nvidia.com/gpu=all -it z-image-cuda:12.1 bash
```

## Use volume mounts to preserve model files downloaded

```bash
podman run -it --name z_image_generator \
    --runtime nvidia \
    -v ./cache:/root/.cache/huggingface/hub \
    -v ./input:/workspace/input \
    -v ./output:/workspace/output \
    your_image_name python3 gene
```

```bash
podman run -it --name z_image_generator \
    --runtime nvidia \
    -v ./cache:/root/.cache/huggingface/hub \
    -v ./input:/workspace/input \
    -v ./output:/workspace/output \
    z-image-cuda:12.1-with-flash-attn-with-repo python3 generate.py bash
```

```bash
podman run -it --name z_image_generator \
    --runtime nvidia \
    -v ./cache:/root/.cache/huggingface/hub \
    -v ./input:/workspace/input \
    -v ./output:/workspace/output \
    z-image-for-cpu:latest python3 generate.py bash
```

```bash
podman run --device nvidia.com/gpu=all -v ./cache:/root/.cache/huggingface/hub -v ./input:/workspace/input -v ./output:/workspace/output -v $(pwd)/outputs:/workspace z-image-cuda:12.1
```

## Instructions to build and run the Docker container with Podman

## Build using **Dockerfile.cuda-with-pip-cache** (rename to Dockerfile)

```bash
podman build -t z-image-cuda:12.1 .
``

## Run

```bash
podman run --device nvidia.com/gpu=all -it -v ./cache:/root/.cache/huggingface/hub -v ./input:/workspace/input -v ./output:/workspace/output z-image-cuda:12.1
```

# Inside container, test GPU:
# nvidia-smi

# Run your script:
# python3 /workspace/generate.py