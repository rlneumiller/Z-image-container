# Containerized Z-image inference with Nvidia GPU 12GB VRAM using sequential CPU offloading and fp32 VAE
Built with podman on debian (13) trixie with a Nvidia 4070 Super with 12GB VRAM and 64GB RAM
You'll need to have the nvidia container toolkit installed your host to be able to utilize your GPU from within the container

* When using the GPU I generate a 1024x1024 image in ~1m
* When using CPU I generate the same image in ~10m (useful to test before getting GPU working)

## Instructions to build and run the `Docker` podman container with Podman

## Build

There are two Dockerfiles included in this repository: the GPU-enabled `Dockerfile` (CUDA 12.1) and the CPU-only `Dockerfile.cpu-10-minute-image-generation`.

Build the default GPU image (the Dockerfile in this folder):

```bash
# Build the GPU-enabled image (default Dockerfile)
podman build -t z-image-cuda:12.1 .
```

If you want to explicitly specify the Dockerfile filename (same effect):

```bash
# Explicitly set which Dockerfile to use
podman build -f Dockerfile -t z-image-cuda:12.1 .
```

Build the CPU-only image (faster to test that the Dockerfile builds as expected):

```bash
podman build -f Dockerfile.cpu-10-minute-image-generation -t z-image-cpu:latest .
```

## Run

GPU (NVIDIA device required - uses the `nvidia` container toolkit):

```bash
podman run --device nvidia.com/gpu=all -it \
  -v ./cache:/root/.cache/huggingface/hub \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  z-image-cuda:12.1
```

CPU (no GPU required):

```bash
podman run -it \
  -v ./cache:/root/.cache/huggingface/hub \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  z-image-cpu:latest

Note: The CPU Dockerfile sets the default command to `bash`. If you want the container to run `generate.py` automatically, you have two options:

* Bake the script into the image by uncommenting `COPY input/generate.py /workspace/generate.py` in `Dockerfile.cpu-10-minute-image-generation` before building, or
* Provide an explicit command to run `generate.py` when starting the container (for example, if you've mounted `./input` at runtime):

```bash
podman run -it \
  -v ./cache:/root/.cache/huggingface/hub \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  z-image-cpu:latest \
  bash -lc "cp /workspace/input/generate.py . && python3 generate.py"
```

Or you can run the container with the default `bash` CMD and run the script interactively:

```bash
podman run -it -v ./cache:/root/.cache/huggingface/hub -v ./input:/workspace/input -v ./output:/workspace/output z-image-cpu:latest
# Inside the container
cp /workspace/input/generate.py .
python3 generate.py
```

## Inside container, test GPU

```bash
nvidia-smi
```

## Running the generation script inside the container

The `Dockerfile`s include an optional `COPY input/generate.py /workspace/generate.py` line (commented out by default). That means, by default, the script is not baked into the image and you should mount your `./input` directory into the container at `/workspace/input` instead.

There are two ways to run the script:

* Mount `./input` at runtime and run the script from the mounted location (recommended for iteration):

```bash
# Mounting input and running the script
podman run -it \
  -v ./cache:/root/.cache/huggingface/hub \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  z-image-cuda:12.1 \
  bash -lc "cp /workspace/input/generate.py . && python3 generate.py"
```

* Or, if you prefer to bake the script into the image (note: uncomment the `COPY` line in the Dockerfile), build the image and run it directly:

```bash
# Uncomment `COPY input/generate.py /workspace/generate.py` in the Dockerfile,
# then build and run the image. The script will be available at /workspace/generate.py.
podman build -t z-image-cuda:12.1 .
podman run --device nvidia.com/gpu=all -it -v ./cache:/root/.cache/huggingface/hub z-image-cuda:12.1
# Inside: python3 /workspace/generate.py
```

Tip: Volume mounts override files baked into the image. If you both copy the script into the image and mount `./input`, the mounted file will hide the baked-in one.
