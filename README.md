# Halide GPU Starter Kits

This repository contains parallel C++ and Python starter projects for experimenting with
Halide GPU pipelines.

- `cpp/` hosts a CMake-based executable that builds a simple gradient pipeline and
  schedules it on the GPU if the chosen Halide `Target` supports one.
- `python/` provides the same pipeline using the Halide Python bindings.

## Prerequisites

1. Download a Halide binary release or build it from source. Ensure the GPU backend you
   need (CUDA, OpenCL, Metal, etc.) is enabled.
2. Point `HL_TARGET` to the target string you want to experiment with, such as
   `host-cuda` or `host-opencl`.
3. Set `Halide_DIR` to the `share/Halide/cmake` folder inside your Halide distribution if
   CMake cannot auto-detect it.

## Python Environment

A virtual environment has been created at `.venv`. Activate it and install dependencies
before running the Python sample:

```bash
source .venv/bin/activate
pip install -r python/requirements.txt
```

## Building the C++ Sample

```bash
cd cpp
mkdir -p build
cmake -S . -B build -DHalide_DIR=/path/to/halide/share/Halide/cmake
cmake --build build
HL_TARGET=host-cuda ./build/halide_gpu_starter
```

Both the C++ and Python samples emit a gradient PNG (`gradient.png` / `gradient_py.png`)
so you can quickly verify that scheduling worked.
