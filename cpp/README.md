# Halide C++ Starter

This directory contains a tiny Halide pipeline that can be scheduled on a GPU. The GPU
schedule is only applied if the chosen Halide `Target` reports GPU support (e.g. by
setting `HL_TARGET=host-cuda`). Otherwise, the example falls back to a CPU schedule.

## Prerequisites

1. A Halide distribution that provides the CMake config package (usually located at
   `share/Halide/cmake`). Set `Halide_DIR` if CMake cannot locate it automatically.
2. A functional GPU backend in your Halide build (CUDA, Metal, OpenCL, etc.).

## Configure & Build

```bash
mkdir -p build
cmake -S . -B build -DHalide_DIR=/path/to/halide/share/Halide/cmake
cmake --build build

cmake -S cpp -B cpp/build && cmake --build cpp/build

./cpp/build/halide_gpu_starter
```

## Run

```bash
# Example using CUDA. Replace with the GPU target supported on your system.
HL_TARGET=host-cuda ./build/halide_gpu_starter
```

The executable writes `gradient.png` showcasing the generated Halide pipeline output.
