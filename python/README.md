# Halide Python Starter

This folder mirrors the C++ sample using the Halide Python bindings. A GPU schedule is
applied only when the active Halide `Target` reports GPU support. Use the `HL_TARGET`
environment variable (e.g. `host-cuda`, `host-opencl`, `host-metal`) to select the
backend you want to experiment with.

## Setup

```bash
python3 -m venv ../.venv            # Already created by setup script
source ../.venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Example enabling CUDA
HL_TARGET=host-cuda python main.py
```

The script writes `gradient_py.png` demonstrating the realized pipeline.
