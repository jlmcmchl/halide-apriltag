# Halide-Accelerated AprilTag Detection

A performance-optimized implementation of AprilTag detection that uses [Halide](https://halide-lang.org/) to accelerate the adaptive thresholding pipeline. This project integrates Halide's high-performance image processing capabilities with the robust AprilTag 3 detection library.

## Overview

This project provides a hybrid approach to AprilTag detection:
- **Halide-accelerated thresholding**: The computationally intensive adaptive thresholding step is implemented using Halide, enabling efficient execution on multiple backends (CPU, CUDA, OpenCL, Hexagon)
- **Original sparse operations**: The rest of the AprilTag pipeline (quad detection, decoding, etc.) remains in the original C implementation for maximum compatibility

The result is a drop-in replacement for AprilTag that can leverage GPU acceleration and other specialized hardware while maintaining full compatibility with the original detection results.

## Features

- **Multi-backend support**: CPU, CUDA, OpenCL, and Hexagon (HVX) backends
- **Performance benchmarking**: Built-in timing harness to compare baseline vs Halide-accelerated performance
- **Full AprilTag 3 compatibility**: Supports all standard tag families (tag36h11, tag25h9, tag16h5, etc.)
- **Drop-in replacement**: Maintains identical detection results to the baseline implementation

## Requirements

- CMake 3.28 or higher
- C++17 compatible compiler
- Halide library (with appropriate backend support)
- Zlib
- Threads library (pthreads)

### Backend-Specific Requirements

- **CUDA**: CUDA Toolkit (for CUDA backend)
- **OpenCL**: OpenCL runtime (for OpenCL backend)
- **Hexagon**: Qualcomm Hexagon SDK (for Hexagon/HVX backend)

## Building

### Basic Build (CPU Backend)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Build with Specific Backend

```bash
# CUDA backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHALIDE_BACKEND=cuda
cmake --build build

# OpenCL backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHALIDE_BACKEND=opencl
cmake --build build

# Hexagon backend
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHALIDE_BACKEND=hexagon
cmake --build build
```

### Disable Halide (Baseline Only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_HALIDE=OFF
cmake --build build
```

## Usage

### Timing Harness

The `apriltag_timing` executable provides a comprehensive benchmarking tool:

```bash
./build/apriltag_timing [options] <image_path>
```

#### Options

- `-f, --family <name>`: Tag family (tag36h11, tag25h9, tag16h5) [default: tag36h11]
- `-d, --decimate <val>`: Decimate input image by this factor [default: 1.0]
- `-b, --blur <val>`: Apply blur to input; negative sharpens [default: 0.0]
- `-t, --threads <n>`: Number of CPU threads to use [default: 1]
- `-r, --runs <n>`: Repeat detections n times (averages timings) [default: 1]
- `--compare-halide`: Run both baseline and Halide hybrid detectors
- `--halide-only`: Run only the Halide hybrid detector
- `-h, --help`: Show help message

#### Examples

```bash
# Compare baseline vs Halide performance
./build/apriltag_timing --compare-halide -r 20 -t 4 image.jpg

# Run Halide-only detection
./build/apriltag_timing --halide-only -f tag36h11 image.jpg

# Benchmark with specific parameters
./build/apriltag_timing -d 1.0 -t 8 --runs 50 --compare-halide image.jpg
```

### Programmatic Usage

The Halide-accelerated thresholding can be enabled programmatically:

```c
#include "halide_threshold.h"
#include "apriltag/apriltag.h"

// Create detector
apriltag_detector_t *td = apriltag_detector_create();
td->use_halide = true;  // Enable Halide acceleration

// Create tag family
apriltag_family_t *tf = tag36h11_create();
apriltag_detector_add_family(td, tf);

// Load image
image_u8_t *im = image_u8_create_from_pnm("image.pnm");

// Detect tags (uses Halide thresholding if enabled)
zarray_t *detections = apriltag_detector_detect(td, im);

// Process detections...
```

## Performance

Benchmarking on select hardware: TODO

Use the timing harness with `--compare-halide` to benchmark performance on your specific hardware and workload.

## Project Structure

```
.
├── CMakeLists.txt              # Main build configuration
├── apriltag_timing.cpp         # Timing harness and benchmarking tool
├── halide_threshold.h          # Halide threshold pipeline header
├── halide_threshold.cpp        # Halide threshold pipeline implementation
├── threshold_generator.cpp     # Halide generator for adaptive thresholding
├── apriltag/                   # AprilTag 3 library (submodule/subdirectory)
└── build/                      # Build directory (generated)
```

## Architecture

### Threshold Pipeline

The adaptive thresholding algorithm is implemented as a Halide pipeline that:

1. Divides the image into tiles
2. Computes min/max values for each tile
3. Expands tile statistics to neighboring tiles
4. Applies adaptive thresholding based on local statistics

This pipeline is optimized by Halide's automatic scheduling and can target various backends.

### Integration

The Halide threshold pipeline integrates with AprilTag by:

1. Replacing the original `apriltag_quad_thresh` function when `use_halide` is enabled
2. Maintaining the same input/output interface
3. Preserving detection accuracy and compatibility

## License

This project includes the AprilTag 3 library, which is licensed under BSD-2-Clause. See `apriltag/LICENSE.md` for details.

## References

- [AprilTag 3](https://github.com/AprilRobotics/apriltag) - Original AprilTag library
- [Halide](https://halide-lang.org/) - High-performance image processing language
- [AprilTag Papers](https://april.eecs.umich.edu/papers/) - Research papers on AprilTag

## Contributing

Contributions are welcome! Please ensure that:

- Code follows the existing style
- Tests pass with `--compare-halide` to verify correctness
- Performance improvements are benchmarked appropriately

## Troubleshooting

### Halide Not Found

Ensure Halide is installed and available in your CMake path, or set `Halide_DIR`:

```bash
cmake -B build -DHalide_DIR=/path/to/halide/lib/cmake/Halide
```

### CUDA Backend Issues

- Verify CUDA Toolkit is installed
- Ensure CUDA-capable GPU is available
- Check CUDA driver compatibility

### Detection Mismatches

If detection results differ between baseline and Halide versions:

- Verify Halide backend is working correctly
- Check that image preprocessing is identical
- Review threshold parameters

Use `--compare-halide` to automatically verify detection accuracy.


## Building for Hexagon

This requires the Qualcomm Hexagon SDK, tested w/ v6.4.0.1. The toolchain in `x86-64-linux-toolchain.cmake` is tested, but may introduce unnecessary restrictions. Older versions of clang are likely to work, I'm just not using them.

This requires setting up an arm64 sysroot on the x86-64 host. This should include `zlib1g-dev`:
To create an ARM64 sysroot at `/opt/sysroots/arm64` and install `zlib1g-dev` for Ubuntu 24.04 (Noble), run:

```bash
# Install required tools (on your x86-64 Ubuntu/Debian host)
sudo apt-get update
sudo apt-get install debootstrap qemu-user-static binfmt-support

# Create the ARM64 sysroot for Ubuntu 24.04 (noble)
sudo debootstrap --arch=arm64 --foreign noble /opt/sysroots/arm64 http://ports.ubuntu.com/

# Copy qemu-aarch64-static into the sysroot to enable running ARM binaries via emulation
sudo cp /usr/bin/qemu-aarch64-static /opt/sysroots/arm64/usr/bin/

# Complete the second stage of debootstrap inside the sysroot
sudo chroot /opt/sysroots/arm64 /debootstrap/debootstrap --second-stage

# Update package list within the sysroot
sudo chroot /opt/sysroots/arm64 apt-get update

# Install zlib1g-dev in the sysroot
sudo chroot /opt/sysroots/arm64 apt-get install -y zlib1g-dev

# (Optional) Clean package cache inside the sysroot
sudo chroot /opt/sysroots/arm64 apt-get clean
```

**Note:**  
- Replace `noble` with `jammy` or another release if you want a different Ubuntu version.  
- Root privileges required for these steps.

### Cross-Compiling: Two-Stage Process

To cross-compile for Hexagon/ARM, perform the following two stages:

#### **1. Build the Host Tools**

*These run on your x86-64 (host) machine and generate code for the target.*

```bash
Halide_DIR=~/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HALIDE_Tools/2.6.01/ \
  cmake -G Ninja -S . -B build-host \
  -DCMAKE_BUILD_TYPE=Release \
  --toolchain x86-64-linux-toolchain.cmake \
  -Dapriltags_halide-halide_generators_ROOT=$PWD/build-host \
  -DHALIDE_BACKEND=hexagon

cmake --build build-host --target apriltags_halide-halide_generators
```

#### **2. Build Applications for the ARM Target**

*These are built to run on the ARM (target) hardware.*

```bash
Halide_DIR=~/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HALIDE_Tools/2.6.01/ \
  cmake -G Ninja -S . -B build-target \
  -DCMAKE_BUILD_TYPE=Release \
  --toolchain arm-64-linux-toolchain.cmake \
  -Dapriltags_halide-halide_generators_ROOT:FILEPATH=$PWD/build-host \
  -DHALIDE_BACKEND=hexagon

cmake --build build-target -v
```

**Tips:**
- Ensure `Halide_DIR` points to your Hexagon Halide tools directory.
- Paths such as `apriltag_halide-halide_generators_ROOT` may differ if you move or rename build directories.
- Use `-v` with `cmake --build` for verbose output during debugging.

#### **3. Sign the device!!**
```bash
# 1. Get your board's serial number (on the device)
cat /sys/devices/soc0/serial_number

# 2. On your x86 host with Hexagon SDK
cd $HEXAGON_SDK_ROOT

# 3. Find the signing script (testsig or elfsign)
find . -name "testsig" -o -name "elfsign" 2>/dev/null

# 4. Run the signature tool (choose one based on what you found):
# Option 1: Using elfsigner
python3 tools/elfsigner/elfsigner.py --testsig <serial_number>
# Option 2: Using testsig
python3 scripts/testsig.py <serial_number>
# (If unsure, check the documentation:)
ls docs/Tools_Signing*

# 5. Copy the generated signing library to the DSP filesystem
sudo cp testsig-*.so /usr/lib/rfsa/adsp/
```
/* 
Tip: Replace <serial_number> with the actual serial number from step 1.
Check the docs in $HEXAGON_SDK_ROOT/docs/ or tools/elfsigner/ for details if you have trouble.
*/

