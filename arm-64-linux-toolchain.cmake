# CMake toolchain file for ARM64 Linux (cross-compilation)
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=arm-64-linux-toolchain.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Use aarch64-linux-gnu cross-compilers
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++")
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc")

set(CMAKE_SYSROOT "/opt/sysroots/arm64")
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
