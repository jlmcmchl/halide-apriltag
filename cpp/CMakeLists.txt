cmake_minimum_required(VERSION 3.16)
project(apriltag_halide LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggressive optimization for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math")

# Make sure we're in Release mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Halide provides a CMake config package. Set Halide_DIR if CMake cannot find it automatically.
find_package(Halide REQUIRED)

# Find image I/O libraries for Halide::ImageIO
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

add_halide_generator(threshold.generator SOURCES src/apriltag_halide_generator.cpp)

add_halide_library(greyscale_and_adaptive_threshold FROM threshold.generator
                    TARGETS host
#                    AUTOSCHEDULER Halide::Adams2019
#                    AUTOSCHEDULER Halide::Anderson2021
                    GENERATOR greyscale_and_adaptive_threshold_gen
                    FEATURES opencl
                    )


add_executable(apriltag_runner src/main.cpp)
target_include_directories(apriltag_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Link against the Halide runtime, generated pipelines, and image IO helpers.
# Link libraries directly to avoid duplicate symbol issues
target_link_libraries(apriltag_runner PRIVATE
    greyscale_and_adaptive_threshold
    Halide::Tools
    Halide::Runtime
    Halide::ImageIO
    PNG::PNG
    JPEG::JPEG
)

find_package(OpenCL)

if (OpenCL_FOUND)
    target_link_libraries(apriltag_runner PRIVATE OpenCL::OpenCL)
    target_include_directories(apriltag_runner PRIVATE ${OpenCL_INCLUDE_DIR})
    target_compile_definitions(apriltag_runner PRIVATE -DUSE_OPENCL)
    
endif()
