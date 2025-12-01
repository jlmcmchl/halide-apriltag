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

# Find OpenCV
find_package(OpenCV REQUIRED)


add_executable(apriltag_runner src/main.cpp)
target_include_directories(apriltag_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR})


add_executable(apriltag_gen src/apriltag_halide_generator.cpp)
target_link_libraries(apriltag_gen PRIVATE Halide::Generator)


add_halide_library(greyscale_and_adaptive_threshold FROM apriltag_gen
                    TARGETS host
                    AUTOSCHEDULER Halide::Adams2019
#                    AUTOSCHEDULER Halide::Anderson2021
                    GENERATOR greyscale_and_adaptive_threshold_gen)

# Link against the Halide runtime, generated pipelines, and image IO helpers.
# Link libraries directly to avoid duplicate symbol issues
target_link_libraries(apriltag_runner PRIVATE
    greyscale_and_adaptive_threshold
    Halide::Tools
    Halide::Runtime
    Halide::ImageIO
    PNG::PNG
    JPEG::JPEG
#    ${METAL_LIBRARY}
#    ${FOUNDATION_LIBRARY}
)

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()
