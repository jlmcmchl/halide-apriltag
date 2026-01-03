cmake_minimum_required(VERSION 3.28)

project(apriltags_halide
        DESCRIPTION "AprilTag timing harness with optional Halide acceleration"
        LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release with debug info if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_HALIDE "Build the Halide-backed threshold pipeline" ON)
option(USE_CUDA "Build the CUDA-backed threshold pipeline" OFF)

set(APRILTAG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/apriltag)
set(APRILTAG_COMMON_DIR ${APRILTAG_DIR}/common)

set(APRILTAG_SOURCES
    ${APRILTAG_DIR}/apriltag.c
    ${APRILTAG_DIR}/apriltag_pose.c
    ${APRILTAG_DIR}/apriltag_quad_thresh.c
    ${APRILTAG_DIR}/tag16h5.c
    ${APRILTAG_DIR}/tag25h9.c
    ${APRILTAG_DIR}/tag36h10.c
    ${APRILTAG_DIR}/tag36h11.c
    ${APRILTAG_DIR}/tagCircle21h7.c
    ${APRILTAG_DIR}/tagCircle49h12.c
    ${APRILTAG_DIR}/tagCustom48h12.c
    ${APRILTAG_DIR}/tagStandard41h12.c
    ${APRILTAG_DIR}/tagStandard52h13.c
)

set(APRILTAG_COMMON_SOURCES
    ${APRILTAG_COMMON_DIR}/g2d.c
    ${APRILTAG_COMMON_DIR}/getopt.c
    ${APRILTAG_COMMON_DIR}/homography.c
    ${APRILTAG_COMMON_DIR}/image_u8.c
    ${APRILTAG_COMMON_DIR}/image_u8_parallel.c
    ${APRILTAG_COMMON_DIR}/image_u8x3.c
    ${APRILTAG_COMMON_DIR}/image_u8x4.c
    ${APRILTAG_COMMON_DIR}/matd.c
    ${APRILTAG_COMMON_DIR}/pam.c
    ${APRILTAG_COMMON_DIR}/pjpeg-idct.c
    ${APRILTAG_COMMON_DIR}/pjpeg.c
    ${APRILTAG_COMMON_DIR}/pnm.c
    ${APRILTAG_COMMON_DIR}/pthreads_cross.c
    ${APRILTAG_COMMON_DIR}/string_util.c
    ${APRILTAG_COMMON_DIR}/svd22.c
    ${APRILTAG_COMMON_DIR}/time_util.c
    ${APRILTAG_COMMON_DIR}/unionfind.c
    ${APRILTAG_COMMON_DIR}/workerpool.c
    ${APRILTAG_COMMON_DIR}/zarray.c
    ${APRILTAG_COMMON_DIR}/zhash.c
    ${APRILTAG_COMMON_DIR}/zmaxheap.c
)

add_executable(apriltag_timing
    apriltag_timing.cpp
    ${APRILTAG_SOURCES}
    ${APRILTAG_COMMON_SOURCES}
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_err)
if (ipo_ok)
    set_property(TARGET apriltag_timing PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO/LTO not supported: ${ipo_err}")
  endif()

target_include_directories(apriltag_timing
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${APRILTAG_DIR}
        ${APRILTAG_COMMON_DIR}
)

find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

target_link_libraries(apriltag_timing
    PRIVATE
        Threads::Threads
        ZLIB::ZLIB
)

find_library(MATH_LIBRARY m)
if (MATH_LIBRARY)
    target_link_libraries(apriltag_timing PRIVATE ${MATH_LIBRARY})
endif()

if (USE_HALIDE)
    find_package(Halide REQUIRED)

    add_halide_generator(threshold.generator threshold_generator.cpp)
    add_halide_library(adaptive_threshold FROM threshold.generator
        TARGETS cmake
        GENERATOR adaptive_threshold_gen
    )

    target_include_directories(apriltag_timing PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_sources(apriltag_timing PRIVATE halide_threshold.cpp)
    target_link_libraries(apriltag_timing PRIVATE Halide::Runtime adaptive_threshold)
    target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_HALIDE=1)
    message(STATUS "Halide found - hybrid detector enabled")
endif()

# Find CUDA toolkit for CUDA headers and libraries
if (USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
    # Add CUDA include directories for C++ files that include CUDA headers
    target_include_directories(apriltag_timing PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # Link against CUDA runtime and driver libraries
    # CUDA::cudart for CUDA runtime API
    # CUDA::cuda_driver for CUDA driver API (cuMemHostAlloc, cuMemHostGetDevicePointer_v2, etc.)
    target_link_libraries(apriltag_timing PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_CUDA=1)
endif() 

# target_compile_options(apriltag_timing PRIVATE -g -pg)
# target_link_options(apriltag_timing PRIVATE -pg)
