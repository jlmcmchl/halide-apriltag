cmake_minimum_required(VERSION 3.16)

project(apriltags_halide
        DESCRIPTION "AprilTag timing harness with optional Halide acceleration"
        LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set default build type to Release with debug info if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_HALIDE "Build the Halide-backed threshold pipeline" ON)

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

target_compile_definitions(apriltag_timing PRIVATE _GNU_SOURCE)

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
    if (CMAKE_VERSION VERSION_LESS 3.28)
        message(WARNING "Halide CMake package requires CMake >= 3.28. Disabling Halide support.")
    else()
        find_package(Halide CONFIG)

        if (Halide_FOUND)
            target_sources(apriltag_timing PRIVATE halide_threshold.cpp)
            target_link_libraries(apriltag_timing PRIVATE
                Halide::Halide
                Halide::Mullapudi2016
                Halide::Li2018
                Halide::Adams2019
                Halide::Anderson2021)
            target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_HALIDE=1)
            message(STATUS "Halide found - hybrid detector enabled")
        else()
            message(WARNING "Halide package not found. Re-run CMake with USE_HALIDE=OFF or provide Halide_DIR.")
        endif()
    endif()
endif()

# Find CUDA toolkit for CUDA headers and libraries
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
    # Add CUDA include directories for C++ files that include CUDA headers
    target_include_directories(apriltag_timing PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # Link against CUDA runtime and driver libraries
    # CUDA::cudart for CUDA runtime API
    # CUDA::cuda_driver for CUDA driver API (cuMemHostAlloc, cuMemHostGetDevicePointer_v2, etc.)
    target_link_libraries(apriltag_timing PRIVATE CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_CUDA=1)
else()
    message(WARNING "CUDA toolkit not found. CUDA support will be disabled.")
endif()

# target_compile_options(apriltag_timing PRIVATE -g -pg)
# target_link_options(apriltag_timing PRIVATE -pg)
