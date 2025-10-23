cmake_minimum_required(VERSION 3.16)

project(apriltags_halide
        DESCRIPTION "AprilTag timing harness with optional Halide acceleration"
        LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_HALIDE "Build the Halide-backed threshold pipeline" ON)
set(HALIDE_ROOT "" CACHE PATH "Root path to a Halide installation (used when auto-detection fails)")

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
    set(_halide_enabled FALSE)

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
        find_package(Halide CONFIG QUIET)
        if (Halide_FOUND)
            target_sources(apriltag_timing PRIVATE halide_threshold.cpp)
            target_link_libraries(apriltag_timing PRIVATE Halide::Halide)
            target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_HALIDE=1)
            set(_halide_enabled TRUE)
            message(STATUS "Halide found via CMake package - hybrid detector enabled")
        endif()
    endif()

    if (NOT _halide_enabled)
        # Manual discovery fallback (works with common package layouts, e.g. Homebrew, source builds)
        set(_halide_search_paths
            ${HALIDE_ROOT}
            $ENV{HALIDE_DIR}
            /opt/homebrew/opt/halide
            /usr/local/opt/halide
            /usr/local
            /opt/halide
        )

        find_path(HALIDE_INCLUDE_DIR
            NAMES Halide.h
            HINTS ${_halide_search_paths}
            PATH_SUFFIXES include
        )

        find_library(HALIDE_LIBRARY
            NAMES Halide
            HINTS ${_halide_search_paths}
            PATH_SUFFIXES lib lib64
        )

        if (HALIDE_INCLUDE_DIR AND HALIDE_LIBRARY)
            target_sources(apriltag_timing PRIVATE halide_threshold.cpp)
            target_include_directories(apriltag_timing PRIVATE ${HALIDE_INCLUDE_DIR})
            target_link_libraries(apriltag_timing PRIVATE ${HALIDE_LIBRARY})
            target_compile_definitions(apriltag_timing PRIVATE APRILTAG_HAVE_HALIDE=1)
            set(_halide_enabled TRUE)
            message(STATUS "Halide located at: include=${HALIDE_INCLUDE_DIR}, lib=${HALIDE_LIBRARY}")
        endif()
    endif()

    if (NOT _halide_enabled)
        message(WARNING "Halide not found. Re-run CMake with -DUSE_HALIDE=OFF or set HALIDE_ROOT to your Halide install.")
    endif()
endif()
