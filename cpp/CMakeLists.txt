cmake_minimum_required(VERSION 3.16)
project(apriltag_halide LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)

# Halide provides a CMake config package. Set Halide_DIR if CMake cannot find it automatically.
find_package(Halide REQUIRED)

# Find image I/O libraries for Halide::ImageIO
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

add_executable(apriltag_generator src/apriltag_generator.cpp)
target_link_libraries(apriltag_generator PRIVATE Halide::Halide)

set(GENERATED_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.h
)

add_custom_command(
    OUTPUT ${GENERATED_LIBS}
    COMMAND apriltag_generator
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS apriltag_generator
    COMMENT "Generating Halide AOT libraries"
)

add_custom_target(build_apriltag_libs DEPENDS ${GENERATED_LIBS})

add_library(apriltag_edge_detect STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_edge_detect PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge_detect.a)
add_dependencies(apriltag_edge_detect build_apriltag_libs)

add_executable(apriltag_runner src/main.cpp)
add_dependencies(apriltag_runner build_apriltag_libs)
target_include_directories(apriltag_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)

add_executable(apriltag_gen src/apriltag_halide_generator.cpp)
target_link_libraries(apriltag_gen PRIVATE Halide::Generator)

add_halide_library(greyscale FROM apriltag_gen
                    TARGETS host
                    AUTOSCHEDULER Halide::Adams2019
#                    AUTOSCHEDULER Halide::Anderson2021
                    GENERATOR greyscale_gen)

add_halide_library(adaptive_thredhold FROM apriltag_gen
                    TARGETS host
                    AUTOSCHEDULER Halide::Adams2019
#                    AUTOSCHEDULER Halide::Anderson2021
                    GENERATOR adaptive_threshold_gen)


# Link against the Halide runtime, generated pipelines, and image IO helpers.
# Link libraries directly to avoid duplicate symbol issues
target_link_libraries(apriltag_runner PRIVATE
    adaptive_thredhold
    greyscale
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
