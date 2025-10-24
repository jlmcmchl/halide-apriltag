cmake_minimum_required(VERSION 3.16)
project(apriltag_halide LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_binary.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_density.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_lut.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_binary.h
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge.h
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_density.h
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_lut.h
)

add_custom_command(
    OUTPUT ${GENERATED_LIBS}
    COMMAND apriltag_generator
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS apriltag_generator
    COMMENT "Generating Halide AOT libraries"
)

add_custom_target(build_apriltag_libs DEPENDS ${GENERATED_LIBS})

add_library(apriltag_binary STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_binary PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_binary.a)
add_dependencies(apriltag_binary build_apriltag_libs)

add_library(apriltag_edge STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_edge PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge.a)
add_dependencies(apriltag_edge build_apriltag_libs)

add_library(apriltag_density STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_density PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_density.a)
add_dependencies(apriltag_density build_apriltag_libs)

add_library(apriltag_lut STATIC IMPORTED GLOBAL)
set_target_properties(apriltag_lut PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/apriltag_lut.a)
add_dependencies(apriltag_lut build_apriltag_libs)

add_executable(apriltag_runner src/main.cpp)
add_dependencies(apriltag_runner build_apriltag_libs)
target_include_directories(apriltag_runner PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)

# Link against the Halide runtime, generated pipelines, and image IO helpers.
# Link libraries directly to avoid duplicate symbol issues
target_link_libraries(apriltag_runner PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_binary.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_edge.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_density.a
    ${CMAKE_CURRENT_BINARY_DIR}/apriltag_lut.a
    Halide::Runtime
    Halide::ImageIO
    PNG::PNG
    JPEG::JPEG
    ${METAL_LIBRARY}
    ${FOUNDATION_LIBRARY}
    ${OpenCV_LIBS}
)

# Add linker flags to handle duplicate symbols
target_link_options(apriltag_runner PRIVATE
    "-Wl,-dead_strip"
)

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()
