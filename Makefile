# AprilTag Detector Timing Utility Makefile
# This Makefile builds the apriltag_timing executable

# Compiler and flags
CXX := clang++
CC := clang
CXXFLAGS := -std=c++11 -O2 -Wall -Wextra
CFLAGS := -std=c99 -O2 -Wall -Wextra -fPIC
LDFLAGS := -lm -lpthread -lz

USE_HALIDE ?= 1
HALIDE_DIR ?= $(shell brew --prefix halide 2>/dev/null)

ifeq ($(HALIDE_DIR),)
HALIDE_DIR := /opt/homebrew/opt/halide
endif

ifeq ($(USE_HALIDE),1)
HALIDE_INC := -I$(HALIDE_DIR)/include
HALIDE_LIBS := -L$(HALIDE_DIR)/lib -Wl,-rpath,$(HALIDE_DIR)/lib -lHalide
HALIDE_DEFS := -DAPRILTAG_HAVE_HALIDE
else
HALIDE_INC :=
HALIDE_LIBS :=
HALIDE_DEFS :=
endif

# Directories
APRILTAG_DIR := apriltag
APRILTAG_COMMON_DIR := $(APRILTAG_DIR)/common
BUILD_DIR := build
BIN_DIR := bin

# Include directories
INCLUDES := -I. -I$(APRILTAG_DIR) -I$(APRILTAG_COMMON_DIR) $(HALIDE_INC)

# Source files
APRILTAG_SOURCES := \
    $(APRILTAG_DIR)/apriltag.c \
    $(APRILTAG_DIR)/apriltag_pose.c \
    $(APRILTAG_DIR)/apriltag_quad_thresh.c \
    $(APRILTAG_DIR)/tag36h11.c \
    $(APRILTAG_DIR)/tag25h9.c \
    $(APRILTAG_DIR)/tag16h5.c \
    $(APRILTAG_DIR)/tag36h10.c \
    $(APRILTAG_DIR)/tag25h9.c \
    $(APRILTAG_DIR)/tagCircle21h7.c \
    $(APRILTAG_DIR)/tagCircle49h12.c \
    $(APRILTAG_DIR)/tagCustom48h12.c \
    $(APRILTAG_DIR)/tagStandard41h12.c \
    $(APRILTAG_DIR)/tagStandard52h13.c

COMMON_SOURCES := \
    $(APRILTAG_COMMON_DIR)/g2d.c \
    $(APRILTAG_COMMON_DIR)/homography.c \
    $(APRILTAG_COMMON_DIR)/image_u8.c \
    $(APRILTAG_COMMON_DIR)/image_u8x3.c \
    $(APRILTAG_COMMON_DIR)/image_u8x4.c \
    $(APRILTAG_COMMON_DIR)/image_u8_parallel.c \
    $(APRILTAG_COMMON_DIR)/matd.c \
    $(APRILTAG_COMMON_DIR)/pam.c \
    $(APRILTAG_COMMON_DIR)/pjpeg.c \
    $(APRILTAG_COMMON_DIR)/pjpeg-idct.c \
    $(APRILTAG_COMMON_DIR)/pnm.c \
    $(APRILTAG_COMMON_DIR)/string_util.c \
    $(APRILTAG_COMMON_DIR)/svd22.c \
    $(APRILTAG_COMMON_DIR)/time_util.c \
    $(APRILTAG_COMMON_DIR)/unionfind.c \
    $(APRILTAG_COMMON_DIR)/workerpool.c \
    $(APRILTAG_COMMON_DIR)/zarray.c \
    $(APRILTAG_COMMON_DIR)/zhash.c \
    $(APRILTAG_COMMON_DIR)/zmaxheap.c \
    $(APRILTAG_COMMON_DIR)/getopt.c \
    $(APRILTAG_COMMON_DIR)/pthreads_cross.c

TIMING_SOURCES := apriltag_timing.cpp

ifeq ($(USE_HALIDE),1)
TIMING_SOURCES += halide_threshold.cpp
endif

# Object files
APRILTAG_OBJS := $(patsubst $(APRILTAG_DIR)/%.c,$(BUILD_DIR)/$(APRILTAG_DIR)/%.o,$(APRILTAG_SOURCES))
COMMON_OBJS := $(patsubst $(APRILTAG_COMMON_DIR)/%.c,$(BUILD_DIR)/$(APRILTAG_COMMON_DIR)/%.o,$(COMMON_SOURCES))
TIMING_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TIMING_SOURCES))

ALL_OBJS := $(APRILTAG_OBJS) $(COMMON_OBJS) $(TIMING_OBJS)

# Targets
TARGET := $(BIN_DIR)/apriltag_timing

# Phony targets
.PHONY: all clean help

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(ALL_OBJS) | $(BIN_DIR)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(HALIDE_LIBS)
	@echo "Build complete: $@"

# Compile C++ files
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	@mkdir -p $(@D)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(HALIDE_DEFS) $(INCLUDES) -c $< -o $@

# Compile C files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@mkdir -p $(@D)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) $(HALIDE_DEFS) $(INCLUDES) -c $< -o $@

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Help target
help:
	@echo "AprilTag Detector Timing Utility - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the apriltag_timing executable (default)"
	@echo "  clean   - Remove all build artifacts"
	@echo "  help    - Display this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build the project"
	@echo "  make clean        # Clean build files"
	@echo "  make help         # Show this help"
	@echo ""
	@echo "After building, run:"
	@echo "  ./bin/apriltag_timing <image_path> [options]"
	@echo ""
	@echo "Example:"
	@echo "  ./bin/apriltag_timing apriltag/test/data/33369213973_9d9bb4cc96_c.jpg"
	@echo "  ./bin/apriltag_timing apriltag/test/data/33369213973_9d9bb4cc96_c.jpg -t 4 -d 1.0"
