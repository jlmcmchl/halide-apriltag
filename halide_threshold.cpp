#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory.h>
#include <vector>

#include "HalideRuntime.h"
#ifdef USE_OPENCL
#include "HalideRuntimeOpenCL.h"
#endif

#ifdef USE_CUDA
#include <cuda.h>
#include "HalideRuntimeCuda.h"
#endif

#ifdef USE_HEXAGON
#include "HalideRuntimeHexagonHost.h"
#endif

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

#include "halide_threshold.h"

#include "adaptive_threshold.h"

template <typename F>
auto benchmark_with_return(std::vector<double> &times, F func)
    -> decltype(func()) {
  auto start = std::chrono::high_resolution_clock::now();
  auto result = func();
  auto end = std::chrono::high_resolution_clock::now();
  times.push_back(
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()) /
      1e6);
  return result;
}

void benchmark(std::vector<double> &times, std::function<void()> func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();
  times.push_back(
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()) /
      1e6);
}

void ThresholdPipeline::reset_stats() {
  copy_to_device_times_.clear();
  pipeline_times_.clear();
  copy_to_host_times_.clear();
}

void ThresholdPipeline::ensure_buffer_size(
    Halide::Runtime::Buffer<uint8_t, 2> &buffer, int width, int height) {
  // Check if buffer needs to be resized
  // Check if buffer is uninitialized (no raw_buffer) or size mismatch
  auto raw = buffer.raw_buffer();
  if (raw == nullptr || buffer.width() != width ||
      buffer.height() != height) {
#ifdef USE_HEXAGON
    // For Hexagon, use zero-copy buffers with aligned stride
    const int VLEN = 128;
    int stride_y = (width + (VLEN) - 1) & (-(VLEN));
    
    // Define the dimensions with aligned stride
    halide_dimension_t x_dim{0, width, 1};
    halide_dimension_t y_dim{0, height, stride_y};
    halide_dimension_t io_shape[2] = {x_dim, y_dim};
    
    // Allocate zero-copy buffer using ION memory
    // nullptr as first argument means device_malloc will allocate ION memory
    // that is visible to both host CPU and Hexagon DSP
    buffer = Halide::Runtime::Buffer<uint8_t>(nullptr, 2, io_shape);
    buffer.device_malloc(halide_hexagon_device_interface());
#else
    // Recreate buffer with new size for other backends
    buffer = Halide::Runtime::Buffer<uint8_t, 2>(width, height);
#endif
  }
}

void ThresholdPipeline::run(int min_white_black_diff, int tile_size,
                            image_u8_t *input_im, image_u8_t *output_im) {
#ifdef USE_HEXAGON
  // TODO: we should to turn this off when we're done, but that isn't happening rn
  halide_hexagon_power_hvx_on(nullptr);
  halide_hexagon_set_performance_mode(nullptr, halide_hexagon_power_max);
#endif
  // Resize buffers to match image dimensions if necessary
  ensure_buffer_size(input_buf_, input_im->width, input_im->height);
  ensure_buffer_size(output_buf_, output_im->width, output_im->height);

  copy_image_to_buffer(input_im, input_buf_);
  run_pipeline(min_white_black_diff, tile_size, input_buf_, output_buf_);
  copy_buffer_to_image(output_buf_, output_im);
}

/*
void ThresholdPipeline::create_input_buffer(int width, int height) {
#ifdef APRILTAG_HAVE_CUDA
    uint8_t *buf_ptr;
    size_t buf_size = sizeof(uint8_t) * width * height;
    auto result = cuMemHostAlloc((void**)&buf_ptr, buf_size,
CU_MEMHOSTALLOC_DEVICEMAP); if (result != CUDA_SUCCESS) { fprintf(stderr,
"cuMemHostAlloc failed: %d\n", result); throw std::runtime_error("cuMemHostAlloc
failed");
    }

    CUdeviceptr gpu_buf_ptr;
    result = cuMemHostGetDevicePointer(&gpu_buf_ptr, buf_ptr, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemHostGetDevicePointer failed: %d\n", result);
        throw std::runtime_error("cuMemHostGetDevicePointer failed");
    }

    input_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(buf_ptr,width,
height, "input_buffer"); input_buf_->device_wrap_native(Halide::DeviceAPI::CUDA,
gpu_buf_ptr, target_); #else input_buf_.allocate() #endif

}

void ThresholdPipeline::create_output_buffer(int width, int height) {
#ifdef APRILTAG_HAVE_CUDA
    uint8_t *buf_ptr;
    size_t buf_size = sizeof(uint8_t) * width * height;
    auto result = cuMemHostAlloc((void**)&buf_ptr, buf_size,
CU_MEMHOSTALLOC_DEVICEMAP); if (result != CUDA_SUCCESS) { fprintf(stderr,
"cuMemHostAlloc failed: %d\n", result); throw std::runtime_error("cuMemHostAlloc
failed");
    }

    CUdeviceptr gpu_buf_ptr;
    result = cuMemHostGetDevicePointer(&gpu_buf_ptr, buf_ptr, 0);
    if (result != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemHostGetDevicePointer failed: %d\n", result);
        throw std::runtime_error("cuMemHostGetDevicePointer failed");
    }
    output_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(buf_ptr,width,
height, "output_buffer");
    output_buf_->device_wrap_native(Halide::DeviceAPI::CUDA, gpu_buf_ptr,
target_); #else output_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(width,
height); #endif
}
*/

void ThresholdPipeline::copy_image_to_buffer(
    image_u8_t *image, Halide::Runtime::Buffer<uint8_t, 2> &buffer) {
  benchmark(copy_to_device_times_, [&]() {
    auto raw_buffer = buffer.raw_buffer();
    const int width = image->width;
    const int height = image->height;
    const int buffer_stride = raw_buffer->dim[1].stride;
    const int image_stride = image->stride;

    // Optimize: if strides match, do a single contiguous copy
    if (buffer_stride == image_stride && buffer_stride == width) {
      std::memcpy(raw_buffer->host, image->buf,
                  width * height * sizeof(uint8_t));
    } else {
      // Row-by-row copy with optimized pattern
      uint8_t *dst = raw_buffer->host;
      uint8_t *src = image->buf;
      for (int y = 0; y < height; y++) {
        std::memcpy(dst, src, width * sizeof(uint8_t));
        dst += buffer_stride;
        src += image_stride;
      }
    }
#ifdef USE_HEXAGON
    // TODO: Is this correct? Difficult to find documentation of memcopying to/from hexagon dma buffers
    // For Hexagon zero-copy buffers, no explicit copy needed
    // The buffer is already in ION memory visible to both CPU and DSP
    // Just mark device dirty so Halide knows to use device memory
    buffer.set_device_dirty();
#else
    buffer.set_host_dirty();
#ifdef USE_OPENCL
    buffer.copy_to_device(halide_opencl_device_interface());
#elif USE_CUDA
    buffer.copy_to_device(halide_cuda_device_interface());    
#endif
#endif
  });
}

void ThresholdPipeline::copy_buffer_to_image(
    Halide::Runtime::Buffer<uint8_t, 2> &buffer, image_u8_t *image) {
  benchmark(copy_to_host_times_, [&]() {
#ifdef USE_OPENCL
    buffer.copy_to_host();
#elif USE_CUDA
    buffer.copy_to_host();
#elif USE_HEXAGON
    // TODO: Is this correct? Difficult to find documentation of memcopying to/from hexagon dma buffers
    // For Hexagon zero-copy buffers, no explicit copy needed
    // The buffer is already in ION memory visible to both CPU and DSP
    // Just ensure device operations are complete
    // buffer.device_sync();
#endif
    auto raw_buffer = buffer.raw_buffer();
    const int width = image->width;
    const int height = image->height;
    const int buffer_stride = raw_buffer->dim[1].stride;
    const int image_stride = image->stride;

    // Optimize: if strides match, do a single contiguous copy
    if (buffer_stride == image_stride && buffer_stride == width) {
      std::memcpy(image->buf, raw_buffer->host,
                  width * height * sizeof(uint8_t));
    } else {
      // Row-by-row copy with optimized pattern
      uint8_t *dst = image->buf;
      uint8_t *src = raw_buffer->host;
      for (int y = 0; y < height; y++) {
        std::memcpy(dst, src, width * sizeof(uint8_t));
        dst += image_stride;
        src += buffer_stride;
      }
    }
  });
}

void ThresholdPipeline::run_pipeline(
    int min_white_black_diff, int tile_size,
    Halide::Runtime::Buffer<uint8_t, 2> &input_buf,
    Halide::Runtime::Buffer<uint8_t, 2> &output_buf) {

  auto result = benchmark_with_return(pipeline_times_, [&]() {
    int ret = adaptive_threshold(input_buf, min_white_black_diff, tile_size,
                                 output_buf);
    // For integrated GPUs: ensure pipeline completes synchronously
    // This prevents copy_to_host() from having to wait
    if (ret == halide_error_code_success) {
#ifdef USE_OPENCL
      output_buf.device_sync();
#elif USE_CUDA
      output_buf.device_sync();
#elif USE_HEXAGON
    // TODO: Is this correct? Difficult to find documentation of memcopying to/from hexagon dma buffers
      output_buf.device_sync();
#endif
    }
    return ret;
  });

  if (result != halide_error_code_success) {
    fprintf(stderr, "Halide runtime error: %d\n", result);
    return;
  }
}

ThresholdPipeline &get_pipeline() {
  static ThresholdPipeline pipeline;
  return pipeline;
}

extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td,
                                        image_u8_t *im) {
  if (im == nullptr || td == nullptr) {
    return nullptr;
  }

  ThresholdPipeline &pipeline = get_pipeline();

  image_u8_t *threshim =
      image_u8_create_alignment(im->width, im->height, im->stride);

  try {
    pipeline.run(td->qtp.min_white_black_diff, 4, im, threshim);
  } catch (const std::exception &e) {
    fprintf(stderr, "Unexpected exception: %s\n", e.what());
    image_u8_destroy(threshim);
    return nullptr;
  }

  return threshim;
}
