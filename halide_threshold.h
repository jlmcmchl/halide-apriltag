#ifndef HALIDE_THRESHOLD_H
#define HALIDE_THRESHOLD_H

#include "HalideBuffer.h"
#include "image_types.h"
#include <cstdint>
#include <vector>

#ifdef APRILTAG_HAVE_HALIDE

extern "C" {
#include "apriltag/apriltag.h"
}

// ThresholdPipeline class declaration
// Full definition is in halide_threshold.cpp
class ThresholdPipeline {
public:
  // Public members for statistics access
  std::vector<double> copy_to_device_times_;
  std::vector<double> pipeline_times_;
  std::vector<double> copy_to_host_times_;

  // Public methods
  void reset_stats();
  void run(int min_white_black_diff, int tile_size, image_u8_t *input_im,
           image_u8_t *output_im);

private:
  void ensure_buffer_size(Halide::Runtime::Buffer<uint8_t, 2> &buffer,
                          int width, int height);
  void copy_image_to_buffer(image_u8_t *image,
                            Halide::Runtime::Buffer<uint8_t, 2> &buffer);
  void copy_buffer_to_image(Halide::Runtime::Buffer<uint8_t, 2> &buffer,
                            image_u8_t *image);
  void run_pipeline(int min_white_black_diff, int tile_size,
                    Halide::Runtime::Buffer<uint8_t, 2> &input_buf,
                    Halide::Runtime::Buffer<uint8_t, 2> &output_buf);
  Halide::Runtime::Buffer<uint8_t, 2> input_buf_;
  Halide::Runtime::Buffer<uint8_t, 2> output_buf_;
};

// C++ interface - get the pipeline singleton
ThresholdPipeline &get_pipeline();

// C interface function
extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td,
                                        image_u8_t *im);

#endif // APRILTAG_HAVE_HALIDE

#endif // HALIDE_THRESHOLD_H
