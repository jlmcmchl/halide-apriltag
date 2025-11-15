#ifndef HALIDE_THRESHOLD_H
#define HALIDE_THRESHOLD_H

#include <vector>
#include <cstdint>
#include <Halide.h>

#ifdef APRILTAG_HAVE_HALIDE

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

namespace {
    Halide::Target find_gpu_target()  {
        // Start with a target suitable for the machine you're running this on.
        Halide::Target target = Halide::get_host_target();
    #ifdef APRILTAG_HAVE_CUDA
        target = target.with_feature(Halide::Target::Feature::CUDA)
            .with_feature(Halide::Target::Feature::CUDACapability86);
    #endif
        return target;
    }
}

// ThresholdPipeline class declaration
// Full definition is in halide_threshold.cpp
class ThresholdPipeline {
public:
    // Public members for statistics access
    std::vector<double> copy_to_device_times_;
    std::vector<double> pipeline_times_;
    std::vector<double> copy_to_host_times_;

    ThresholdPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input"),
          min_white_black_diff_("min_white_black_diff") {}

    // Public methods
    void compile_once();
    void reset_stats();
    void prepare_buffers(int width, int height);
    void run(int min_white_black_diff, uint8_t *input_data, uint8_t *output_data, int width, int height);

private:
    void build();
    void create_input_buffer(int width, int height);
    void create_output_buffer(int width, int height);
    void copy_input_to_buffer(uint8_t *input_data, int width, int height);
    void copy_buffer_to_output(uint8_t *output_data, int width, int height);
    void run_pipeline(int min_white_black_diff, uint8_t *input_data, uint8_t *output_data, int width, int height);
    Halide::ImageParam input_;
    Halide::Param<int> min_white_black_diff_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    Halide::Target target_ = find_gpu_target();
    std::once_flag init_flag_;
    std::unique_ptr<Halide::Buffer<uint8_t>> input_buf_;
    std::unique_ptr<Halide::Buffer<uint8_t>> output_buf_;
};

// C++ interface - get the pipeline singleton
ThresholdPipeline &get_pipeline();

// C interface function
extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td, image_u8_t *im);

#endif // APRILTAG_HAVE_HALIDE

#endif // HALIDE_THRESHOLD_H


