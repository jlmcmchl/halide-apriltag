#include <Halide.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/unionfind.h"
}
#include "metal_basic_adaptive_threshold.h"

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;

namespace {

class MetalThresholdPipeline {
public:
    MetalThresholdPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "metal_input"),
          min_white_black_diff_("metal_min_white_black_diff"),
          tile_size_(4),
          pointer_jump_iterations_(8) {
    }

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    bool run(apriltag_detector_t *td,
             image_u8_t *im,
             image_u8_t **thresh_out,
             unionfind_t **uf_out) {
        if (!td || !im || !thresh_out || !uf_out) {
            return false;
        }

        compile_once();

        const int width = im->width;
        const int height = im->height;

        Halide::Buffer<uint8_t> input_buf(im->buf, width, height);
        auto *input_raw = input_buf.raw_buffer();
        input_raw->dim[0].stride = 1;
        input_raw->dim[1].stride = im->stride;
        input_raw->dim[0].min = 0;
        input_raw->dim[1].min = 0;

        Halide::Buffer<uint8_t> threshold_buf(width, height);
        auto *threshold_raw = threshold_buf.raw_buffer();
        threshold_raw->dim[0].stride = 1;
        threshold_raw->dim[1].stride = width;
        threshold_raw->dim[0].min = 0;
        threshold_raw->dim[1].min = 0;

        Halide::Buffer<int32_t> labels_buf(width, height);
        auto *labels_raw = labels_buf.raw_buffer();
        labels_raw->dim[0].stride = 1;
        labels_raw->dim[1].stride = width;
        labels_raw->dim[0].min = 0;
        labels_raw->dim[1].min = 0;

        input_.set(input_buf);
        min_white_black_diff_.set(td->qtp.min_white_black_diff);

        Halide::Realization outputs({threshold_buf, labels_buf});

        try {
            metal_basic_adaptive_threshold(input_buf.raw_buffer(), min_white_black_diff_.get(), tile_size_, pointer_jump_iterations_, threshold_buf.raw_buffer(), labels_buf.raw_buffer());
            threshold_buf.copy_to_host();
            labels_buf.copy_to_host();
        } catch (const Halide::RuntimeError &e) {
            fprintf(stderr, "Halide runtime error (Metal pipeline): %s\n", e.what());
            return false;
        } catch (const Halide::CompileError &e) {
            fprintf(stderr, "Halide compile error (Metal pipeline): %s\n", e.what());
            return false;
        } catch (const std::exception &e) {
            fprintf(stderr, "Unexpected Halide exception (Metal pipeline): %s\n", e.what());
            return false;
        }

        image_u8_t *threshim = image_u8_create_alignment(width, height, im->stride);
        auto *dst = threshim->buf;
        const auto *src = threshold_buf.data();
        for (int y = 0; y < height; ++y) {
            std::copy(src + y * width, src + (y + 1) * width, dst + y * threshim->stride);
        }

        unionfind_t *uf = unionfind_create(width * height);
        std::vector<uint32_t> counts(width * height, 0);

        const int32_t *labels_ptr = labels_buf.data();
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int32_t label = labels_ptr[y * width + x];
                if (label >= 0 && label < width * height) {
                    counts[label]++;
                }
            }
        }

        for (int idx = 0; idx < width * height; ++idx) {
            uint32_t count = counts[idx];
            if (count > 0) {
                uf->parent[idx] = idx;
                uf->size[idx] = count - 1;
            } else {
                uf->parent[idx] = idx;
                uf->size[idx] = 0;
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int32_t label = labels_ptr[y * width + x];
                if (label >= 0 && label < width * height) {
                    uf->parent[idx] = static_cast<uint32_t>(label);
                }
            }
        }

        *thresh_out = threshim;
        *uf_out = uf;

        return true;
    }

private:
    void build() {}

    ImageParam input_;
    Param<int> min_white_black_diff_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    std::once_flag init_flag_;
    const int tile_size_;
    const int pointer_jump_iterations_;
};

MetalThresholdPipeline &get_pipeline() {
    static MetalThresholdPipeline pipeline;
    return pipeline;
}

} // namespace

extern "C" bool halide_metal_threshold_and_label(apriltag_detector_t *td,
                                                 image_u8_t *im,
                                                 image_u8_t **thresh_out,
                                                 unionfind_t **uf_out)
{
    if (!thresh_out || !uf_out) {
        return false;
    }

    *thresh_out = nullptr;
    *uf_out = nullptr;

    if (!td || !im) {
        return false;
    }

    MetalThresholdPipeline &pipeline = get_pipeline();
    return pipeline.run(td, im, thresh_out, uf_out);
}
