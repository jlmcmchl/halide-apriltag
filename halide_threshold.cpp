#include <Halide.h>
#include <cstdio>
#include <memory>
#include <mutex>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;

namespace {

Halide::Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Halide::Target target = Halide::get_host_target();
    // return target;

    std::vector<Halide::Target::Feature> features_to_try;
    if (target.os == Halide::Target::Windows) {
        // Try D3D12 first; if that fails, try OpenCL.
        if (sizeof(void*) == 8) {
            // D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.push_back(Halide::Target::D3D12Compute);
        }
        features_to_try.push_back(Halide::Target::OpenCL);
    } else if (target.os == Halide::Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Halide::Target::Metal);
    } else {
        features_to_try.push_back(Halide::Target::OpenCL);
    }
    // Uncomment the following lines to also try CUDA:
    features_to_try.push_back(Halide::Target::CUDA);

    for (Halide::Target::Feature f : features_to_try) {
        Halide::Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}

class ThresholdPipeline {
public:
    ThresholdPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input"),
          min_white_black_diff_("min_white_black_diff") {}

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    void run(const Halide::Buffer<uint8_t> &input_buf,
             int min_white_black_diff,
             Halide::Buffer<uint8_t> &output_buf) {
        compile_once();

        input_.set(input_buf);
        min_white_black_diff_.set(min_white_black_diff);

        pipeline_->realize(output_buf);
    }

private:
    void build() {
        Var x("x"), y("y");

        Func padded = repeat_edge(input_, {{0, input_.width()}, {0, input_.height()}});

        Var tx("tx"), ty("ty");
        const int tilesz = 4;

        Halide::Expr tile_w = Halide::max(1, input_.width() / tilesz);
        Halide::Expr tile_h = Halide::max(1, input_.height() / tilesz);

        Halide::Expr clamped_tile_w = tile_w - 1;
        Halide::Expr clamped_tile_h = tile_h - 1;

        Halide::Expr tile_limit_x = tile_w * tilesz;
        Halide::Expr tile_limit_y = tile_h * tilesz;

        Halide::RDom tile_dom(0, tilesz, 0, tilesz);

        Func tile_min("tile_min"), tile_max("tile_max");
        Halide::Expr sx = Halide::clamp(tx * tilesz + tile_dom.x, 0, input_.width() - 1);
        Halide::Expr sy = Halide::clamp(ty * tilesz + tile_dom.y, 0, input_.height() - 1);
        tile_min(tx, ty) = Halide::cast<uint8_t>(255);
        tile_max(tx, ty) = Halide::cast<uint8_t>(0);
        tile_min(tx, ty) = Halide::min(tile_min(tx, ty), padded(sx, sy));
        tile_max(tx, ty) = Halide::max(tile_max(tx, ty), padded(sx, sy));

        Halide::RDom neigh_dom(-1, 3, -1, 3);
        Func neigh_min("neigh_min"), neigh_max("neigh_max");
        Halide::Expr ntx = Halide::clamp(tx + neigh_dom.x, 0, clamped_tile_w);
        Halide::Expr nty = Halide::clamp(ty + neigh_dom.y, 0, clamped_tile_h);
        neigh_min(tx, ty) = Halide::cast<uint8_t>(255);
        neigh_max(tx, ty) = Halide::cast<uint8_t>(0);
        neigh_min(tx, ty) = Halide::min(neigh_min(tx, ty), tile_min(ntx, nty));
        neigh_max(tx, ty) = Halide::max(neigh_max(tx, ty), tile_max(ntx, nty));

        Func min_px("min_px"), max_px("max_px");
        Halide::Expr tile_x = Halide::clamp(
            Halide::select(x < tile_limit_x, x / tilesz, clamped_tile_w),
            0, clamped_tile_w);
        Halide::Expr tile_y = Halide::clamp(
            Halide::select(y < tile_limit_y, y / tilesz, clamped_tile_h),
            0, clamped_tile_h);
        min_px(x, y) = neigh_min(tile_x, tile_y);
        max_px(x, y) = neigh_max(tile_x, tile_y);

        Func output("halide_threshold_output");
        Halide::Expr diff = Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
        Halide::Expr threshold = Halide::cast<uint8_t>(
            Halide::cast<int>(min_px(x, y)) + diff / 2);
        output(x, y) = Halide::cast<uint8_t>(
            Halide::select(diff < min_white_black_diff_,
                           127,
                           Halide::select(padded(x, y) > threshold, 255, 0)));

        // Scheduling tuned for CPU parallelism; GPU scheduling can be
        // layered on later if desired.
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        output.compute_root().tile(x, y, xo, yo, xi, yi, 64, 32)
              .parallel(yo)
              .vectorize(xi, 16);

        tile_min.compute_root().parallel(ty).vectorize(tx, 16);
        tile_min.update().parallel(ty);

        tile_max.compute_root().parallel(ty).vectorize(tx, 16);
        tile_max.update().parallel(ty);

        neigh_min.compute_root().parallel(ty).vectorize(tx, 16);
        neigh_min.update().parallel(ty);

        neigh_max.compute_root().parallel(ty).vectorize(tx, 16);
        neigh_max.update().parallel(ty);

        pipeline_ = std::make_unique<Halide::Pipeline>(output);
        Halide::Target target = Halide::get_host_target();
        pipeline_->compile_jit(target);
    }

    ImageParam input_;
    Param<int> min_white_black_diff_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    std::once_flag init_flag_;
};

ThresholdPipeline &get_pipeline() {
    static ThresholdPipeline pipeline;
    return pipeline;
}

} // namespace

extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td, image_u8_t *im)
{
    if (im == nullptr || td == nullptr) {
        return nullptr;
    }

    ThresholdPipeline &pipeline = get_pipeline();
    pipeline.compile_once();

    Halide::Buffer<uint8_t> input_buf(im->buf, im->width, im->height);
    auto *input_raw = input_buf.raw_buffer();
    input_raw->dim[0].stride = 1;
    input_raw->dim[1].stride = im->stride;
    input_raw->dim[0].min = 0;
    input_raw->dim[1].min = 0;

    image_u8_t *threshim = image_u8_create_alignment(im->width, im->height, im->stride);
    Halide::Buffer<uint8_t> output_buf(threshim->buf, threshim->width, threshim->height);
    auto *output_raw = output_buf.raw_buffer();
    output_raw->dim[0].stride = 1;
    output_raw->dim[1].stride = threshim->stride;
    output_raw->dim[0].min = 0;
    output_raw->dim[1].min = 0;

    try {
        pipeline.run(input_buf, td->qtp.min_white_black_diff, output_buf);
    } catch (const Halide::RuntimeError &e) {
        fprintf(stderr, "Halide runtime error: %s\n", e.what());
        image_u8_destroy(threshim);
        return nullptr;
    } catch (const Halide::CompileError &e) {
        fprintf(stderr, "Halide compile error: %s\n", e.what());
        image_u8_destroy(threshim);
        return nullptr;
    } catch (const std::exception &e) {
        fprintf(stderr, "Unexpected Halide exception: %s\n", e.what());
        image_u8_destroy(threshim);
        return nullptr;
    }

    return threshim;
}
