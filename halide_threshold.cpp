#include <Halide.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <chrono>
#include <vector>

#ifdef APRILTAG_HAVE_CUDA
#include <cuda.h>
#endif

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::RVar;
using Halide::BoundaryConditions::repeat_edge;
using Halide::TailStrategy;
using Halide::MemoryType;

Halide::Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Halide::Target target = Halide::get_host_target();
#ifdef APRILTAG_HAVE_CUDA
    target = target.with_feature(Halide::Target::Feature::CUDA)
        .with_feature(Halide::Target::Feature::CUDACapability86);
#endif
    return target;
}

class ThresholdPipeline {
public:
    // Public members first to match forward declaration in apriltag_timing.cpp
    std::vector<double> copy_to_device_times_;
    std::vector<double> pipeline_times_;
    std::vector<double> copy_to_host_times_;

    ThresholdPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input"),
          min_white_black_diff_("min_white_black_diff") {}

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    void create_input_buffer(int width, int height) {
#ifdef APRILTAG_HAVE_CUDA
        uint8_t *buf_ptr;
        size_t buf_size = sizeof(uint8_t) * width * height;
        auto result = cuMemHostAlloc((void**)&buf_ptr, buf_size, CU_MEMHOSTALLOC_DEVICEMAP);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemHostAlloc failed: %d\n", result);
            throw std::runtime_error("cuMemHostAlloc failed");
        }

        CUdeviceptr gpu_buf_ptr;
        result = cuMemHostGetDevicePointer(&gpu_buf_ptr, buf_ptr, 0);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemHostGetDevicePointer failed: %d\n", result);
            throw std::runtime_error("cuMemHostGetDevicePointer failed");
        }

        input_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(buf_ptr,width, height, "input_buffer");
        input_buf_->device_wrap_native(Halide::DeviceAPI::CUDA, gpu_buf_ptr, target_);
#else
        input_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(width, height);
        auto *input_raw = input_buf_->raw_buffer();
        input_raw->dim[0].stride = 1;
        input_raw->dim[1].stride = width;  // Use width as stride for owned buffer
        input_raw->dim[0].min = 0;
        input_raw->dim[1].min = 0;
#endif
    }

    void create_output_buffer(int width, int height) {
#ifdef APRILTAG_HAVE_CUDA
        uint8_t *buf_ptr;
        size_t buf_size = sizeof(uint8_t) * width * height;
        auto result = cuMemHostAlloc((void**)&buf_ptr, buf_size, CU_MEMHOSTALLOC_DEVICEMAP);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemHostAlloc failed: %d\n", result);
            throw std::runtime_error("cuMemHostAlloc failed");
        }

        CUdeviceptr gpu_buf_ptr;
        result = cuMemHostGetDevicePointer(&gpu_buf_ptr, buf_ptr, 0);
        if (result != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemHostGetDevicePointer failed: %d\n", result);
            throw std::runtime_error("cuMemHostGetDevicePointer failed");
        }
        output_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(buf_ptr,width, height, "output_buffer");
        output_buf_->device_wrap_native(Halide::DeviceAPI::CUDA, gpu_buf_ptr, target_);
#else
        output_buf_ = std::make_unique<Halide::Buffer<uint8_t>>(width, height);
        auto *output_raw = output_buf_->raw_buffer();
        output_raw->dim[0].stride = 1;
        output_raw->dim[1].stride = width;  // Use width as stride for owned buffer
        output_raw->dim[0].min = 0;
        output_raw->dim[1].min = 0;
#endif
    }

    void prepare_buffers(uint8_t *input_data, int width, int height, int input_stride) {
        if (!input_buf_) {
            create_input_buffer(width, height);
        }

        bool input_changed = !input_buf_ || 
                            input_buf_->width() != width ||
                            input_buf_->height() != height;

        if (input_changed) {
            create_input_buffer(width, height);
        }

        for (int y = 0; y < height; y++) {
            std::memcpy(input_buf_->data() + y * input_buf_->stride(1),
                       input_data + y * input_stride,
                       width);
        }

        if (!output_buf_) {
            create_output_buffer(width, height);
        }

        bool output_changed = !output_buf_ || 
                              output_buf_->width() != width ||
                              output_buf_->height() != height;

        if (output_changed) {
            create_output_buffer(width, height);
        }
    }

    void copy_output_to_buffer(uint8_t *output_data, int width, int height, int output_stride) {
        if (!output_buf_) {
            return;
        }

        for (int y = 0; y < height; y++) {
            std::memcpy(output_data + y * output_stride,
                       output_buf_->data() + y * output_buf_->stride(1),
                       width);
        }
    }

    void run(int min_white_black_diff, uint8_t *output_data, int width, int height, int output_stride) {
        compile_once();

        if (!target_ || !pipeline_ || !input_buf_ || !output_buf_) {
            fprintf(stderr, "Error: Pipeline not properly initialized\n");
            return;
        }

        auto copy_to_device_start = std::chrono::high_resolution_clock::now();
        if (target_->has_gpu_feature()) {
            input_buf_->set_host_dirty();
            input_buf_->copy_to_device(target_->get_required_device_api(), *target_);
        }
        input_.set(*input_buf_);

        auto copy_to_device_end = std::chrono::high_resolution_clock::now();
        copy_to_device_times_.push_back(
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(copy_to_device_end - copy_to_device_start).count()
            ) / 1e6  // Convert nanoseconds to milliseconds
        );

        min_white_black_diff_.set(min_white_black_diff);

        // Realize to cached output device buffer or host buffer
        if (target_->has_gpu_feature()) {
            pipeline_->realize(*output_buf_);
        } else {
            pipeline_->realize(*output_buf_);
        }
        auto pipeline_end = std::chrono::high_resolution_clock::now();
        pipeline_times_.push_back(
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(pipeline_end - copy_to_device_end).count()
            ) / 1e6  // Convert nanoseconds to milliseconds
        );

        auto copy_to_host_start = std::chrono::high_resolution_clock::now();
        if (target_->has_gpu_feature()) {
            output_buf_->set_device_dirty();
            output_buf_->copy_to_host();
        }
        copy_output_to_buffer(output_data, width, height, output_stride);
        auto copy_to_host_end = std::chrono::high_resolution_clock::now();
        copy_to_host_times_.push_back(
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(copy_to_host_end - copy_to_host_start).count()
            ) / 1e6  // Convert nanoseconds to milliseconds
        );
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

        try {
            Halide::Target target = find_gpu_target();
            printf("Target: %s\n", target.to_string().c_str());
            target_ = std::make_unique<Halide::Target>(target);


#ifdef AUTO_SCHEDULE
            Halide::load_plugin("libautoschedule_mullapudi2016.so");
            Halide::load_plugin("libautoschedule_li2018.so");
            Halide::load_plugin("libautoschedule_adams2019.so");
            Halide::load_plugin("libautoschedule_anderson2021.so");

            printf("AutoScheduling...\n");
            input_.set_estimates({{0, 1600}, {0, 1304}});
            min_white_black_diff_.set_estimate(5);
            output.set_estimates({{0, 1600}, {0, 1304}});

            pipeline_ = std::make_unique<Halide::Pipeline>(output);
            // Parameter specs:
            // Mullapudi2016: https://github.com/halide/Halide/blob/main/src/autoschedulers/mullapudi2016/AutoSchedule.cpp#L24
            // Li2018: https://github.com/halide/Halide/blob/main/src/autoschedulers/li2018/GradientAutoscheduler.cpp#L11
            // Adams2019: https://github.com/halide/Halide/blob/main/src/autoschedulers/adams2019/CostModel.h#L19
            // Anderson2021: https://github.com/halide/Halide/blob/main/src/autoschedulers/anderson2021/CostModel.h#L18
            // ^ gpu only
            auto results = pipeline_->apply_autoscheduler(target_, {"Adams2019", {{"parallelism", "32"}, {"beam_size", "64"}}});
            printf("Autoscheduler results: %s\n", results.schedule_source.c_str());
#else
            if (target.has_gpu_feature()) {
                Var _0(padded.get_schedule().dims()[0].var);
                Var _0i("_0i");
                Var _1(padded.get_schedule().dims()[1].var);
                Var _1i("_1i");
                Var _1ii("_1ii");
                Var tx(neigh_min.get_schedule().dims()[0].var);
                Var txi("txi");
                Var txii("txii");
                Var ty(neigh_min.get_schedule().dims()[1].var);
                Var tyi("tyi");
                Var x(output.get_schedule().dims()[0].var);
                Var xi("xi");
                Var xii("xii");
                Var y(output.get_schedule().dims()[1].var);
                Var yi("yi");
                Var yii("yii");
                RVar r10_x(tile_min.update(0).get_schedule().dims()[0].var);
                RVar r10_y(tile_min.update(0).get_schedule().dims()[1].var);
                RVar r23_x(neigh_min.update(0).get_schedule().dims()[0].var);
                RVar r23_y(neigh_min.update(0).get_schedule().dims()[1].var);
                Var ty_serial_outer("ty_serial_outer");
                Var tyi_serial_outer("tyi_serial_outer");
                Var _1i_serial_outer("_1i_serial_outer");
                Var txi_serial_outer("txi_serial_outer");
                Var _0i_serial_outer("_0i_serial_outer");
                Var tx_serial_outer("tx_serial_outer");
                Var xi_serial_outer("xi_serial_outer");
                output
                    .split(x, x, xi, 64, TailStrategy::ShiftInwards)
                    .split(y, y, yi, 4, TailStrategy::ShiftInwards)
                    .split(xi, xi, xii, 4, TailStrategy::ShiftInwards)
                    .split(yi, yi, yii, 4, TailStrategy::ShiftInwards)
                    .unroll(xii)
                    .unroll(yii)
                    .compute_root()
                    .reorder(xii, yii, xi, yi, x, y)
                    .gpu_blocks(x)
                    .gpu_blocks(y)
                    .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
                    .gpu_threads(xi);
                neigh_min
                    .compute_at(output, x)
                    .reorder(tx, ty)
                    .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
                    .gpu_threads(tx);
                neigh_min.update(0)
                    .reorder(r23_x, r23_y, tx, ty)
                    .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
                    .gpu_threads(tx);
                tile_min
                    .split(tx, tx, txi, 32, TailStrategy::RoundUp)
                    .split(ty, ty, tyi, 2, TailStrategy::RoundUp)
                    .compute_root()
                    .reorder(txi, tyi, tx, ty)
                    .gpu_blocks(tx)
                    .gpu_blocks(ty)
                    .split(txi, txi_serial_outer, txi, 32, TailStrategy::GuardWithIf)
                    .gpu_threads(txi)
                    .split(tyi, tyi_serial_outer, tyi, 2, TailStrategy::GuardWithIf)
                    .gpu_threads(tyi);
                tile_min.update(0)
                    .split(tx, tx, txi, 32, TailStrategy::RoundUp)
                    .split(ty, ty, tyi, 2, TailStrategy::RoundUp)
                    .reorder(r10_x, r10_y, txi, tyi, tx, ty)
                    .gpu_blocks(tx)
                    .gpu_blocks(ty)
                    .split(txi, txi_serial_outer, txi, 32, TailStrategy::GuardWithIf)
                    .gpu_threads(txi)
                    .split(tyi, tyi_serial_outer, tyi, 2, TailStrategy::GuardWithIf)
                    .gpu_threads(tyi);
                neigh_max
                    .split(tx, tx, txi, 32, TailStrategy::RoundUp)
                    .split(ty, ty, tyi, 8, TailStrategy::RoundUp)
                    .split(txi, txi, txii, 2, TailStrategy::RoundUp)
                    .unroll(txii)
                    .compute_root()
                    .reorder(txii, txi, tyi, tx, ty)
                    .gpu_blocks(tx)
                    .gpu_blocks(ty)
                    .split(txi, txi_serial_outer, txi, 16, TailStrategy::GuardWithIf)
                    .gpu_threads(txi)
                    .split(tyi, tyi_serial_outer, tyi, 8, TailStrategy::GuardWithIf)
                    .gpu_threads(tyi);
                neigh_max.update(0)
                    .split(tx, tx, txi, 32, TailStrategy::RoundUp)
                    .split(ty, ty, tyi, 8, TailStrategy::RoundUp)
                    .split(txi, txi, txii, 2, TailStrategy::GuardWithIf)
                    .unroll(txii)
                    .reorder(txii, r23_x, r23_y, txi, tyi, tx, ty)
                    .gpu_blocks(tx)
                    .gpu_blocks(ty)
                    .split(txi, txi_serial_outer, txi, 16, TailStrategy::GuardWithIf)
                    .gpu_threads(txi)
                    .split(tyi, tyi_serial_outer, tyi, 8, TailStrategy::GuardWithIf)
                    .gpu_threads(tyi);
                tile_max
                    .compute_at(neigh_max, tx)
                    .reorder(tx, ty)
                    .split(tx, tx_serial_outer, tx, 34, TailStrategy::GuardWithIf)
                    .gpu_threads(tx)
                    .split(ty, ty_serial_outer, ty, 10, TailStrategy::GuardWithIf)
                    .gpu_threads(ty);
                tile_max.update(0)
                    .reorder(r10_x, r10_y, tx, ty)
                    .split(tx, tx_serial_outer, tx, 34, TailStrategy::GuardWithIf)
                    .gpu_threads(tx)
                    .split(ty, ty_serial_outer, ty, 10, TailStrategy::GuardWithIf)
                    .gpu_threads(ty);
                padded
                    .split(_0, _0, _0i, 32, TailStrategy::ShiftInwards)
                    .split(_1, _1, _1i, 8, TailStrategy::ShiftInwards)
                    .split(_1i, _1i, _1ii, 4, TailStrategy::ShiftInwards)
                    .unroll(_1ii)
                    .compute_root()
                    .reorder(_1ii, _0i, _1i, _0, _1)
                    .gpu_blocks(_0)
                    .gpu_blocks(_1)
                    .split(_0i, _0i_serial_outer, _0i, 32, TailStrategy::GuardWithIf)
                    .gpu_threads(_0i)
                    .split(_1i, _1i_serial_outer, _1i, 2, TailStrategy::GuardWithIf)
                    .gpu_threads(_1i);
            } else {
                Var _0(padded.get_schedule().dims()[0].var);
                Var _0i("_0i");
                Var _1(padded.get_schedule().dims()[1].var);
                Var tx(neigh_min.get_schedule().dims()[0].var);
                Var txi("txi");
                Var ty(neigh_min.get_schedule().dims()[1].var);
                Var x(output.get_schedule().dims()[0].var);
                Var xi("xi");
                Var y(output.get_schedule().dims()[1].var);
                Var yi("yi");
                Var yii("yii");
                RVar r10_x(tile_min.update(0).get_schedule().dims()[0].var);
                RVar r10_y(tile_min.update(0).get_schedule().dims()[1].var);
                RVar r23_x(neigh_min.update(0).get_schedule().dims()[0].var);
                RVar r23_y(neigh_min.update(0).get_schedule().dims()[1].var);
                output
                    .split(y, y, yi, 44, TailStrategy::ShiftInwards)
                    .split(yi, yi, yii, 11, TailStrategy::ShiftInwards)
                    .split(x, x, xi, 16, TailStrategy::ShiftInwards)
                    .vectorize(xi)
                    .compute_root()
                    .reorder({xi, x, yii, yi, y})
                    .parallel(y);
                min_px
                    .store_in(MemoryType::Stack)
                    .split(x, x, xi, 16, TailStrategy::RoundUp)
                    .vectorize(xi)
                    .compute_at(output, x)
                    .reorder({xi, x, y});
                neigh_min
                    .store_in(MemoryType::Stack)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .compute_at(output, yi)
                    .reorder({txi, tx, ty});
                neigh_min.update(0)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .reorder({txi, r23_x, r23_y, tx, ty});
                tile_min
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .compute_at(output, y)
                    .reorder({txi, tx, ty});
                tile_min.update(0)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .reorder({txi, r10_x, r10_y, tx, ty});
                max_px
                    .store_in(MemoryType::Stack)
                    .split(x, x, xi, 16, TailStrategy::RoundUp)
                    .vectorize(xi)
                    .compute_at(output, x)
                    .reorder({xi, x, y});
                neigh_max
                    .store_in(MemoryType::Stack)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .compute_at(output, yi)
                    .reorder({txi, tx, ty});
                neigh_max.update(0)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .reorder({txi, r23_x, r23_y, tx, ty});
                tile_max
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .compute_at(output, y)
                    .reorder({txi, tx, ty});
                tile_max.update(0)
                    .split(tx, tx, txi, 16, TailStrategy::RoundUp)
                    .vectorize(txi)
                    .reorder({txi, r10_x, r10_y, tx, ty});
                padded
                    .split(_0, _0, _0i, 16, TailStrategy::ShiftInwards)
                    .vectorize(_0i)
                    .compute_at(output, y)
                    .reorder({_0i, _0, _1});
            }
            
            pipeline_ = std::make_unique<Halide::Pipeline>(output);
#endif // AUTO_SCHEDULE
            
            pipeline_->compile_jit(target);
        } catch (const Halide::CompileError &e) {
            fprintf(stderr, "Halide GPU JIT compile error: %s\n", e.what());
            throw;
        } catch (const Halide::InternalError &e) {
            fprintf(stderr, "Halide GPU JIT internal error: %s\n", e.what());
            throw;
        }
    }

    ImageParam input_;
    Param<int> min_white_black_diff_;
    std::unique_ptr<Halide::Pipeline> pipeline_;
    std::unique_ptr<Halide::Target> target_;
    std::once_flag init_flag_;
    std::unique_ptr<Halide::Buffer<uint8_t>> input_buf_;
    std::unique_ptr<Halide::Buffer<uint8_t>> output_buf_;
};

ThresholdPipeline &get_pipeline() {
    static ThresholdPipeline pipeline;
    return pipeline;
}

extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td, image_u8_t *im)
{
    if (im == nullptr || td == nullptr) {
        return nullptr;
    }

    ThresholdPipeline &pipeline = get_pipeline();
    pipeline.compile_once();

    image_u8_t *threshim = image_u8_create_alignment(im->width, im->height, im->stride);

    try {
        pipeline.prepare_buffers(im->buf, im->width, im->height, im->stride);
        pipeline.run(td->qtp.min_white_black_diff, threshim->buf, im->width, im->height, im->stride);
        // Debugging - used to affirm 'correctness' for different schedules / algorithms
        // image_u8_write_pnm(threshim, "debug_output_buf.pgm");
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
