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
            pipeline_->realize(outputs);
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
    void build() {
        Var x("x"), y("y");
        Var tx("tx"), ty("ty");

        Func padded = repeat_edge(input_, {{0, input_.width()}, {0, input_.height()}});

        const int tilesz = tile_size_;

        Halide::Expr tile_w = (input_.width() + tilesz - 1) / tilesz;
        Halide::Expr tile_h = (input_.height() + tilesz - 1) / tilesz;
        Halide::Expr clamped_tile_w = Halide::max(tile_w - 1, 0);
        Halide::Expr clamped_tile_h = Halide::max(tile_h - 1, 0);

        Halide::RDom tile_dom(0, tilesz, 0, tilesz);

        Func tile_min("metal_tile_min"), tile_max("metal_tile_max");
        Halide::Expr sx = Halide::min(tx * tilesz + tile_dom.x, input_.width() - 1);
        Halide::Expr sy = Halide::min(ty * tilesz + tile_dom.y, input_.height() - 1);
        tile_min(tx, ty) = Halide::cast<uint8_t>(255);
        tile_max(tx, ty) = Halide::cast<uint8_t>(0);
        tile_min(tx, ty) = Halide::min(tile_min(tx, ty), padded(sx, sy));
        tile_max(tx, ty) = Halide::max(tile_max(tx, ty), padded(sx, sy));

        Halide::RDom neigh_dom(-1, 3, -1, 3);
        Func neigh_min("metal_neigh_min"), neigh_max("metal_neigh_max");
        Halide::Expr ntx = Halide::clamp(tx + neigh_dom.x, 0, clamped_tile_w);
        Halide::Expr nty = Halide::clamp(ty + neigh_dom.y, 0, clamped_tile_h);
        neigh_min(tx, ty) = Halide::cast<uint8_t>(255);
        neigh_max(tx, ty) = Halide::cast<uint8_t>(0);
        neigh_min(tx, ty) = Halide::min(neigh_min(tx, ty), tile_min(ntx, nty));
        neigh_max(tx, ty) = Halide::max(neigh_max(tx, ty), tile_max(ntx, nty));

        Func min_px("metal_min_px"), max_px("metal_max_px");
        Halide::Expr tile_x = Halide::min(x / tilesz, clamped_tile_w);
        Halide::Expr tile_y = Halide::min(y / tilesz, clamped_tile_h);
        min_px(x, y) = neigh_min(tile_x, tile_y);
        max_px(x, y) = neigh_max(tile_x, tile_y);

        Func threshold("metal_threshold");
        Halide::Expr diff = Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
        Halide::Expr threshold_val = Halide::cast<uint8_t>(
            Halide::cast<int>(min_px(x, y)) + diff / 2);
        threshold(x, y) = Halide::cast<uint8_t>(
            Halide::select(diff < min_white_black_diff_,
                           127,
                           Halide::select(padded(x, y) > threshold_val, 255, 0)));

        Func binary_val("metal_binary_val");
        binary_val(x, y) = threshold(x, y);

        Func parent_init("metal_parent_init");
        Halide::Expr width_expr = input_.width();
        Halide::Expr height_expr = input_.height();
        Halide::Expr idx_expr = y * width_expr + x;
        Halide::Expr self_value = binary_val(x, y);
        parent_init(x, y) = Halide::cast<int32_t>(
            Halide::select(self_value == 127, -1, idx_expr));

        Func parent_hook("metal_parent_hook");
        Halide::Expr self_idx = parent_init(x, y);
        Halide::Expr valid = self_idx >= 0;

        auto neighbor_expr = [&](int dx, int dy) -> Halide::Expr {
            Halide::Expr nx = Halide::clamp(x + dx, 0, width_expr - 1);
            Halide::Expr ny = Halide::clamp(y + dy, 0, height_expr - 1);
            Halide::Expr n_val = binary_val(nx, ny);
            Halide::Expr n_idx = parent_init(nx, ny);
            return Halide::select(valid && (n_val == self_value), n_idx, self_idx);
        };

        Halide::Expr best_idx = self_idx;
        best_idx = Halide::min(best_idx, neighbor_expr(-1, 0));
        best_idx = Halide::min(best_idx, neighbor_expr(0, -1));
        best_idx = Halide::min(best_idx, neighbor_expr(-1, -1));
        best_idx = Halide::min(best_idx, neighbor_expr(1, -1));

        parent_hook(x, y) = Halide::select(valid, best_idx, Halide::cast<int32_t>(-1));

        std::vector<Func> parent_stages;
        parent_stages.push_back(parent_hook);

        for (int i = 0; i < pointer_jump_iterations_; ++i) {
            Func stage("metal_parent_stage_" + std::to_string(i));
            Halide::Expr current = parent_stages.back()(x, y);
            Halide::Expr current_valid = current >= 0;
            Halide::Expr safe_current = Halide::select(current_valid, current, 0);
            Halide::Expr parent_y = Halide::clamp(safe_current / width_expr, 0, height_expr - 1);
            Halide::Expr parent_x = Halide::clamp(safe_current - parent_y * width_expr, 0, width_expr - 1);
            Halide::Expr parent_idx = parent_stages.back()(parent_x, parent_y);
            parent_idx = Halide::select(parent_idx >= 0, parent_idx, safe_current);
            Halide::Expr merged = Halide::min(safe_current, parent_idx);
            stage(x, y) = Halide::select(current_valid, merged, Halide::cast<int32_t>(-1));
            parent_stages.push_back(stage);
        }

        Func final_parent = parent_stages.back();

        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        threshold.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 32, 16);

        for (Func &stage : parent_stages) {
            stage.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 32, 16);
        }

        tile_min.compute_root().parallel(ty);
        tile_min.update().parallel(ty);
        tile_max.compute_root().parallel(ty);
        tile_max.update().parallel(ty);
        neigh_min.compute_root().parallel(ty);
        neigh_min.update().parallel(ty);
        neigh_max.compute_root().parallel(ty);
        neigh_max.update().parallel(ty);
        min_px.compute_root().parallel(y);
        max_px.compute_root().parallel(y);
        parent_init.compute_root().parallel(y);

        pipeline_ = std::make_unique<Halide::Pipeline>(
            std::vector<Func>{threshold, final_parent});

        Halide::Target target = Halide::get_host_target();
        target.set_feature(Halide::Target::Metal);
        pipeline_->compile_jit(target);
    }

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
