#include <Halide.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
}

#include "basic_adaptive_threshold.h"

using Halide::Func;
using Halide::ImageParam;
using Halide::Param;
using Halide::Var;
using Halide::BoundaryConditions::repeat_edge;

struct halide_candidate_region {
    int32_t x0;
    int32_t y0;
    int32_t x1;
    int32_t y1;
    float avg_gradient;
    int32_t edge_pixels;
};

namespace {

class ThresholdPipeline {
public:
    ThresholdPipeline()
        : input_(Halide::type_of<uint8_t>(), 2, "input"),
          min_white_black_diff_("min_white_black_diff"),
          tile_size_(4),
          roi_tile_size_(16), //down from 32
          enable_roi_filter_(false) {
        if (const char *env = std::getenv("APRILTAG_HALIDE_ROI")) {
            enable_roi_filter_ = std::atoi(env) != 0;
        }
    }

    void compile_once() {
        std::call_once(init_flag_, [&]() { build(); });
    }

    void run(Halide::Buffer<uint8_t> &input_buf,
             int min_white_black_diff,
             Halide::Buffer<uint8_t> &output_buf) {
        compile_once();

        input_.set(input_buf);
        min_white_black_diff_.set(min_white_black_diff);

        const int width = output_buf.width();
        const int height = output_buf.height();

        tile_width_ = (width + roi_tile_size_ - 1) / roi_tile_size_;
        tile_height_ = (height + roi_tile_size_ - 1) / roi_tile_size_;

        Halide::Buffer<int32_t> tile_edge_buf(tile_width_, tile_height_);
        Halide::Buffer<int32_t> tile_fg_buf(tile_width_, tile_height_);
        basic_adaptive_threshold(input_buf.raw_buffer(), min_white_black_diff, tile_size_, roi_tile_size_, output_buf.raw_buffer(), tile_edge_buf.raw_buffer(), tile_fg_buf.raw_buffer());
        output_buf.copy_to_host();
        tile_edge_buf.copy_to_host();
        tile_fg_buf.copy_to_host();

        if (enable_roi_filter_) {
            compute_candidates_and_mask(output_buf, tile_edge_buf, tile_fg_buf);
        } else {
            candidate_regions_.clear();
        }
    }

    const std::vector<halide_candidate_region> &candidate_regions() const {
        return candidate_regions_;
    }

private:
    void build() {}

    void compute_candidates_and_mask(Halide::Buffer<uint8_t> &output_buf,
                                     const Halide::Buffer<int32_t> &tile_edge_buf,
                                     const Halide::Buffer<int32_t> &tile_fg_buf) {
        const int width = output_buf.width();
        const int height = output_buf.height();
        if (width == 0 || height == 0) {
            candidate_regions_.clear();
            return;
        }

        candidate_regions_.clear();

        const int stride = output_buf.raw_buffer()->dim[1].stride;
        const int tilesz = roi_tile_size_;
        const int tile_w = tile_edge_buf.width();
        const int tile_h = tile_edge_buf.height();
        const int tile_count = tile_w * tile_h;

        std::vector<uint8_t> keep(tile_count, 0);

        const int edge_threshold = std::max(tilesz * 4, 64);
        const int fg_threshold = std::max(tilesz * tilesz / 5, 128);
        const float ratio_threshold = 0.4f;

        for (int ty = 0; ty < tile_h; ++ty) {
            for (int tx = 0; tx < tile_w; ++tx) {
                const int idx = ty * tile_w + tx;
                const int edge_count = tile_edge_buf(tx, ty);
                const int fg_count = tile_fg_buf(tx, ty);
                const float ratio = fg_count > 0 ? static_cast<float>(edge_count) / static_cast<float>(fg_count) : 0.0f;
                if (edge_count >= edge_threshold && fg_count >= fg_threshold && ratio >= ratio_threshold) {
                    keep[idx] = 1;
                }
            }
        }

        const int keep_count = std::accumulate(keep.begin(), keep.end(), 0);
        const double keep_ratio = tile_count > 0 ? static_cast<double>(keep_count) / static_cast<double>(tile_count) : 0.0;

        if (keep_count == 0 || keep_ratio >= 0.10) {
            return;
        }

        std::vector<uint8_t> current = keep;
        std::vector<uint8_t> next = current;
        const int iterations = 3;
        for (int iter = 0; iter < iterations; ++iter) {
            next = current;
            bool changed = false;
            for (int ty = 0; ty < tile_h; ++ty) {
                for (int tx = 0; tx < tile_w; ++tx) {
                    const int idx = ty * tile_w + tx;
                    if (current[idx]) {
                        continue;
                    }
                    bool any = false;
                    for (int dy = -1; dy <= 1 && !any; ++dy) {
                        const int ny = ty + dy;
                        if (ny < 0 || ny >= tile_h) {
                            continue;
                        }
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = tx + dx;
                            if (nx < 0 || nx >= tile_w) {
                                continue;
                            }
                            if (current[ny * tile_w + nx]) {
                                any = true;
                                break;
                            }
                        }
                    }
                    if (any) {
                        next[idx] = 1;
                        changed = true;
                    }
                }
            }
            current.swap(next);
            if (!changed) {
                break;
            }
        }

        const std::vector<uint8_t> &final_keep = current;
        std::vector<uint8_t> visited(tile_count, 0);

        for (int ty = 0; ty < tile_h; ++ty) {
            for (int tx = 0; tx < tile_w; ++tx) {
                const int idx = ty * tile_w + tx;
                if (!final_keep[idx] || visited[idx]) {
                    continue;
                }

                int min_tx = tx;
                int max_tx = tx;
                int min_ty = ty;
                int max_ty = ty;
                int64_t edge_sum = 0;
                int64_t fg_sum = 0;

                std::vector<int> stack;
                stack.push_back(idx);
                visited[idx] = 1;

                static const int kDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

                while (!stack.empty()) {
                    const int current_idx = stack.back();
                    stack.pop_back();

                    const int cx = current_idx % tile_w;
                    const int cy = current_idx / tile_w;

                    min_tx = std::min(min_tx, cx);
                    max_tx = std::max(max_tx, cx);
                    min_ty = std::min(min_ty, cy);
                    max_ty = std::max(max_ty, cy);

                    edge_sum += tile_edge_buf(cx, cy);
                    fg_sum += tile_fg_buf(cx, cy);

                    for (const auto &dir : kDirs) {
                        const int nx = cx + dir[0];
                        const int ny = cy + dir[1];
                        if (nx < 0 || nx >= tile_w || ny < 0 || ny >= tile_h) {
                            continue;
                        }
                        const int nidx = ny * tile_w + nx;
                        if (!final_keep[nidx] || visited[nidx]) {
                            continue;
                        }
                        visited[nidx] = 1;
                        stack.push_back(nidx);
                    }
                }

                const int margin = tilesz * 2;
                int x0 = min_tx * tilesz;
                int y0 = min_ty * tilesz;
                int x1 = std::min(width, (max_tx + 1) * tilesz) - 1;
                int y1 = std::min(height, (max_ty + 1) * tilesz) - 1;

                x0 = std::max(0, x0 - margin);
                y0 = std::max(0, y0 - margin);
                x1 = std::min(width - 1, x1 + margin);
                y1 = std::min(height - 1, y1 + margin);

                halide_candidate_region region{};
                region.x0 = x0;
                region.y0 = y0;
                region.x1 = x1;
                region.y1 = y1;
                region.edge_pixels = static_cast<int32_t>(edge_sum);
                region.avg_gradient = fg_sum > 0 ? static_cast<float>(edge_sum) / static_cast<float>(fg_sum) : 0.0f;

                candidate_regions_.push_back(region);
            }
        }

        if (candidate_regions_.empty()) {
            return;
        }

        std::sort(candidate_regions_.begin(), candidate_regions_.end(), [](const halide_candidate_region &a, const halide_candidate_region &b) {
            const int64_t area_a = static_cast<int64_t>(a.x1 - a.x0 + 1) * static_cast<int64_t>(a.y1 - a.y0 + 1);
            const int64_t area_b = static_cast<int64_t>(b.x1 - b.x0 + 1) * static_cast<int64_t>(b.y1 - b.y0 + 1);
            return area_a > area_b;
        });

        constexpr size_t kMaxCandidates = 128;
        if (candidate_regions_.size() > kMaxCandidates) {
            candidate_regions_.resize(kMaxCandidates);
        }

        int64_t total_roi_area = 0;
        for (const auto &roi : candidate_regions_) {
            total_roi_area += static_cast<int64_t>(roi.x1 - roi.x0 + 1) * static_cast<int64_t>(roi.y1 - roi.y0 + 1);
        }

        const int64_t total_pixels = static_cast<int64_t>(width) * static_cast<int64_t>(height);
        const double coverage = static_cast<double>(total_roi_area) / static_cast<double>(total_pixels);

        if (coverage > 0.6) {
            return;
        }

        for (int ty = 0; ty < tile_h; ++ty) {
            const int y0 = ty * tilesz;
            const int y1 = std::min(height, y0 + tilesz);
            for (int tx = 0; tx < tile_w; ++tx) {
                if (final_keep[ty * tile_w + tx]) {
                    continue;
                }
                const int x0 = tx * tilesz;
                const int x1 = std::min(width, x0 + tilesz);
                for (int y = y0; y < y1; ++y) {
                    uint8_t *row = output_buf.data() + y * stride;
                    std::fill(row + x0, row + x1, static_cast<uint8_t>(127));
                }
            }
        }
    }

    ImageParam input_;
    Param<int> min_white_black_diff_;
    std::once_flag init_flag_;
    const int tile_size_;
    const int roi_tile_size_;
    bool enable_roi_filter_;
    int tile_width_ = 0;
    int tile_height_ = 0;
    std::vector<halide_candidate_region> candidate_regions_;
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

extern "C" const halide_candidate_region *halide_get_candidate_rois(size_t *count)
{
    ThresholdPipeline &pipeline = get_pipeline();
    const auto &regions = pipeline.candidate_regions();
    if (count) {
        *count = regions.size();
    }
    return regions.empty() ? nullptr : regions.data();
}
