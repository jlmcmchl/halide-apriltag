// Halide tutorial lesson 21: Auto-Scheduler

// So far we have written Halide schedules by hand, but it is also possible to
// ask Halide to suggest a reasonable schedule. We call this auto-scheduling.
// This lesson demonstrates how to use the autoscheduler to generate a
// copy-pasteable CPU schedule that can be subsequently improved upon.

// On linux or os x, you can compile and run it like so:

// g++ lesson_21_auto_scheduler_generate.cpp <path/to/tools/halide_image_io.h>/GenGen.cpp -g -std=c++17 -fno-rtti -I <path/to/Halide.h> -L <path/to/libHalide.so> -lHalide -lpthread -ldl -o lesson_21_generate
// export LD_LIBRARY_PATH=<path/to/libHalide.so>   # For linux
// export DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> # For OS X
// ./lesson_21_generate -o . -g auto_schedule_gen -f auto_schedule_false -e static_library,h,schedule target=host auto_schedule=false
// ./lesson_21_generate -o . -g auto_schedule_gen -f auto_schedule_true -e static_library,h,schedule -p <path/to/libautoschedule_mullapudi2016.so> -S Mullapudi2016 target=host autoscheduler=Mullapudi2016 autoscheduler.parallelism=32 autoscheduler.last_level_cache_size=16777216 autoscheduler.balance=40
// g++ lesson_21_auto_scheduler_run.cpp -std=c++17 -I <path/to/Halide.h> -I <path/to/tools/halide_image_io.h> auto_schedule_false.a auto_schedule_true.a -ldl -lpthread -o lesson_21_run
// ./lesson_21_run

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_21_auto_scheduler_run
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"

using namespace Halide;
using Halide::BoundaryConditions::repeat_edge;

// We will define a generator to auto-schedule.
class BasicAdaptiveThreshold : public Halide::Generator<BasicAdaptiveThreshold> {
public:
    Input<Buffer<uint8_t, 2>> input{"input"};
    Input<int> min_white_black_diff{"min_white_black_diff"};
    Input<int> tile_size{"tile_size"};
    Input<int> roi_tile_size{"roi_tile_size"};

    Output<Buffer<uint8_t, 2>> output{"output"};
    Output<Buffer<int32_t, 2>> tile_edge_count{"tile_edge_count"};
    Output<Buffer<int32_t, 2>> tile_fg_count{"tile_fg_count"};


    void generate() {

        Func padded = repeat_edge(input, {{0, input.width()}, {0, input.height()}});

        Halide::Expr tile_w = (input.width() + tile_size - 1) / tile_size;
        Halide::Expr tile_h = (input.height() + tile_size - 1) / tile_size;

        Halide::Expr clamped_tile_w = tile_w - 1;
        Halide::Expr clamped_tile_h = tile_h - 1;

        Halide::RDom tile_dom(0, tile_size, 0, tile_size);

        Halide::Expr sx = Halide::min(tx * tile_size + tile_dom.x, input.width() - 1);
        Halide::Expr sy = Halide::min(ty * tile_size + tile_dom.y, input.height() - 1);
        tile_min(tx, ty) = Halide::cast<uint8_t>(255);
        tile_max(tx, ty) = Halide::cast<uint8_t>(0);
        tile_min(tx, ty) = Halide::min(tile_min(tx, ty), padded(sx, sy));
        tile_max(tx, ty) = Halide::max(tile_max(tx, ty), padded(sx, sy));

        Halide::RDom neigh_dom(-1, 3, -1, 3);
        Halide::Expr ntx = Halide::clamp(tx + neigh_dom.x, 0, clamped_tile_w);
        Halide::Expr nty = Halide::clamp(ty + neigh_dom.y, 0, clamped_tile_h);
        neigh_min(tx, ty) = Halide::cast<uint8_t>(255);
        neigh_max(tx, ty) = Halide::cast<uint8_t>(0);
        neigh_min(tx, ty) = Halide::min(neigh_min(tx, ty), tile_min(ntx, nty));
        neigh_max(tx, ty) = Halide::max(neigh_max(tx, ty), tile_max(ntx, nty));

        Halide::Expr tile_x = Halide::min(x / tile_size, clamped_tile_w);
        Halide::Expr tile_y = Halide::min(y / tile_size, clamped_tile_h);
        min_px(x, y) = neigh_min(tile_x, tile_y);
        max_px(x, y) = neigh_max(tile_x, tile_y);

        Halide::Expr diff = Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
        Halide::Expr threshold = Halide::cast<uint8_t>(
            Halide::cast<int>(min_px(x, y)) + diff / 2);
        output(x, y) = Halide::cast<uint8_t>(
            Halide::select(diff < min_white_black_diff,
                        127,
                        Halide::select(padded(x, y) > threshold, 255, 0)));

        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        

        Halide::Expr roi_tile_w = (input.width() + roi_tile_size - 1) / roi_tile_size;
        Halide::Expr roi_tile_h = (input.height() + roi_tile_size - 1) / roi_tile_size;

        Halide::RDom roi_dom(0, roi_tile_size, 0, roi_tile_size);
        tile_edge_count(tx, ty) = 0;
        tile_fg_count(tx, ty) = 0;

        Halide::Expr rx = Halide::min(tx * roi_tile_size + roi_dom.x, input.width() - 1);
        Halide::Expr ry = Halide::min(ty * roi_tile_size + roi_dom.y, input.height() - 1);
        Halide::Expr center_v = output(rx, ry);
        Halide::Expr valid = center_v != 127;
        Halide::Expr right_v = output(Halide::min(rx + 1, input.width() - 1), ry);
        Halide::Expr down_v = output(rx, Halide::min(ry + 1, input.height() - 1));
        Halide::Expr edge = Halide::select(valid && right_v != 127 && center_v != right_v, 1, 0) +
                            Halide::select(valid && down_v != 127 && center_v != down_v, 1, 0);
        tile_edge_count(tx, ty) += edge;
        tile_fg_count(tx, ty) += Halide::select(valid, 1, 0);
    }

    void schedule() {
        if (using_autoscheduler()) {
            // The autoscheduler requires estimates on all the input/output
            // sizes and parameter values in order to compare different
            // alternatives and decide on a good schedule.

            // To provide estimates (min and extent values) for each dimension
            // of the input images ('input', 'filter', and 'bias'), we use the
            // set_estimates() method. set_estimates() takes in a list of
            // (min, extent) of the corresponding dimension as arguments.
            input.set_estimates({{0, 1280}, {0, 951}});

            // To provide estimates on the parameter values, we use the
            // set_estimate() method.
            tile_size.set_estimate(4.0);
            min_white_black_diff.set_estimate(30.0f);
            roi_tile_size.set_estimate(16.0);

            // To provide estimates (min and extent values) for each dimension
            // of pipeline outputs, we use the set_estimates() method. set_estimates()
            // takes in a list of (min, extent) for each dimension.
            output.set_estimates({{0, 1280}, {0, 951}});
            tile_edge_count.set_estimates({{0, 1280}, {0, 951}});
            tile_fg_count.set_estimates({{0, 1280}, {0, 951}});

            // Technically, the estimate values can be anything, but the closer
            // they are to the actual use-case values, the better the generated
            // schedule will be.

        } else {

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

            tile_edge_count.compute_root().parallel(ty);
            tile_edge_count.update().parallel(ty);

            tile_fg_count.compute_root().parallel(ty);
            tile_fg_count.update().parallel(ty);
        }
    }

private:
    Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"}, tx{"tx"}, ty{"ty"};

    Func tile_min{"tile_min"}, tile_max{"tile_max"}, binary{"binary"}, grey{"grey"};
    Func min_px{"min_px"}, max_px{"max_px"};
    Func neigh_min{"neigh_min"}, neigh_max{"neigh_max"};
};

HALIDE_REGISTER_GENERATOR(BasicAdaptiveThreshold, basic_adaptive_threshold_gen)

class MetalBasicAdaptiveThreshold: public Halide::Generator<MetalBasicAdaptiveThreshold> {
public:
    Input<Buffer<uint8_t, 2>> input{"input"};
    Input<int> min_white_black_diff{"min_white_black_diff"};
    Input<int> tile_size{"tile_size"};
    Input<int> roi_tile_size{"roi_tile_size"};
    GeneratorParam<int> pointer_jump_iterations{"pointer_jump_iterations", 8};

    Output<Buffer<uint8_t, 2>> threshold{"threshold"};
    Output<Buffer<int32_t, 2>> parent{"parent"};

    void generate() {
        Func padded = repeat_edge(input, {{0, input.width()}, {0, input.height()}});

        Halide::Expr tile_w = (input.width() + tile_size - 1) / tile_size;
        Halide::Expr tile_h = (input.height() + tile_size - 1) / tile_size;
        Halide::Expr clamped_tile_w = Halide::max(tile_w - 1, 0);
        Halide::Expr clamped_tile_h = Halide::max(tile_h - 1, 0);

        Halide::RDom tile_dom(0, tile_size, 0, tile_size);

        Halide::Expr sx = Halide::min(tx * tile_size + tile_dom.x, input.width() - 1);
        Halide::Expr sy = Halide::min(ty * tile_size + tile_dom.y, input.height() - 1);
        tile_min(tx, ty) = Halide::cast<uint8_t>(255);
        tile_max(tx, ty) = Halide::cast<uint8_t>(0);
        tile_min(tx, ty) = Halide::min(tile_min(tx, ty), padded(sx, sy));
        tile_max(tx, ty) = Halide::max(tile_max(tx, ty), padded(sx, sy));

        Halide::RDom neigh_dom(-1, 3, -1, 3);
        Halide::Expr ntx = Halide::clamp(tx + neigh_dom.x, 0, clamped_tile_w);
        Halide::Expr nty = Halide::clamp(ty + neigh_dom.y, 0, clamped_tile_h);
        neigh_min(tx, ty) = Halide::cast<uint8_t>(255);
        neigh_max(tx, ty) = Halide::cast<uint8_t>(0);
        neigh_min(tx, ty) = Halide::min(neigh_min(tx, ty), tile_min(ntx, nty));
        neigh_max(tx, ty) = Halide::max(neigh_max(tx, ty), tile_max(ntx, nty));

        Halide::Expr tile_x = Halide::min(x / tile_size, clamped_tile_w);
        Halide::Expr tile_y = Halide::min(y / tile_size, clamped_tile_h);
        min_px(x, y) = neigh_min(tile_x, tile_y);
        max_px(x, y) = neigh_max(tile_x, tile_y);

        Halide::Expr diff = Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
        Halide::Expr threshold_val = Halide::cast<uint8_t>(
            Halide::cast<int>(min_px(x, y)) + diff / 2);
        threshold(x, y) = Halide::cast<uint8_t>(
            Halide::select(diff < min_white_black_diff,
                            127,
                            Halide::select(padded(x, y) > threshold_val, 255, 0)));

        binary_val(x, y) = threshold(x, y);
        

        Halide::Expr width_expr = input.width();
        Halide::Expr height_expr = input.height();
        Halide::Expr idx_expr = y * width_expr + x;
        Halide::Expr self_value = binary_val(x, y);
        parent_init(x, y) = Halide::cast<int32_t>(
            Halide::select(self_value == 127, -1, idx_expr));

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

        parent_stages.push_back(parent_hook);

        for (int i = 0; i < pointer_jump_iterations; ++i) {
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

        parent(x, y) = parent_stages.back()(x, y);
    }

    void schedule() {
        if (using_autoscheduler()) {
            // The autoscheduler requires estimates on all the input/output
            // sizes and parameter values in order to compare different
            // alternatives and decide on a good schedule.

            input.set_estimates({{0, 1280}, {0, 951}});
            min_white_black_diff.set_estimate(30.0f);
            tile_size.set_estimate(4.0);
            roi_tile_size.set_estimate(16.0);

            threshold.set_estimates({{0, 1280}, {0, 951}});
            parent.set_estimates({{0, 1280}, {0, 951}});

        } else {
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
        }
    }
std::vector<Func> parent_stages;
Var xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

    Func  binary{"binary"}, grey{"grey"};

    Var x{"x"}, y{"y"};
        Var tx{"tx"}, ty{"ty"};
        Func tile_min{"metal_tile_min"}, tile_max{"metal_tile_max"};

        Func neigh_min{"metal_neigh_min"}, neigh_max{"metal_neigh_max"};

        Func min_px{"metal_min_px"}, max_px{"metal_max_px"};

        Func binary_val{"metal_binary_val"};

        Func parent_init{"metal_parent_init"};
        Func parent_hook{"metal_parent_hook"};
};

HALIDE_REGISTER_GENERATOR(MetalBasicAdaptiveThreshold, metal_basic_adaptive_threshold_gen)
