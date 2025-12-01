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

// We will define a generator to auto-schedule.
class GreyscaleAndAdaptiveThreshold : public Halide::Generator<GreyscaleAndAdaptiveThreshold> {
    public:
        Input<Buffer<uint8_t, 3>> input{"input"};
        Input<int> tile_size{"tile_size"};
        Input<float> min_contrast{"min_contrast"};
    
        Output<Buffer<uint8_t, 2>> output{"output"};
    
    
        void generate() {
            Expr width = input.width();
            Expr height = input.height();
            Expr max_x = width - 1;
            Expr max_y = height - 1;

            grey(x, y) = input(x, y, 0) * 0.299f + input(x, y, 1) * 0.587f + input(x, y, 2) * 0.114f;

    
            // Step 1: Clamp input
            Func clamped("clamped");
            clamped(x, y) = grey(clamp(x, 0, max_x), clamp(y, 0, max_y));
    
            // Step 2: Compute local min/max in TILE_SIZE x TILE_SIZE neighborhoods
            // This is the key to adaptive thresholding - no manual tuning needed
            Expr tiles_x = (width + tile_size - 1) / tile_size;
            Expr tiles_y = (height + tile_size - 1) / tile_size;
            
            RDom r(0, tile_size, 0, tile_size, "tile_minmax");
            
            tile_min(tx, ty) = 255.0f;
            tile_max(tx, ty) = 0.0f;
            
            Expr px = clamp(tx * tile_size + r.x, 0, max_x);
            Expr py = clamp(ty * tile_size + r.y, 0, max_y);
            tile_min(tx, ty) = min(tile_min(tx, ty), clamped(px, py));
            tile_max(tx, ty) = max(tile_max(tx, ty), clamped(px, py));
    
            // Step 3: For each pixel, look up the min/max from surrounding tiles
            // and compute adaptive threshold
            Expr tile_x = clamp(x / tile_size, 0, tiles_x - 1);
            Expr tile_y = clamp(y / tile_size, 0, tiles_y - 1);
            
            // Sample 3x3 neighborhood of tiles for smoother thresholding
            Func local_min("local_min"), local_max("local_max");
            Expr lmin = cast<float>(255.0f);
            Expr lmax = cast<float>(0.0f);
            
            for (int dy = -1; dy <= 1; ++dy) {
                Expr ty_n = clamp(tile_y + dy, 0, tiles_y - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    Expr tx_n = clamp(tile_x + dx, 0, tiles_x - 1);
                    lmin = min(lmin, tile_min(tx_n, ty_n));
                    lmax = max(lmax, tile_max(tx_n, ty_n));
                }
            }
            
            local_min(x, y) = lmin;
            local_max(x, y) = lmax;
    
            // Step 4: Adaptive threshold
            // threshold = (min + max) / 2
            // Only mark as foreground if contrast is sufficient
            Expr thresh = (local_min(x, y) + local_max(x, y)) / 2.0f;
            Expr contrast = local_max(x, y) - local_min(x, y);
            
            // Binary: 255 if dark (below threshold), 0 if light, with contrast check
            binary(x, y) = cast<uint8_t>(select(
                contrast > min_contrast && clamped(x, y) < thresh,
                255,
                0
            ));
    
            output(x, y) = binary(x, y);
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
                input.set_estimates({{0, 640}, {0, 400}, {0, 3}});
    
                // To provide estimates on the parameter values, we use the
                // set_estimate() method.
                tile_size.set_estimate(4.0);
                min_contrast.set_estimate(30.0f);
    
                // To provide estimates (min and extent values) for each dimension
                // of pipeline outputs, we use the set_estimates() method. set_estimates()
                // takes in a list of (min, extent) for each dimension.
                output.set_estimates({{0, 1024}, {0, 1024}});
    
                // Technically, the estimate values can be anything, but the closer
                // they are to the actual use-case values, the better the generated
                // schedule will be.
    
            } else {

                grey.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

                tile_min.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);
                tile_min.update().unscheduled();
                tile_max.compute_root().gpu_tile(tx, ty, xo, yo, xi, yi, 16, 16);
                tile_max.update().unscheduled();
                
                binary.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            }
        }
    
    private:
        Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"}, tx{"tx"}, ty{"ty"};
    
        Func tile_min{"tile_min"}, tile_max{"tile_max"}, binary{"binary"}, grey{"grey"};
    };

HALIDE_REGISTER_GENERATOR(GreyscaleAndAdaptiveThreshold, greyscale_and_adaptive_threshold_gen)
