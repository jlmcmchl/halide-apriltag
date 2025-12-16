#include "Halide.h"

using namespace Halide;

// We will define a generator to auto-schedule.
class GreyscaleAndAdaptiveThreshold
    : public Halide::Generator<GreyscaleAndAdaptiveThreshold> {
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

    grey(x, y) = input(x, y, 0) * 0.299f + input(x, y, 1) * 0.587f +
                 input(x, y, 2) * 0.114f;

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

    // Binary: 127 if without contrast, 0 if contrast & darker, 255 if contrast
    // & brighter
    binary(x, y) = cast<uint8_t>(select(
        contrast > min_contrast && clamped(x, y) > thresh, 255,
        select(contrast > min_contrast && clamped(x, y) < thresh, 0, 127)));

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
      if (get_target().has_gpu_feature()) {
        Var tx(tile_min.get_schedule().dims()[0].var);
        Var txi("txi");
        Var ty(tile_min.get_schedule().dims()[1].var);
        Var x(binary.get_schedule().dims()[0].var);
        Var xi("xi");
        Var xii("xii");
        Var y(binary.get_schedule().dims()[1].var);
        Var yi("yi");
        Var yii("yii");
        RVar tile_minmax_x(tile_min.update(0).get_schedule().dims()[0].var);
        RVar tile_minmax_y(tile_min.update(0).get_schedule().dims()[1].var);
        Var ty_serial_outer("ty_serial_outer");
        Var tx_serial_outer("tx_serial_outer");
        Var yi_serial_outer("yi_serial_outer");
        Var xi_serial_outer("xi_serial_outer");
        output
            .split(x, x, xi, 128, TailStrategy::ShiftInwards)
            .split(y, y, yi, 128, TailStrategy::ShiftInwards)
            .split(xi, xi, xii, 2, TailStrategy::ShiftInwards)
            .split(yi, yi, yii, 8, TailStrategy::ShiftInwards)
            .unroll(xii)
            .unroll(yii)
            .compute_root()
            .reorder(xii, yii, xi, yi, x, y)
            .gpu_blocks(x)
            .gpu_blocks(y)
            .split(xi, xi_serial_outer, xi, 16, TailStrategy::GuardWithIf)
            .gpu_threads(xi)
            .split(yi, yi_serial_outer, yi, 8, TailStrategy::GuardWithIf)
            .gpu_threads(yi);
        tile_min
            .split(tx, tx, txi, 3, TailStrategy::RoundUp)
            .unroll(txi)
            .compute_at(output, x)
            .reorder(txi, tx, ty)
            .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 2, TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_min.update(0)
            .split(tx, tx, txi, 3, TailStrategy::RoundUp)
            .unroll(txi)
            .reorder(txi, tile_minmax_x, tile_minmax_y, tx, ty)
            .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 2, TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_max
            .split(tx, tx, txi, 3, TailStrategy::RoundUp)
            .unroll(txi)
            .compute_at(output, x)
            .reorder(txi, tx, ty)
            .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 2, TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_max.update(0)
            .split(tx, tx, txi, 3, TailStrategy::RoundUp)
            .unroll(txi)
            .reorder(txi, tile_minmax_x, tile_minmax_y, tx, ty)
            .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 2, TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        grey
            .split(x, x, xi, 32, TailStrategy::ShiftInwards)
            .split(y, y, yi, 8, TailStrategy::ShiftInwards)
            .split(yi, yi, yii, 8, TailStrategy::ShiftInwards)
            .unroll(yii)
            .compute_root()
            .reorder(yii, xi, yi, x, y)
            .gpu_blocks(x)
            .gpu_blocks(y)
            .split(xi, xi_serial_outer, xi, 32, TailStrategy::GuardWithIf)
            .gpu_threads(xi);
      } else {
        Var tx(tile_min.get_schedule().dims()[0].var);
        Var ty(tile_min.get_schedule().dims()[1].var);
        Var tyi("tyi");
        Var x(binary.get_schedule().dims()[0].var);
        Var xi("xi");
        Var xii("xii");
        Var xiii("xiii");
        Var y(binary.get_schedule().dims()[1].var);
        Var yi("yi");
        Var yii("yii");
        RVar tile_minmax_x(tile_min.update(0).get_schedule().dims()[0].var);
        RVar tile_minmax_y(tile_min.update(0).get_schedule().dims()[1].var);
        output
            .split(x, x, xi, 256, TailStrategy::ShiftInwards)
            .split(y, y, yi, 64, TailStrategy::ShiftInwards)
            .split(xi, xi, xii, 128, TailStrategy::ShiftInwards)
            .split(yi, yi, yii, 2, TailStrategy::ShiftInwards)
            .split(xii, xii, xiii, 16, TailStrategy::ShiftInwards)
            .unroll(xii)
            .vectorize(xiii)
            .compute_root()
            .reorder({xiii, xii, yii, yi, xi, x, y})
            .fuse(x, y, x)
            .parallel(x);
        binary
            .store_in(MemoryType::Stack)
            .split(x, x, xi, 16, TailStrategy::RoundUp)
            .unroll(x)
            .vectorize(xi)
            .compute_at(output, yii)
            .store_at(output, yi)
            .reorder({xi, x, y});
        local_min
            .store_in(MemoryType::Stack)
            .split(x, x, xi, 4, TailStrategy::RoundUp)
            .vectorize(xi)
            .compute_at(output, yii)
            .store_at(output, yi)
            .reorder({xi, x, y});
        tile_min
            .store_in(MemoryType::Stack)
            .split(ty, ty, tyi, 4, TailStrategy::RoundUp)
            .vectorize(tyi)
            .compute_at(output, xi)
            .reorder({tyi, ty, tx})
            .reorder_storage(ty, tx);
        tile_min.update(0)
            .split(ty, ty, tyi, 4, TailStrategy::RoundUp)
            .vectorize(tyi)
            .reorder({tyi, tile_minmax_x, tile_minmax_y, ty, tx});
        local_max
            .store_in(MemoryType::Stack)
            .split(x, x, xi, 4, TailStrategy::RoundUp)
            .vectorize(xi)
            .compute_at(output, yii)
            .store_at(output, yi)
            .reorder({xi, x, y});
        tile_max
            .store_in(MemoryType::Stack)
            .split(ty, ty, tyi, 4, TailStrategy::RoundUp)
            .vectorize(tyi)
            .compute_at(output, xi)
            .reorder({tyi, ty, tx})
            .reorder_storage(ty, tx);
        tile_max.update(0)
            .split(ty, ty, tyi, 4, TailStrategy::RoundUp)
            .vectorize(tyi)
            .reorder({tyi, tile_minmax_x, tile_minmax_y, ty, tx});
        grey
            .store_in(MemoryType::Stack)
            .split(y, y, yi, 16, TailStrategy::GuardWithIf)
            .vectorize(yi)
            .compute_at(output, xi)
            .reorder({yi, y, x})
            .reorder_storage(y, x);
      }
    }
  }

private:
  Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"}, tx{"tx"},
      ty{"ty"};

  Func tile_min{"tile_min"}, tile_max{"tile_max"}, binary{"binary"},
      grey{"grey"}, local_min{"local_min"}, local_max{"local_max"};
};

HALIDE_REGISTER_GENERATOR(GreyscaleAndAdaptiveThreshold,
                          greyscale_and_adaptive_threshold_gen)
