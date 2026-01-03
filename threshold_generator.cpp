#include "Halide.h"

using namespace Halide;

// We will define a generator to auto-schedule.
class AdaptiveThreshold : public Halide::Generator<AdaptiveThreshold> {
public:
  Input<Buffer<uint8_t, 2>> input{"input"};
  Input<int> tile_size{"tile_size"};
  Input<int> min_white_black_diff{"min_contrast"};

  Output<Buffer<uint8_t, 2>> output{"output"};

  void generate() {
    Expr width = input.width();
    Expr height = input.height();

    Func padded =
        BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});

    Halide::Expr tile_w = Halide::max(1, width / tile_size);
    Halide::Expr tile_h = Halide::max(1, height / tile_size);

    Halide::Expr clamped_tile_w = tile_w - 1;
    Halide::Expr clamped_tile_h = tile_h - 1;

    Halide::Expr tile_limit_x = tile_w * tile_size;
    Halide::Expr tile_limit_y = tile_h * tile_size;

    Halide::RDom tile_dom(0, tile_size, 0, tile_size);

    Halide::Expr sx = Halide::clamp(tx * tile_size + tile_dom.x, 0, width - 1);
    Halide::Expr sy = Halide::clamp(ty * tile_size + tile_dom.y, 0, height - 1);
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

    Halide::Expr tile_x = Halide::clamp(
        Halide::select(x < tile_limit_x, x / tile_size, clamped_tile_w), 0,
        clamped_tile_w);
    Halide::Expr tile_y = Halide::clamp(
        Halide::select(y < tile_limit_y, y / tile_size, clamped_tile_h), 0,
        clamped_tile_h);
    min_px(x, y) = neigh_min(tile_x, tile_y);
    max_px(x, y) = neigh_max(tile_x, tile_y);

    Halide::Expr diff =
        Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
    Halide::Expr threshold =
        Halide::cast<uint8_t>(Halide::cast<int>(min_px(x, y)) + diff / 2);
    binary(x, y) = Halide::cast<uint8_t>(
        Halide::select(diff < min_white_black_diff, 127,
                       Halide::select(padded(x, y) > threshold, 255, 0)));
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
      input.set_estimates({{0, 1600}, {0, 1200}});

      // To provide estimates on the parameter values, we use the
      // set_estimate() method.
      tile_size.set_estimate(4.0);
      min_white_black_diff.set_estimate(30.0f);

      // To provide estimates (min and extent values) for each dimension
      // of pipeline outputs, we use the set_estimates() method. set_estimates()
      // takes in a list of (min, extent) for each dimension.
      output.set_estimates({{0, 1600}, {0, 1200}});

      // Technically, the estimate values can be anything, but the closer
      // they are to the actual use-case values, the better the generated
      // schedule will be.

    } else if (get_target().has_gpu_feature()) {
      printf("Scheduling for GPU\n");
      using ::Halide::Func;
      using ::Halide::MemoryType;
      using ::Halide::RVar;
      using ::Halide::TailStrategy;
      using ::Halide::Var;
      auto pipeline = get_pipeline();
      Func output = pipeline.get_func(10);
      Func binary = pipeline.get_func(9);
      Func min_px = pipeline.get_func(8);
      Func neigh_min = pipeline.get_func(7);
      Func tile_min = pipeline.get_func(6);
      Func max_px = pipeline.get_func(5);
      Func neigh_max = pipeline.get_func(4);
      Func tile_max = pipeline.get_func(3);
      Func repeat_edge = pipeline.get_func(2);
      Func lambda_0 = pipeline.get_func(1);
      Var _0(repeat_edge.get_schedule().dims()[0].var);
      Var _0i("_0i");
      Var _1(repeat_edge.get_schedule().dims()[1].var);
      Var _1i("_1i");
      Var _1ii("_1ii");
      Var tx(neigh_min.get_schedule().dims()[0].var);
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
      Var _0i_serial_outer("_0i_serial_outer");
      Var ty_serial_outer("ty_serial_outer");
      Var tx_serial_outer("tx_serial_outer");
      Var yi_serial_outer("yi_serial_outer");
      Var xi_serial_outer("xi_serial_outer");
      output.split(x, x, xi, 64, TailStrategy::ShiftInwards)
          .split(y, y, yi, 64, TailStrategy::ShiftInwards)
          .split(xi, xi, xii, 2, TailStrategy::ShiftInwards)
          .split(yi, yi, yii, 8, TailStrategy::ShiftInwards)
          .unroll(xii)
          .unroll(yii)
          .compute_root()
          .reorder(xii, yii, xi, yi, x, y)
          .gpu_blocks(x)
          .gpu_blocks(y)
          .split(xi, xi_serial_outer, xi, 32, TailStrategy::GuardWithIf)
          .gpu_threads(xi)
          .split(yi, yi_serial_outer, yi, 8, TailStrategy::GuardWithIf)
          .gpu_threads(yi);
      neigh_min.split(ty, ty, tyi, 4, TailStrategy::RoundUp)
          .unroll(tyi)
          .compute_at(output, x)
          .reorder(tyi, tx, ty)
          .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 4, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      neigh_min.update(0)
          .split(ty, ty, tyi, 4, TailStrategy::RoundUp)
          .unroll(tyi)
          .reorder(tyi, r23_x, r23_y, tx, ty)
          .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 4, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      tile_min.compute_at(output, x)
          .reorder(tx, ty)
          .split(tx, tx_serial_outer, tx, 18, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 18, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      tile_min.update(0)
          .reorder(r10_x, r10_y, tx, ty)
          .split(tx, tx_serial_outer, tx, 18, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 18, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      neigh_max.split(ty, ty, tyi, 16, TailStrategy::RoundUp)
          .unroll(tyi)
          .compute_at(output, x)
          .reorder(tyi, tx, ty)
          .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
          .gpu_threads(tx);
      neigh_max.update(0)
          .split(ty, ty, tyi, 16, TailStrategy::RoundUp)
          .unroll(tyi)
          .reorder(tyi, r23_x, r23_y, tx, ty)
          .split(tx, tx_serial_outer, tx, 16, TailStrategy::GuardWithIf)
          .gpu_threads(tx);
      tile_max.compute_at(output, x)
          .reorder(tx, ty)
          .split(tx, tx_serial_outer, tx, 18, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 18, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      tile_max.update(0)
          .reorder(r10_x, r10_y, tx, ty)
          .split(tx, tx_serial_outer, tx, 18, TailStrategy::GuardWithIf)
          .gpu_threads(tx)
          .split(ty, ty_serial_outer, ty, 18, TailStrategy::GuardWithIf)
          .gpu_threads(ty);
      repeat_edge.split(_0, _0, _0i, 32, TailStrategy::ShiftInwards)
          .split(_1, _1, _1i, 8, TailStrategy::ShiftInwards)
          .split(_1i, _1i, _1ii, 8, TailStrategy::ShiftInwards)
          .unroll(_1ii)
          .compute_root()
          .reorder(_1ii, _0i, _1i, _0, _1)
          .gpu_blocks(_0)
          .gpu_blocks(_1)
          .split(_0i, _0i_serial_outer, _0i, 32, TailStrategy::GuardWithIf)
          .gpu_threads(_0i);
    } else {
      printf("Scheduling for CPU\n");
      using ::Halide::Func;
      using ::Halide::MemoryType;
      using ::Halide::RVar;
      using ::Halide::TailStrategy;
      using ::Halide::Var;
      auto pipeline = get_pipeline();
      Func output = pipeline.get_func(10);
      Func binary = pipeline.get_func(9);
      Func min_px = pipeline.get_func(8);
      Func neigh_min = pipeline.get_func(7);
      Func tile_min = pipeline.get_func(6);
      Func max_px = pipeline.get_func(5);
      Func neigh_max = pipeline.get_func(4);
      Func tile_max = pipeline.get_func(3);
      Func repeat_edge = pipeline.get_func(2);
      Func lambda_0 = pipeline.get_func(1);
      Var tx(neigh_min.get_schedule().dims()[0].var);
      Var txi("txi");
      Var ty(neigh_min.get_schedule().dims()[1].var);
      Var x(output.get_schedule().dims()[0].var);
      Var xi("xi");
      Var xii("xii");
      Var xiii("xiii");
      Var y(output.get_schedule().dims()[1].var);
      Var yi("yi");
      Var yii("yii");
      RVar r10_x(tile_min.update(0).get_schedule().dims()[0].var);
      RVar r10_y(tile_min.update(0).get_schedule().dims()[1].var);
      RVar r23_x(neigh_min.update(0).get_schedule().dims()[0].var);
      RVar r23_y(neigh_min.update(0).get_schedule().dims()[1].var);
      output.split(x, x, xi, 400, TailStrategy::ShiftInwards)
          .split(y, y, yi, 75, TailStrategy::ShiftInwards)
          .split(xi, xi, xii, 208, TailStrategy::ShiftInwards)
          .split(yi, yi, yii, 4, TailStrategy::ShiftInwards)
          .split(xii, xii, xiii, 16, TailStrategy::ShiftInwards)
          .vectorize(xiii)
          .compute_root()
          .reorder({xiii, xii, yii, yi, xi, x, y})
          .fuse(x, y, x)
          .parallel(x);
      binary.store_in(MemoryType::Stack)
          .split(x, x, xi, 112, TailStrategy::ShiftInwards)
          .split(y, y, yi, 2, TailStrategy::ShiftInwards)
          .split(xi, xi, xii, 16, TailStrategy::ShiftInwards)
          .unroll(xi)
          .vectorize(xii)
          .compute_at(output, yi)
          .reorder({xii, xi, yi, x, y});
      min_px.store_in(MemoryType::Stack)
          .split(x, x, xi, 16, TailStrategy::RoundUp)
          .unroll(x)
          .vectorize(xi)
          .compute_at(binary, yi)
          .store_at(binary, x)
          .reorder({xi, x, y});
      neigh_min.store_in(MemoryType::Stack)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .compute_at(output, xi)
          .reorder({txi, tx, ty});
      neigh_min.update(0)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .reorder({txi, r23_x, r23_y, tx, ty});
      tile_min.store_in(MemoryType::Stack)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .compute_at(output, xi)
          .reorder({txi, tx, ty});
      tile_min.update(0)
          .split(tx, tx, txi, 16, TailStrategy::GuardWithIf)
          .vectorize(txi)
          .reorder({txi, r10_x, r10_y, tx, ty});
      neigh_max.store_in(MemoryType::Stack)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .compute_at(output, xi)
          .reorder({txi, tx, ty});
      neigh_max.update(0)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .reorder({txi, r23_x, r23_y, tx, ty});
      tile_max.store_in(MemoryType::Stack)
          .split(tx, tx, txi, 16, TailStrategy::RoundUp)
          .vectorize(txi)
          .compute_at(output, xi)
          .reorder({txi, tx, ty});
      tile_max.update(0)
          .split(tx, tx, txi, 16, TailStrategy::GuardWithIf)
          .vectorize(txi)
          .reorder({txi, r10_x, r10_y, tx, ty});
    }
  }

private:
  Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"}, tx{"tx"},
      ty{"ty"};

  Func tile_min{"tile_min"}, tile_max{"tile_max"}, binary{"binary"},
      grey{"grey"}, neigh_min{"neigh_min"}, neigh_max{"neigh_max"},
      min_px{"min_px"}, max_px{"max_px"};
};

HALIDE_REGISTER_GENERATOR(AdaptiveThreshold, adaptive_threshold_gen)
