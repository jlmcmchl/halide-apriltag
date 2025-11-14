#include "Halide.h"

using namespace Halide;
using Halide::BoundaryConditions::repeat_edge;

class Threshold : public Generator<Threshold> {
public:
  Input<Buffer<uint8_t>> input{"input", 2};
  Output<Buffer<uint8_t>> output{"output", 2};
  GeneratorParam<int> min_white_black_diff{"min_white_black_diff", 5};
  Func padded{"padded"}, tile_min{"tile_min"}, tile_max{"tile_max"},
      neigh_min{"neigh_min"}, neigh_max{"neigh_max"}, min_px{"min_px"},
      max_px{"max_px"};
  Var x{"x"}, y{"y"}, tx{"tx"}, ty{"ty"};

  void generate() {
    Func padded = repeat_edge(input, {{0, input.width()}, {0, input.height()}});

    const int tilesz = 4;

    Halide::Expr tile_w = Halide::max(1, input.width() / tilesz);
    Halide::Expr tile_h = Halide::max(1, input.height() / tilesz);

    Halide::Expr clamped_tile_w = tile_w - 1;
    Halide::Expr clamped_tile_h = tile_h - 1;

    Halide::Expr tile_limit_x = tile_w * tilesz;
    Halide::Expr tile_limit_y = tile_h * tilesz;

    Halide::RDom tile_dom(0, tilesz, 0, tilesz);

    Halide::Expr sx =
        Halide::clamp(tx * tilesz + tile_dom.x, 0, input.width() - 1);
    Halide::Expr sy =
        Halide::clamp(ty * tilesz + tile_dom.y, 0, input.height() - 1);
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
        Halide::select(x < tile_limit_x, x / tilesz, clamped_tile_w), 0,
        clamped_tile_w);
    Halide::Expr tile_y = Halide::clamp(
        Halide::select(y < tile_limit_y, y / tilesz, clamped_tile_h), 0,
        clamped_tile_h);
    min_px(x, y) = neigh_min(tile_x, tile_y);
    max_px(x, y) = neigh_max(tile_x, tile_y);

    Halide::Expr diff =
        Halide::cast<int>(max_px(x, y)) - Halide::cast<int>(min_px(x, y));
    Halide::Expr threshold =
        Halide::cast<uint8_t>(Halide::cast<int>(min_px(x, y)) + diff / 2);
    output(x, y) = Halide::cast<uint8_t>(
        Halide::select(diff < min_white_black_diff, 127,
                       Halide::select(padded(x, y) > threshold, 255, 0)));
  }

  void schedule() {
    if (using_autoscheduler()) {
      input.set_estimates({{0, 1280}, {0, 800}});
      output.set_estimates({{0, 1280}, {0, 800}});
    } else {
      if (target.value().has_gpu_feature()) {
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
        Var x("x");
        Var xi("xi");
        Var xii("xii");
        Var y("y");
        Var yi("yi");
        Var yii("yii");
        RVar r10_x(tile_min.update(0).get_schedule().dims()[0].var);
        RVar r10_y(tile_min.update(0).get_schedule().dims()[1].var);
        RVar r23_x(neigh_min.update(0).get_schedule().dims()[0].var);
        RVar r23_y(neigh_min.update(0).get_schedule().dims()[1].var);
        Var _0i_serial_outer("_0i_serial_outer");
        Var tyi_serial_outer("tyi_serial_outer");
        Var tx_serial_outer("tx_serial_outer");
        Var txi_serial_outer("txi_serial_outer");
        Var ty_serial_outer("ty_serial_outer");
        Var yi_serial_outer("yi_serial_outer");
        Var xi_serial_outer("xi_serial_outer");
        output.split(x, x, xi, 64, Halide::TailStrategy::ShiftInwards)
            .split(y, y, yi, 32, Halide::TailStrategy::ShiftInwards)
            .split(xi, xi, xii, 2, Halide::TailStrategy::ShiftInwards)
            .split(yi, yi, yii, 4, Halide::TailStrategy::ShiftInwards)
            .unroll(xii)
            .unroll(yii)
            .compute_root()
            .reorder(xii, yii, xi, yi, x, y)
            .gpu_blocks(x)
            .gpu_blocks(y)
            .split(xi, xi_serial_outer, xi, 32,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(xi)
            .split(yi, yi_serial_outer, yi, 8,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(yi);
        neigh_min.compute_at(output, x)
            .reorder(tx, ty)
            .split(tx, tx_serial_outer, tx, 16,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 8,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        neigh_min.update(0)
            .reorder(r23_x, r23_y, tx, ty)
            .split(tx, tx_serial_outer, tx, 16,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 8,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_min.compute_at(output, x)
            .reorder(tx, ty)
            .split(tx, tx_serial_outer, tx, 18,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 10,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_min.update(0)
            .reorder(r10_x, r10_y, tx, ty)
            .split(tx, tx_serial_outer, tx, 18,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 10,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        neigh_max.split(tx, tx, txi, 32, Halide::TailStrategy::RoundUp)
            .split(ty, ty, tyi, 8, Halide::TailStrategy::RoundUp)
            .split(txi, txi, txii, 2, Halide::TailStrategy::RoundUp)
            .unroll(txii)
            .compute_root()
            .reorder(txii, txi, tyi, tx, ty)
            .gpu_blocks(tx)
            .gpu_blocks(ty)
            .split(txi, txi_serial_outer, txi, 16,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(txi)
            .split(tyi, tyi_serial_outer, tyi, 8,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tyi);
        neigh_max.update(0)
            .split(tx, tx, txi, 32, Halide::TailStrategy::RoundUp)
            .split(ty, ty, tyi, 8, Halide::TailStrategy::RoundUp)
            .split(txi, txi, txii, 2, Halide::TailStrategy::GuardWithIf)
            .unroll(txii)
            .reorder(txii, r23_x, r23_y, txi, tyi, tx, ty)
            .gpu_blocks(tx)
            .gpu_blocks(ty)
            .split(txi, txi_serial_outer, txi, 16,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(txi)
            .split(tyi, tyi_serial_outer, tyi, 8,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tyi);
        tile_max.compute_at(neigh_max, tx)
            .reorder(tx, ty)
            .split(tx, tx_serial_outer, tx, 34,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 10,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        tile_max.update(0)
            .reorder(r10_x, r10_y, tx, ty)
            .split(tx, tx_serial_outer, tx, 34,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(tx)
            .split(ty, ty_serial_outer, ty, 10,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(ty);
        padded.split(_0, _0, _0i, 64, Halide::TailStrategy::ShiftInwards)
            .split(_1, _1, _1i, 4, Halide::TailStrategy::ShiftInwards)
            .split(_1i, _1i, _1ii, 4, Halide::TailStrategy::ShiftInwards)
            .unroll(_1ii)
            .compute_root()
            .reorder(_1ii, _0i, _1i, _0, _1)
            .gpu_blocks(_0)
            .gpu_blocks(_1)
            .split(_0i, _0i_serial_outer, _0i, 64,
                   Halide::TailStrategy::GuardWithIf)
            .gpu_threads(_0i);
      } else {
        // Manual scheduling tuned for CPU parallelism; must be done
        // before creating the pipeline
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
        output.compute_root()
            .tile(x, y, xo, yo, xi, yi, 64, 32)
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
      }
    }
  }
};

HALIDE_REGISTER_GENERATOR(Threshold, threshold);
