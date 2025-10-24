#include "Halide.h"

#include <vector>

using namespace Halide;

namespace {

Func build_binary_pipeline(ImageParam input) {
    Var x("x"), y("y");

    // Use a fixed threshold instead of computing global mean
    Expr threshold = cast<float>(50.0f);

    /**
        RDom r(0, input.width(), 0, input.height());
    Expr total_sum = sum(input(r.x, r.y));
    Expr total_pixels = cast<float>(input.width() * input.height());
    Expr mean = total_sum / total_pixels; */

    Func binary("binary");
    binary(x, y) = cast<uint8_t>(select(input(x, y) > threshold, 50, 0));

    binary.compute_root();

    // Disable GPU tiling for CPU-only compilation
    // Var xi("xi"), yi("yi");
    // binary.gpu_tile(x, y, xi, yi, 16, 16);

    return binary;
}

Func build_edge_pipeline(ImageParam binary_in) {
    Var x("x"), y("y");

    Expr max_x = binary_in.dim(0).extent() - 1;
    Expr max_y = binary_in.dim(1).extent() - 1;

    Expr left = binary_in(clamp(x - 1, 0, max_x), y);
    Expr right = binary_in(clamp(x + 1, 0, max_x), y);
    Expr up = binary_in(x, clamp(y - 1, 0, max_y));
    Expr down = binary_in(x, clamp(y + 1, 0, max_y));

    Expr is_edge = (left + right == 50) || (up + down == 50);

    Func edge("edge");
    edge(x, y) = select(is_edge, 100, 0);

    edge.compute_root();

    // Temporarily disable GPU tiling to isolate the issue
    // Var xi("xi"), yi("yi");
    // edge.gpu_tile(x, y, xi, yi, 16, 16);

    return edge;
}

Func build_density_pipeline(ImageParam edge_in) {
    Var x("x"), y("y");

    Expr max_x = edge_in.dim(0).extent() - 1;
    Expr max_y = edge_in.dim(1).extent() - 1;

    const int density_radius = 3;
    RDom neighborhood(-density_radius, 2 * density_radius + 1,
                      -density_radius, 2 * density_radius + 1);

    Expr nx = clamp(x + neighborhood.x, 0, max_x);
    Expr ny = clamp(y + neighborhood.y, 0, max_y);
    Expr neighborhood_count = sum(select(edge_in(nx, ny) > 0, 1, 0));
    Expr max_active = ((2 * density_radius + 1) * (2 * density_radius + 1)) * 0.5f;

    Func density("density");
    density(x, y) = cast<int32_t>(select(neighborhood_count > max_active, 0, edge_in(x, y)));

    //density.compute_root();

    // Temporarily disable GPU tiling to debug assertion error
    density.compute_root();
    // Var xi("xi"), yi("yi");
    // density.gpu_tile(x, y, xi, yi, 16, 16);

    return density;
}

Func build_lut_pipeline(ImageParam density_in) {
    Var x("x"), y("y");

    Expr width = cast<int32_t>(density_in.dim(0).extent());

    Func lut("lut");
    Expr idx = cast<int32_t>(x) + cast<int32_t>(y) * width;
    lut(x, y) = select(density_in(x, y) > 0, idx + 1, 0);

    lut.compute_root();

    // Temporarily disable GPU tiling to debug assertion error
    // Var xi("xi"), yi("yi");
    // lut.gpu_tile(x, y, xi, yi, 16, 16);

    return lut;
}


} // namespace

int main(int argc, char **argv) {
    try {
        Target target = get_target_from_environment();

    ImageParam input_gray(Float(32), 2, "input_gray");
    ImageParam binary_in(UInt(8), 2, "binary_in");
    ImageParam edge_in(Int(32), 2, "edge_in");
    ImageParam density_in(Int(32), 2, "density_in");
    ImageParam labels_in(Int(32), 2, "labels_in");

    Func binary = build_binary_pipeline(input_gray);
    Func edge = build_edge_pipeline(binary_in);
    Func density = build_density_pipeline(edge_in);
    Func lut = build_lut_pipeline(density_in);

    std::vector<Argument> binary_args = {input_gray};
    std::vector<Argument> edge_args = {binary_in};
    std::vector<Argument> density_args = {edge_in};
    std::vector<Argument> lut_args = {density_in};

    // Compile all pipelines together to avoid duplicate Metal runtime symbols
    // but expose them as separate functions by compiling each individually with the same target
    Target gpu_target = target;
    gpu_target.set_feature(Target::Metal);
    gpu_target.set_feature(Target::Profile);

    // Compile each pipeline separately but with the same target to share runtime
    binary.compile_to_static_library("apriltag_binary", binary_args, "atag_binary", gpu_target);
    edge.compile_to_static_library("apriltag_edge", edge_args, "atag_edge", gpu_target);
    density.compile_to_static_library("apriltag_density", density_args, "atag_density", gpu_target);
    lut.compile_to_static_library("apriltag_lut", lut_args, "atag_lut", gpu_target);

    binary.compile_to_header("apriltag_binary.h", binary_args, "atag_binary", gpu_target);
    edge.compile_to_header("apriltag_edge.h", edge_args, "atag_edge", gpu_target);
    density.compile_to_header("apriltag_density.h", density_args, "atag_density", gpu_target);
    lut.compile_to_header("apriltag_lut.h", lut_args, "atag_lut", gpu_target);

    return 0;
    } catch (const Halide::CompileError &e) {
        std::cerr << "Halide compilation error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
