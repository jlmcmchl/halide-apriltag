#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "apriltag_binary.h"
#include "apriltag_edge.h"
#include "apriltag_density.h"
#include "apriltag_lut.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <numeric>

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

namespace {

// Fast quad detection structures
struct Point2D {
    float x, y;
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
    Point2D operator+(const Point2D& other) const { return Point2D(x + other.x, y + other.y); }
    Point2D operator-(const Point2D& other) const { return Point2D(x - other.x, y - other.y); }
    Point2D operator*(float s) const { return Point2D(x * s, y * s); }
    float dot(const Point2D& other) const { return x * other.x + y * other.y; }
    float norm() const { return std::sqrt(x * x + y * y); }
    
    // Comparison operators for std::sort
    bool operator<(const Point2D& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    bool operator==(const Point2D& other) const {
        return x == other.x && y == other.y;
    }
};

struct Quad {
    Point2D corners[4];
    float error;
    Quad() : error(std::numeric_limits<float>::max()) {}
};

#include <vector>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "HalideBuffer.h"   // Halide::Runtime::Buffer

struct Rect {
    float x1, y1, x2, y2, x3, y3, x4, y4; // corners in order
    float angle; // radians
};

static inline float sqr(float v) { return v * v; }

static inline float dist(float x1, float y1, float x2, float y2) {
    return std::sqrt(sqr(x2 - x1) + sqr(y2 - y1));
}

static inline float dot(float ax, float ay, float bx, float by) {
    return ax * bx + ay * by;
}

std::vector<Rect> find_rotated_rects(Buffer<int32_t> &map,
                                     float aspect_tol = 3.0f,
                                     float angle_tol_deg = 30.0f)
{
    const int W = map.width();
    const int H = map.height();

    // --- Gather corners ---
    std::vector<std::pair<int,int>> tl, tr, bl, br;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            switch (map(x, y)) {
                case 1: tl.emplace_back(x, y); break;
                case 2: tr.emplace_back(x, y); break;
                case 3: bl.emplace_back(x, y); break;
                case 4: br.emplace_back(x, y); break;
                default: break;
            }
        }
    }

    // Hash map for quick lookup of BR corners
    std::unordered_map<int, bool> br_map;
    br_map.reserve(br.size() * 2);
    for (auto &[x, y] : br)
        br_map[y * W + x] = true;

    std::vector<Rect> rects;
    float angle_tol = angle_tol_deg * M_PI / 180.0f;

    // --- Match candidates ---
    for (auto &[x1, y1] : tl) {
        for (auto &[x2, y2] : tr) {
            // top edge vector
            float dx1 = x2 - x1, dy1 = y2 - y1;
            if (dx1 == 0 && dy1 == 0) continue;
            float len_top = std::sqrt(dx1*dx1 + dy1*dy1);

            // find a BL candidate
            for (auto &[x3, y3] : bl) {
                float dx2 = x3 - x1, dy2 = y3 - y1;
                if (dx2 == 0 && dy2 == 0) continue;
                float len_left = std::sqrt(dx2*dx2 + dy2*dy2);

                // Check angle between edges ≈ 90°
                float cosang = dot(dx1, dy1, dx2, dy2) / (len_top * len_left);
                if (std::abs(cosang) > std::cos(angle_tol)) continue;

                // Expected BR position
                float x4 = x3 + dx1;
                float y4 = y3 + dy1;

                int xi4 = std::round(x4);
                int yi4 = std::round(y4);
                if (xi4 < 0 || xi4 >= W || yi4 < 0 || yi4 >= H) continue;
                if (!br_map.count(yi4 * W + xi4)) continue;

                // Aspect ratio check
                float ratio = len_top / len_left;
                if (ratio < 1.0f / aspect_tol || ratio > aspect_tol) continue;

                rects.push_back({(float)x1, (float)y1, (float)x2, (float)y2,
                                 (float)x3, (float)y3, x4, y4,
                                 std::atan2(dy1, dx1)});
            }
        }
    }

    return rects;
}

void draw_line(Halide::Runtime::Buffer<int32_t> &buf,
               float x0, float y0, float x1, float y1, uint8_t val)
{
    int W = buf.width(), H = buf.height();
    int dx = std::abs((int)(x1 - x0)), dy = std::abs((int)(y1 - y0));
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
            buf((int)x0, (int)y0) = val;
        if ((int)x0 == (int)x1 && (int)y0 == (int)y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

void draw_rects(Halide::Runtime::Buffer<int32_t> &buf,
                const std::vector<Rect> &rects,
                uint8_t color = 255)
{
    for (auto &r : rects) {
        draw_line(buf, r.x1, r.y1, r.x2, r.y2, color);
        draw_line(buf, r.x2, r.y2, r.x4, r.y4, color);
        draw_line(buf, r.x4, r.y4, r.x3, r.y3, color);
        draw_line(buf, r.x3, r.y3, r.x1, r.y1, color);
    }
}


// Helper function to record equivalence between two labels (optimized)
inline void record_equivalence(std::vector<int>& parent, int label1, int label2) {
    int max_label = std::max(label1, label2);
    int min_label = std::min(label1, label2);
    
    // Only resize if absolutely necessary
    if (max_label >= parent.size()) {
        int old_size = parent.size();
        parent.resize(max_label + 1);
        // Initialize new entries as self-parents
        for (int i = old_size; i <= max_label; ++i) {
            parent[i] = i;
        }
    }
    parent[max_label] = min_label;
}

// Maximum performance connected components with SIMD hints and branch prediction
Buffer<int32_t> ultra_fast_connected_components(const Buffer<int32_t> &edges) {
    int width = edges.width();
    int height = edges.height();
    
    Buffer<int32_t> labels(width, height);
    
    // Initialize labels to 0 using memset for speed
    memset(labels.data(), 0, width * height * sizeof(int32_t));
    
    // Union-find structure for label equivalences - pre-allocate for speed
    std::vector<int> parent;
    parent.reserve(width * height / 2); // More conservative estimate for 8-connectivity
    int current_label = 1;
    
    // First pass: assign initial labels and build equivalence table
    for (int y = 0; y < height; ++y) {
        int32_t* label_row = &labels(0, y);
        const int32_t* edge_row = &edges(0, y);
        
        for (int x = 0; x < width; ++x) {
            // Branch prediction hint: most pixels are background
            if (__builtin_expect(edge_row[x] > 0, 0)) {
                // Check 8-connectivity: left, top-left, top, top-right
                int left_label = (x > 0) ? label_row[x - 1] : 0;
                int top_label = (y > 0) ? labels(x, y - 1) : 0;
                int top_left_label = (x > 0 && y > 0) ? labels(x - 1, y - 1) : 0;
                int top_right_label = (x < width - 1 && y > 0) ? labels(x + 1, y - 1) : 0;
                
                // Find the minimum non-zero label among neighbors (optimized)
                int min_neighbor_label = 0;
                if (left_label > 0) min_neighbor_label = left_label;
                if (top_label > 0) {
                    min_neighbor_label = (min_neighbor_label == 0) ? top_label : std::min(min_neighbor_label, top_label);
                }
                if (top_left_label > 0) {
                    min_neighbor_label = (min_neighbor_label == 0) ? top_left_label : std::min(min_neighbor_label, top_left_label);
                }
                if (top_right_label > 0) {
                    min_neighbor_label = (min_neighbor_label == 0) ? top_right_label : std::min(min_neighbor_label, top_right_label);
                }
                
                // Optimize the most common case first
                if (__builtin_expect(min_neighbor_label == 0, 1)) {
                    // New component
                    label_row[x] = current_label;
                    // Ensure parent vector is large enough and initialize self-parent
                    if (current_label >= parent.size()) {
                        parent.resize(current_label + 1);
                    }
                    parent[current_label] = current_label; // Self-parent
                    current_label++;
                } else {
                    // We have at least one neighbor with a label
                    label_row[x] = min_neighbor_label;
                    
                    // Record equivalences for all neighbors that have different labels
                    // Use a more efficient approach - check all neighbors at once
                    int neighbors[4] = {left_label, top_label, top_left_label, top_right_label};
                    for (int i = 0; i < 4; ++i) {
                        if (neighbors[i] > 0 && neighbors[i] != min_neighbor_label) {
                            record_equivalence(parent, neighbors[i], min_neighbor_label);
                        }
                    }
                }
            }
        }
    }
    
    // Resolve equivalences using iterative path compression (faster than recursive)
    // This ensures all labels point directly to their root
    for (int i = 1; i < parent.size(); ++i) {
        if (parent[i] != i) {
            // Find root with path compression
            int root = i;
            while (parent[root] != root) {
                root = parent[root];
            }
            // Path compression: make all nodes on path point directly to root
            int current = i;
            while (parent[current] != root) {
                int next = parent[current];
                parent[current] = root;
                current = next;
            }
        }
    }
    
    // Second pass: resolve all labels to their root labels
    for (int y = 0; y < height; ++y) {
        int32_t* label_row = &labels(0, y);
        for (int x = 0; x < width; ++x) {
            if (__builtin_expect(label_row[x] > 0, 0)) {
                // Safety check: ensure label is within parent array bounds
                if (label_row[x] < parent.size()) {
                    label_row[x] = parent[label_row[x]];
                }
                // If somehow out of bounds, keep original label (shouldn't happen)
            }
        }
    }
    
    return labels;
}

// Ultra-fast quad detection with clean bounding boxes
std::vector<Quad> ultra_fast_quad_detection(const Buffer<int32_t> &labels) {
    std::vector<Quad> quads;
    int width = labels.width();
    int height = labels.height();
    
    // Find all unique labels and their boundary points
    std::unordered_map<int32_t, std::vector<Point2D>> component_points;
    
    // Extract boundary points for each component (4-connectivity for speed)
    for (int y = 1; y < height - 1; ++y) {
        const int32_t* label_row = &labels(0, y);
        
        for (int x = 1; x < width - 1; ++x) {
            int32_t current_label = label_row[x];
            if (current_label == 0) continue;
            
            // Check if this is a boundary point
            if (label_row[x - 1] != current_label || label_row[x + 1] != current_label ||
                labels(x, y - 1) != current_label || labels(x, y + 1) != current_label) {
                component_points[current_label].emplace_back(x, y);
            }
        }
    }
    
    // Process each component for quad detection
    for (auto& [label, points] : component_points) {
        if (points.size() < 4) continue; // Minimum for quad
        
        // Find bounding box
        float min_x = points[0].x, max_x = points[0].x;
        float min_y = points[0].y, max_y = points[0].y;
        
        for (const auto& p : points) {
            if (p.x < min_x) min_x = p.x;
            if (p.x > max_x) max_x = p.x;
            if (p.y < min_y) min_y = p.y;
            if (p.y > max_y) max_y = p.y;
        }
        
        float width_bb = max_x - min_x;
        float height_bb = max_y - min_y;
        
        // Better filtering for AprilTag-like objects
        if (width_bb < 15 || height_bb < 15) continue; // Too small
        if (width_bb > width/2 || height_bb > height/2) continue; // Too large
        
        // Check aspect ratio (AprilTags are roughly square)
        float aspect_ratio = width_bb / height_bb;
        if (aspect_ratio < 0.5f || aspect_ratio > 2.0f) continue; // Not square enough
        
        // Create clean quad from bounding box
        Quad quad;
        quad.corners[0] = Point2D(min_x, min_y); // Top-left
        quad.corners[1] = Point2D(max_x, min_y); // Top-right
        quad.corners[2] = Point2D(max_x, max_y); // Bottom-right
        quad.corners[3] = Point2D(min_x, max_y); // Bottom-left
        
        // Simple error metric
        quad.error = width_bb + height_bb;
        
        quads.push_back(quad);
    }
    
    return quads;
}

Buffer<float> convert_to_grayscale(const Buffer<uint8_t> &input) {
    Buffer<float> gray(input.width(), input.height());

    if (input.channels() == 1) {
        for (int y = 0; y < input.height(); ++y) {
            for (int x = 0; x < input.width(); ++x) {
                gray(x, y) = static_cast<float>(input(x, y));
            }
        }
    } else {
        for (int y = 0; y < input.height(); ++y) {
            for (int x = 0; x < input.width(); ++x) {
                float r = static_cast<float>(input(x, y, 0));
                float g = static_cast<float>(input(x, y, 1));
                float b = static_cast<float>(input(x, y, 2));
                gray(x, y) = 0.299f * r + 0.587f * g + 0.114f * b;
            }
        }
    }

    return gray;
}

Buffer<uint8_t> visualize_labels(const Buffer<int32_t> &labels) {
    Buffer<uint8_t> vis(labels.width(), labels.height(), 3);

    auto label_to_color = [](int32_t label) -> std::array<uint8_t, 3> {
        uint32_t h = static_cast<uint32_t>(label) * 2654435761u;
        uint8_t r = static_cast<uint8_t>(h & 0xFF);
        uint8_t g = static_cast<uint8_t>((h >> 8) & 0xFF);
        uint8_t b = static_cast<uint8_t>((h >> 16) & 0xFF);
        if (label != 0 && (r == 0 && g == 0 && b == 0)) {
            r = g = b = 128;
        }
        return {r, g, b};
    };

    for (int y = 0; y < labels.height(); ++y) {
        for (int x = 0; x < labels.width(); ++x) {
            int32_t label = labels(x, y);
            if (label == 0) {
                vis(x, y, 0) = 0;
                vis(x, y, 1) = 0;
                vis(x, y, 2) = 0;
            } else {
                auto color = label_to_color(label);
                vis(x, y, 0) = color[0];
                vis(x, y, 1) = color[1];
                vis(x, y, 2) = color[2];
            }
        }
    }

    return vis;
}

Buffer<uint8_t> visualize_quads(const Buffer<uint8_t> &input, const std::vector<Quad> &quads) {
    Buffer<uint8_t> vis(input.width(), input.height(), 3);
    
    // Copy input image
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            if (input.channels() == 1) {
                vis(x, y, 0) = vis(x, y, 1) = vis(x, y, 2) = input(x, y);
            } else {
                vis(x, y, 0) = input(x, y, 0);
                vis(x, y, 1) = input(x, y, 1);
                vis(x, y, 2) = input(x, y, 2);
            }
        }
    }
    
    // Draw quads in bright colors
    for (const auto& quad : quads) {
        // Draw quad outline
        for (int i = 0; i < 4; ++i) {
            Point2D p1 = quad.corners[i];
            Point2D p2 = quad.corners[(i + 1) % 4];
            
            // Draw line between p1 and p2
            int steps = std::max(1, (int)std::abs(p2.x - p1.x) + (int)std::abs(p2.y - p1.y));
            for (int step = 0; step <= steps; ++step) {
                float t = (float)step / steps;
                int x = (int)(p1.x + t * (p2.x - p1.x));
                int y = (int)(p1.y + t * (p2.y - p1.y));
                
                if (x >= 0 && x < vis.width() && y >= 0 && y < vis.height()) {
                    vis(x, y, 0) = 255; // Red
                    vis(x, y, 1) = 0;
                    vis(x, y, 2) = 0;
                }
            }
        }
        
        // Draw corner points
        for (int i = 0; i < 4; ++i) {
            int x = (int)quad.corners[i].x;
            int y = (int)quad.corners[i].y;
            
            // Draw small circle around corner
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    int px = x + dx;
                    int py = y + dy;
                    if (px >= 0 && px < vis.width() && py >= 0 && py < vis.height()) {
                        if (dx * dx + dy * dy <= 4) {
                            vis(px, py, 0) = 0;   // Green corners
                            vis(px, py, 1) = 255;
                            vis(px, py, 2) = 0;
                        }
                    }
                }
            }
        }
    }
    
    return vis;
}

void check_halide(int result, const char *stage, std::chrono::duration<double> duration) {
    if (result != 0) {
        throw std::runtime_error(std::string("Halide pipeline failure at ") + stage + ": " + std::to_string(result));
    }
    std::cout << "Stage '" << stage << "' completed in " << duration.count() * 1000 << " milliseconds" << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const char *input_path = (argc > 1) ? argv[1] : "../apriltags.png";
        
        // Parse density filter parameters
        int density_size = 7;           // Default 3x3 neighborhood
        float density_ratio = 0.75f;      // Default 50% threshold
        
        if (argc > 2) density_size = std::atoi(argv[2]);
        if (argc > 3) density_ratio = std::atof(argv[3]);
        
        std::cout << "Loading " << input_path << std::endl;
        std::cout << "Density filter: " << density_size << "x" << density_size 
                  << " neighborhood, " << (density_ratio * 100) << "% threshold" << std::endl;
        Buffer<uint8_t> input = load_image(input_path);

        std::cout << "Input dimensions: " << input.width() << "x" << input.height() << "x" << input.channels() << std::endl;

        Buffer<float> gray = convert_to_grayscale(input);

        Buffer<int32_t> binary(gray.width(), gray.height());
        auto start = std::chrono::high_resolution_clock::now();
        int result = atag_binary(gray, binary);
        auto end = std::chrono::high_resolution_clock::now();

        auto start_copy = std::chrono::high_resolution_clock::now();
        binary.copy_to_host();
        auto end_copy = std::chrono::high_resolution_clock::now();
        std::cout << "Copy to host completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_copy - start_copy).count() << " milliseconds" << std::endl;

        auto start_label = std::chrono::high_resolution_clock::now();
        // for(int y = binary.height() - 1; y > 0; y--) {
        //     for(int x = binary.width() - 1; x > 0; x--) {
        //         if(__builtin_expect(binary(x, y) > 0, 0)) {
        //             if(__builtin_expect(binary(x+1, y) > 0, 0)) {
        //                 int32_t maximum = std::max(binary(x, y), binary(x+1, y));
        //                 maximum = std::max(maximum, binary(x, y+1));
        //                 maximum = std::max(maximum, binary(x+1, y+1));
        //                 maximum = std::max(maximum, binary(x, y-1));
        //                 maximum = std::max(maximum, binary(x+1, y-1));
        //                 maximum = std::max(maximum, binary(x-1, y));
        //                 maximum = std::max(maximum, binary(x-1, y+1));
        //                 maximum = std::max(maximum, binary(x-1, y-1));
        //                 binary(x, y) = maximum;
        //             }
        //         }
        //     }
        // }
        // for(int x = binary.width() - 1; x > 0; x--) {
        //     for(int y = binary.height() - 1; y > 0; y--) {
        //         if(__builtin_expect(binary(x, y) > 0, 0)) {
        //             if(__builtin_expect(binary(x, y+1) > 0, 0)) {
        //                 binary(x, y) = binary(x, y+1);
        //             }
        //         }
        //     }
        // }
        auto end_label = std::chrono::high_resolution_clock::now();
        std::cout << "Labeling completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_label - start_label).count() << " milliseconds" << std::endl;
        check_halide(result, "binary", end - start);

        gray.device_free();
        input.device_free();
 

        Buffer<uint8_t> binary_vis(binary.width(), binary.height());
        binary_vis.device_free();
        binary_vis.copy_to_host();
        // for (int y = 0; y < binary.height(); ++y) {
        //     for (int x = 0; x < binary.width(); ++x) {
        //         if (binary(x, y) > 0) {
        //             binary_vis(x, y, 0) = binary(x, y) % 255;
        //         }else{
        //             binary_vis(x, y, 0) = 0;
        //         }
        //     }
        // }

        auto rects = find_rotated_rects(binary);

        printf("Detected %zu rectangles\n", rects.size());

        draw_rects(binary, rects, 200);

        for (int y = 0; y < binary.height(); ++y) {
            for (int x = 0; x < binary.width(); ++x) {
                if (binary(x, y) > 0) {
                    binary_vis(x, y, 0) = binary(x, y) % 255;
                }else{
                    binary_vis(x, y, 0) = 0;
                }
            }
        }
        
        save_image(binary_vis, "gradient_otsu.png");

        return 0;

        binary.device_free();
        binary_vis.device_free();

        Buffer<int32_t> edges(gray.width(), gray.height());
        start = std::chrono::high_resolution_clock::now();
        result = atag_edge(binary, edges);
        end = std::chrono::high_resolution_clock::now();
        check_halide(result, "edge", end - start);

        Buffer<uint8_t> edges_vis(edges.width(), edges.height(), 3);
        for (int y = 0; y < edges.height(); ++y) {
            for (int x = 0; x < edges.width(); ++x) {
                if (edges(x, y) > 0) {
                    edges_vis(x, y, 0) = 255;
                }
            }
        }

        save_image(edges_vis, "edge.png");

        // Ultra-fast density filter with configurable parameters
        // Buffer<int32_t> density(gray.width(), gray.height());
        // start = std::chrono::high_resolution_clock::now();
        // result = atag_density(edges, density_size, density_ratio, density);
        // end = std::chrono::high_resolution_clock::now();
        // check_halide(result, "ultra_fast_density", end - start);

        Buffer<int32_t> lut(gray.width(), gray.height());
        start = std::chrono::high_resolution_clock::now();
        result = atag_lut(edges, lut);
        end = std::chrono::high_resolution_clock::now();
        check_halide(result, "lut", end - start);

        // Use fast union-find connected components labeling
        start = std::chrono::high_resolution_clock::now();
        
        // Apply ultra-fast connected components labeling on density-filtered edges
        Buffer<int32_t> labels = ultra_fast_connected_components(edges);
        
        end = std::chrono::high_resolution_clock::now();
        check_halide(0, "fast_connected_components", end - start);
        
        std::cout << "Fast connected components completed" << std::endl;

        Buffer<uint8_t> labels_vis = visualize_labels(labels);
        save_image(labels_vis, "labelling_output.png");

        // Ultra-fast quad detection
        start = std::chrono::high_resolution_clock::now();
        std::vector<Quad> quads = ultra_fast_quad_detection(labels);
        end = std::chrono::high_resolution_clock::now();
        check_halide(0, "quad_detection", end - start);
        
        std::cout << "Found " << quads.size() << " candidate quads" << std::endl;

        // Visualize quads on original image
        Buffer<uint8_t> quads_vis = visualize_quads(input, quads);
        save_image(quads_vis, "quad_detection_output.png");

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
