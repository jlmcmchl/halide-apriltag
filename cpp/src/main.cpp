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
        std::cout << "Loading " << input_path << std::endl;
        Buffer<uint8_t> input = load_image(input_path);

        std::cout << "Input dimensions: " << input.width() << "x" << input.height() << "x" << input.channels() << std::endl;

        Buffer<float> gray = convert_to_grayscale(input);

        Buffer<uint8_t> binary(gray.width(), gray.height());
        auto start = std::chrono::high_resolution_clock::now();
        int result = atag_binary(gray, binary);
        auto end = std::chrono::high_resolution_clock::now();
        check_halide(result, "binary", end - start);
        
        save_image(binary, "gradient_otsu.png");

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

        // Buffer<int32_t> density(gray.width(), gray.height());
        // start = std::chrono::high_resolution_clock::now();
        // result = atag_density(edges, density);
        // end = std::chrono::high_resolution_clock::now();
        // check_halide(result, "density", end - start);

        Buffer<int32_t> lut(gray.width(), gray.height());
        start = std::chrono::high_resolution_clock::now();
        result = atag_lut(edges, lut);
        end = std::chrono::high_resolution_clock::now();
        check_halide(result, "lut", end - start);

        // Use fast union-find connected components labeling
        start = std::chrono::high_resolution_clock::now();
        
        // Apply ultra-fast connected components labeling directly on edges
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
