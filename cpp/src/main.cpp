#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "greyscale_and_adaptive_threshold.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Halide::Runtime::Buffer;
using namespace Halide::Tools;

// =============================================================================
// Data Structures
// =============================================================================

struct Point2D {
  float x, y;
  Point2D(float x = 0, float y = 0) : x(x), y(y) {}
  Point2D operator+(const Point2D &o) const { return {x + o.x, y + o.y}; }
  Point2D operator-(const Point2D &o) const { return {x - o.x, y - o.y}; }
  Point2D operator*(float s) const { return {x * s, y * s}; }
  float dot(const Point2D &o) const { return x * o.x + y * o.y; }
  float cross(const Point2D &o) const { return x * o.y - y * o.x; }
  float norm() const { return std::sqrt(x * x + y * y); }
  Point2D normalized() const {
    float n = norm();
    return n > 0 ? Point2D(x / n, y / n) : Point2D();
  }
};

struct Quad {
  Point2D corners[4];
  float confidence;

  bool is_valid() const {
    float side_lengths[4];
    for (int i = 0; i < 4; i++) {
      side_lengths[i] = (corners[(i + 1) % 4] - corners[i]).norm();
    }

    float min_side = *std::min_element(side_lengths, side_lengths + 4);
    float max_side = *std::max_element(side_lengths, side_lengths + 4);

    if (min_side < 6.0f)
      return false;
    if (max_side / min_side > 7.0f)
      return false;

    // Check convexity
    for (int i = 0; i < 4; i++) {
      Point2D v1 = corners[(i + 1) % 4] - corners[i];
      Point2D v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4];
      if (v1.cross(v2) < 0)
        return false;
    }
    return true;
  }

  float perimeter() const {
    float perimeter = 0;
    for (int i = 0; i < 4; i++) {
      perimeter += (corners[(i + 1) % 4] - corners[i]).norm();
    }
    return perimeter;
  }

  float area() const {
    return std::abs(corners[0].cross(corners[2]) -
                    corners[1].cross(corners[3])) /
           2;
  }

  float parallelness() const {
    Point2D sides[4];
    for (int i = 0; i < 4; i++) {
      sides[i] = (corners[(i + 1) % 4] - corners[i]);
    }

    float pair_1 = std::abs((sides[0]).cross(sides[2])) /
                   (sides[0].norm() * sides[2].norm());
    float pair_2 = std::abs(sides[1].cross(sides[3])) /
                   (sides[1].norm() * sides[3].norm());
    return pair_1 + pair_2;
  }
};

// =============================================================================
// Union-Find for Connected Components
// =============================================================================

class UnionFind {
public:
  std::vector<int> parent, rank_vec;

  UnionFind(int n) : parent(n), rank_vec(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    if (parent[x] != x)
      parent[x] = find(parent[x]);
    return parent[x];
  }

  void unite(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py)
      return;
    if (rank_vec[px] < rank_vec[py])
      std::swap(px, py);
    parent[py] = px;
    if (rank_vec[px] == rank_vec[py])
      rank_vec[px]++;
  }
};

// =============================================================================
// Convex Hull (Andrew's Monotone Chain)
// =============================================================================

std::vector<Point2D> convex_hull(std::vector<Point2D> points) {
  int n = points.size();
  if (n < 3)
    return points;

  std::sort(points.begin(), points.end(),
            [](const Point2D &a, const Point2D &b) {
              return a.x < b.x || (a.x == b.x && a.y < b.y);
            });

  std::vector<Point2D> hull;

  // Lower hull
  for (int i = 0; i < n; i++) {
    while (hull.size() >= 2) {
      Point2D a = hull[hull.size() - 2];
      Point2D b = hull[hull.size() - 1];
      if ((b - a).cross(points[i] - a) <= 0)
        hull.pop_back();
      else
        break;
    }
    hull.push_back(points[i]);
  }

  // Upper hull
  int lower_size = hull.size();
  for (int i = n - 2; i >= 0; i--) {
    while (hull.size() > lower_size) {
      Point2D a = hull[hull.size() - 2];
      Point2D b = hull[hull.size() - 1];
      if ((b - a).cross(points[i] - a) <= 0)
        hull.pop_back();
      else
        break;
    }
    hull.push_back(points[i]);
  }

  hull.pop_back();
  return hull;
}

// =============================================================================
// Convex Hull (QuickHull Algorithm)
// =============================================================================

// Distance from point p to line defined by a and b
float point_line_distance(const Point2D &p, const Point2D &a,
                          const Point2D &b) {
  return std::abs((b - a).cross(p - a));
}

// Find points on one side of line from a to b
void partition_points(const std::vector<Point2D> &points, const Point2D &a,
                      const Point2D &b, std::vector<Point2D> &result) {
  for (const auto &p : points) {
    float cross = (b - a).cross(p - a);
    if (cross > 0) { // Point is on left side of line
      result.push_back(p);
    }
  }
}

// Recursive function to find hull points on one side
void quickhull_recursive(const std::vector<Point2D> &points, const Point2D &a,
                         const Point2D &b, std::vector<Point2D> &hull) {
  if (points.empty())
    return;

  // Find farthest point from line ab
  float max_dist = -1;
  Point2D farthest;

  for (const auto &p : points) {
    float dist = point_line_distance(p, a, b);
    if (dist > max_dist) {
      max_dist = dist;
      farthest = p;
    }
  }

  // Partition points into two sets relative to lines a-farthest and
  // farthest-b
  std::vector<Point2D> left_set, right_set;

  for (const auto &p : points) {
    // Skip the farthest point itself
    if (std::abs(p.x - farthest.x) < 1e-6 &&
        std::abs(p.y - farthest.y) < 1e-6) {
      continue;
    }
    if ((farthest - a).cross(p - a) > 0) {
      left_set.push_back(p);
    } else if ((b - farthest).cross(p - farthest) > 0) {
      right_set.push_back(p);
    }
  }

  // Recursively find hull on left side of a-farthest
  quickhull_recursive(left_set, a, farthest, hull);
  // Add the farthest point to hull
  hull.push_back(farthest);

  // Recursively find hull on left side of farthest-b
  quickhull_recursive(right_set, farthest, b, hull);
}

std::vector<Point2D> quickhull(std::vector<Point2D> blob) {
  if (blob.size() < 3)
    return blob;
  // Find leftmost and rightmost points
  auto minmax_x = std::minmax_element(
      blob.begin(), blob.end(), [](const Point2D &a, const Point2D &b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
      });

  Point2D left = *minmax_x.first;
  Point2D right = *minmax_x.second;

  // Partition points into upper and lower sets
  std::vector<Point2D> upper_set, lower_set;

  for (const auto &p : blob) {
    // Skip left and right points
    if ((std::abs(p.x - left.x) < 1e-6 && std::abs(p.y - left.y) < 1e-6) ||
        (std::abs(p.x - right.x) < 1e-6 && std::abs(p.y - right.y) < 1e-6)) {
      continue;
    }

    float cross = (right - left).cross(p - left);
    if (cross > 0) {
      upper_set.push_back(p);
    } else if (cross < 0) {
      lower_set.push_back(p);
    }
  }

  // Build hull starting from leftmost point
  std::vector<Point2D> hull;
  hull.push_back(left);

  // Find upper hull
  quickhull_recursive(upper_set, left, right, hull);
  hull.push_back(right);

  // Find lower hull
  quickhull_recursive(lower_set, right, left, hull);
  std::reverse(hull.begin(), hull.end());
  return hull;
}

// =============================================================================
// Quad Fitting from Hull
// =============================================================================

std::tuple<int, int> farthest_points(const std::vector<Point2D> &hull) {
  int n = hull.size();
  float best_distance = 0;
  int best_index1 = 0;
  int best_index2 = 0;
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      float distance = (hull[i] - hull[j]).norm();
      if (distance > best_distance) {
        best_distance = distance;
        best_index1 = i;
        best_index2 = j;
      }
    }
  }
  return std::make_tuple(best_index1, best_index2);
}

int point_making_largest_triangle(const std::vector<Point2D> &hull, int index1,
                                  int index2) {
  int n = hull.size();
  float best_area = 0;
  int best_index = -1;
  for (int i = (index1 + 1) % n; i != index2; i = (i + 1) % n) {
    float area =
        std::abs((hull[i] - hull[index1]).cross(hull[index2] - hull[index1]));
    if (area > best_area) {
      best_area = area;
      best_index = i;
    }
  }
  return best_index;
}

Quad fit_quad_to_hull(const std::vector<Point2D> &hull, float total_perimeter) {
  Quad quad;
  quad.confidence = total_perimeter;

  if (hull.size() < 4)
    return quad;

  std::vector<int> corner_indices;

  if (true) {
    auto [index1, index2] = farthest_points(hull);
    corner_indices.push_back(index1);
    corner_indices.push_back(
        point_making_largest_triangle(hull, index1, index2));
    corner_indices.push_back(index2);
    corner_indices.push_back(
        point_making_largest_triangle(hull, index2, index1));
  } else {
    int n = hull.size();

    // Find 4 corners with maximum curvature
    std::vector<std::pair<float, int>> curvatures;

    for (int i = 0; i < n; i++) {
      Point2D prev = hull[(i - 1 + n) % n];
      Point2D curr = hull[i];
      Point2D next = hull[(i + 1) % n];

      Point2D v1 = (curr - prev).normalized();
      Point2D v2 = (next - curr).normalized();

      float angle = std::atan2(std::abs(v1.cross(v2)), v1.dot(v2));
      curvatures.push_back({angle, i});
    }

    std::sort(curvatures.begin(), curvatures.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });

    for (int i = 0; i < std::min(4, (int)curvatures.size()); i++) {
      corner_indices.push_back(curvatures[i].second);
    }

    std::sort(corner_indices.begin(), corner_indices.end());

    if (corner_indices.size() < 4) {
      // Fallback to extreme points
      Point2D centroid(0, 0);
      for (const auto &p : hull) {
        centroid.x += p.x;
        centroid.y += p.y;
      }
      centroid.x /= n;
      centroid.y /= n;

      float best_score[4] = {-1e9f, -1e9f, -1e9f, -1e9f};
      int best_idx[4] = {0, 0, 0, 0};
      Point2D dirs[4] = {{1, -1}, {1, 1}, {-1, 1}, {-1, -1}};

      for (int i = 0; i < n; i++) {
        Point2D delta = hull[i] - centroid;
        for (int d = 0; d < 4; d++) {
          float score = delta.dot(dirs[d]);
          if (score > best_score[d]) {
            best_score[d] = score;
            best_idx[d] = i;
          }
        }
      }

      corner_indices.clear();
      for (int d = 0; d < 4; d++)
        corner_indices.push_back(best_idx[d]);
      std::sort(corner_indices.begin(), corner_indices.end());
    }
  }

  for (int i = 0; i < 4; i++) {
    quad.corners[i] = hull[corner_indices[i]];
  }

  return quad;
}

// =============================================================================
// Find Quads from Binary Image (like original AprilTag)
// =============================================================================

// Check if pixel is on boundary of a black region
inline bool is_boundary_pixel(const Buffer<uint8_t> &binary, int x, int y) {
  if (binary(x, y) == 0)
    return false; // Not black

  // Boundary if any 4-connected neighbor is white (0)
  int w = binary.width(), h = binary.height();
  if (x > 0 && binary(x - 1, y) == 0)
    return true;
  if (x < w - 1 && binary(x + 1, y) == 0)
    return true;
  if (y > 0 && binary(x, y - 1) == 0)
    return true;
  if (y < h - 1 && binary(x, y + 1) == 0)
    return true;
  return false;
}

std::vector<Quad> find_quads_from_binary(const Buffer<uint8_t> &binary,
                                         int min_area, int max_area) {
  int width = binary.width();
  int height = binary.height();

  // Find connected components of BLACK pixels using Union-Find
  UnionFind uf(width * height);

  // 4-connectivity for black pixels
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (binary(x, y) == 0)
        continue; // Skip white

      int idx = y * width + x;

      if (x + 1 < width && binary(x + 1, y) > 0) {
        uf.unite(idx, idx + 1);
      }
      if (y + 1 < height && binary(x, y + 1) > 0) {
        uf.unite(idx, idx + width);
      }
    }
  }

  // Gather BOUNDARY points for each component
  std::unordered_map<int, std::vector<Point2D>> component_boundary;
  std::unordered_map<int, int> component_area;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (binary(x, y) == 0)
        continue;

      int idx = y * width + x;
      int root = uf.find(idx);

      component_area[root]++;

      // Only add boundary pixels
      if (is_boundary_pixel(binary, x, y)) {
        component_boundary[root].push_back({(float)x, (float)y});
      }
    }
  }

  std::vector<Quad> quads;

  for (auto &[root, boundary_points] : component_boundary) {
    int area = component_area[root];

    // Filter by area
    if (area < min_area || area > max_area)
      continue;

    // Need enough boundary points
    if (boundary_points.size() < 20)
      continue;

    // Compute bounding box
    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (const auto &p : boundary_points) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }

    float box_width = max_x - min_x;
    float box_height = max_y - min_y;

    // Must be reasonably sized
    if (box_width < 15 || box_height < 15)
      continue;
    if (box_width > width * 0.8f || box_height > height * 0.8f)
      continue;

    // Aspect ratio check (AprilTags are roughly square)
    float aspect = std::max(box_width, box_height) /
                   std::max(1.0f, std::min(box_width, box_height));
    if (aspect > 3.0f)
      continue;

    // Check "rectangularity" - boundary should have points distributed around
    // all 4 sides
    bool has_top = false, has_bottom = false, has_left = false,
         has_right = false;
    float margin = 0.25f;

    for (const auto &p : boundary_points) {
      float rel_x = (p.x - min_x) / box_width;
      float rel_y = (p.y - min_y) / box_height;

      if (rel_y < margin)
        has_top = true;
      if (rel_y > 1.0f - margin)
        has_bottom = true;
      if (rel_x < margin)
        has_left = true;
      if (rel_x > 1.0f - margin)
        has_right = true;
    }

    if (!has_top || !has_bottom || !has_left || !has_right)
      continue;

    // Compute convex hull of boundary
    std::vector<Point2D> hull = convex_hull(boundary_points);
    if (hull.size() < 4)
      continue;

    // Fit quad
    Quad quad = fit_quad_to_hull(hull, (float)area);

    if (quad.is_valid()) {
      quads.push_back(quad);
    }
  }

  // Sort by area (larger is better confidence for AprilTags)
  std::sort(quads.begin(), quads.end(), [](const Quad &a, const Quad &b) {
    return a.confidence > b.confidence;
  });

  // Non-maximum suppression
  std::vector<bool> suppressed(quads.size(), false);
  std::vector<Quad> result;

  for (size_t i = 0; i < quads.size(); i++) {
    if (suppressed[i])
      continue;
    result.push_back(quads[i]);

    Point2D ci = (quads[i].corners[0] + quads[i].corners[2]) * 0.5f;
    float ri = (quads[i].corners[2] - quads[i].corners[0]).norm() / 2;

    for (size_t j = i + 1; j < quads.size(); j++) {
      if (suppressed[j])
        continue;
      Point2D cj = (quads[j].corners[0] + quads[j].corners[2]) * 0.5f;
      if ((cj - ci).norm() < ri * 0.5f) {
        suppressed[j] = true;
      }
    }
  }

  return result;
}

// =============================================================================
// FAST Quad Detection (Decimation + Raw Pointers + Flattened Union-Find)
// =============================================================================

struct FastUF {
  int *parent;
  int n;

  FastUF(int size) : n(size) {
    parent = new int[size];
    for (int i = 0; i < size; i++)
      parent[i] = i;
  }
  ~FastUF() { delete[] parent; }

  inline int find(int x) {
    int root = x;
    while (root != parent[root])
      root = parent[root];
    while (x != root) {
      int next = parent[x];
      parent[x] = root;
      x = next;
    }
    return root;
  }

  inline void unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx != ry)
      parent[ry] = rx;
  }

  void flatten() {
    for (int i = 0; i < n; i++) {
      parent[i] = find(i);
    }
  }
};

struct FindQuadsResult {
  std::vector<int> area_counts;
  std::vector<int> active_roots;
  std::vector<std::vector<Point2D>> clusters;
  std::vector<std::vector<Point2D>> hulls;
  std::vector<Quad> quads;
  FindQuadsResult(std::vector<int> area_counts, std::vector<int> active_roots,
                  std::vector<std::vector<Point2D>> clusters,
                  std::vector<std::vector<Point2D>> hulls,
                  std::vector<Quad> quads)
      : area_counts(area_counts), active_roots(active_roots),
        clusters(clusters), hulls(hulls), quads(quads) {}
  FindQuadsResult()
      : area_counts(std::vector<int>()), active_roots(std::vector<int>()),
        clusters(std::vector<std::vector<Point2D>>()),
        hulls(std::vector<std::vector<Point2D>>()), quads(std::vector<Quad>()) {
  }
};

FindQuadsResult
find_quads_fast(std::vector<std::pair<std::string, double>> &timings,
                const Buffer<uint8_t> &binary, int min_area, int max_area,
                int decimation = 1, int num_threads = 4) {
  using Clock = std::chrono::steady_clock;
  auto to_ms = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto func_start = Clock::now();
  auto stage_start = Clock::now();
  const int w = binary.width();
  const int h = binary.height();
  const int s_w = w / decimation;
  const int s_h = h / decimation;

  const uint8_t *__restrict__ ptr = binary.data();
  const int stride = binary.stride(1);
  auto stage_end = Clock::now();
  FindQuadsResult result(std::vector<int>(s_w * s_h, 0), std::vector<int>(),
                         std::vector<std::vector<Point2D>>(s_w * s_h),
                         std::vector<std::vector<Point2D>>(),
                         std::vector<Quad>());
  result.active_roots.reserve(256);
  timings.emplace_back("find_quads_fast_setup", to_ms(stage_end - stage_start));

  // Pass 1: Union-Find on decimated grid
  stage_start = Clock::now();
  FastUF uf(s_w * s_h);

  for (int sy = 0; sy < s_h; sy++) {
    const int y = sy * decimation;
    const uint8_t *row = ptr + y * stride;

    for (int sx = 0; sx < s_w; sx++) {
      const int x = sx * decimation;
      if (row[x] == 127 || row[x] == 255)
        continue; // Not black

      const int idx = sy * s_w + sx;

      // Connect right
      if (sx + 1 < s_w && row[x + decimation] == 0) {
        uf.unite(idx, idx + 1);
      }
      // Connect down
      if (sy + 1 < s_h && ptr[(y + decimation) * stride + x] == 0) {
        uf.unite(idx, idx + s_w);
      }
    }
  }
  stage_end = Clock::now();
  timings.emplace_back("find_quads_fast_uf_pass_1",
                       to_ms(stage_end - stage_start));

  // KEY OPTIMIZATION: Flatten all parents for O(1) lookup in pass 2
  stage_start = Clock::now();
  uf.flatten();
  stage_end = Clock::now();
  timings.emplace_back("find_quads_fast_uf_flatten",
                       to_ms(stage_end - stage_start));

  stage_start = Clock::now();
  // Pass 2: Collect boundary points + count area
  // Use vectors indexed by root (sparse, but fast)
  for (int sy = 1; sy < s_h - 1; sy++) {
    const int y = sy * decimation;
    const uint8_t *row = ptr + y * stride;
    const uint8_t *row_up = ptr + (y - decimation) * stride;
    const uint8_t *row_down = ptr + (y + decimation) * stride;

    for (int sx = 1; sx < s_w - 1; sx++) {
      const int x = sx * decimation;
      if (row[x] != 0)
        continue;

      const int root = uf.parent[sy * s_w + sx]; // O(1) - already flattened!

      if (result.area_counts[root] == 0) {
        result.active_roots.push_back(root);
      }
      result.area_counts[root]++;

      // Boundary check
      if ((row[x - decimation] == 255) | (row[x + decimation] == 255) |
          (row_up[x] == 255) | (row_down[x] == 255)) {
        result.clusters[root].emplace_back((float)x, (float)y);
      }
    }
  }
  stage_end = Clock::now();
  timings.emplace_back("find_quads_fast_cluster",
                       to_ms(stage_end - stage_start));

  // Pass 3: Fit quads
  stage_start = Clock::now();
  const int scaled_min_area = min_area / (decimation * decimation);
  const int scaled_max_area = max_area / (decimation * decimation);

  result.quads.reserve(result.active_roots.size());
  result.hulls.reserve(result.active_roots.size());

  for (int root : result.active_roots) {
    const int area = result.area_counts[root];
    if (area < scaled_min_area || area > scaled_max_area)
      continue;

    std::vector<Point2D> &boundary = result.clusters[root];
    if (boundary.size() < 10)
      continue;

    // Bounding box check
    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (const auto &p : boundary) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
    }
    if ((max_x - min_x) < 6 || (max_y - min_y) < 6)
      continue;

    std::vector<Point2D> hull = quickhull(boundary);
    if (hull.size() < 4)
      continue;

    Quad quad = fit_quad_to_hull(hull, (float)area * decimation * decimation);

    result.hulls.push_back(hull);
    // if (quad.is_valid()) {
    result.quads.push_back(quad);
    // }
  }
  stage_end = Clock::now();
  timings.emplace_back("find_quads_fast_hull_and_fit_quads",
                       to_ms(stage_end - stage_start));
  timings.emplace_back("find_quads_fast_total", to_ms(stage_end - func_start));

  // std::sort(quads.begin(), quads.end(), [](const Quad& a, const Quad& b) {
  //     return a.confidence > b.confidence;
  // });

  // return quads;

  return result;
}

// =============================================================================
// Visualization
// =============================================================================

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
        float r = input(x, y, 0);
        float g = input(x, y, 1);
        float b = input(x, y, 2);
        gray(x, y) = 0.299f * r + 0.587f * g + 0.114f * b;
      }
    }
  }
  return gray;
}

void draw_line(Buffer<uint8_t> &buf, Point2D p1, Point2D p2, uint8_t r,
               uint8_t g, uint8_t b) {
  int steps = std::max(1, (int)(std::abs(p2.x - p1.x) + std::abs(p2.y - p1.y)));
  for (int i = 0; i <= steps; ++i) {
    float t = (float)i / steps;
    int x = (int)(p1.x + t * (p2.x - p1.x));
    int y = (int)(p1.y + t * (p2.y - p1.y));
    if (x >= 0 && x < buf.width() && y >= 0 && y < buf.height()) {
      buf(x, y, 0) = r;
      buf(x, y, 1) = g;
      buf(x, y, 2) = b;
    }
  }
}

Buffer<uint8_t>
visualize_hulls(const Buffer<uint8_t> &input,
                const std::vector<std::vector<Point2D>> &hulls) {
  Buffer<uint8_t> vis(input.width(), input.height(), 3);

  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      uint8_t v = (input.channels() == 1) ? input(x, y)
                                          : (uint8_t)(0.299f * input(x, y, 0) +
                                                      0.587f * input(x, y, 1) +
                                                      0.114f * input(x, y, 2));
      vis(x, y, 0) = vis(x, y, 1) = vis(x, y, 2) = v;
    }
  }

  for (const auto &hull : hulls) {
    if (hull.empty())
      continue;

    for (int i = 0; i < hull.size(); ++i) {
      draw_line(vis, hull[i], hull[(i + 1) % hull.size()], 255, 0, 0);
    }

    for (int i = 0; i < hull.size(); ++i) {
      int cx = (int)hull[i].x;
      int cy = (int)hull[i].y;
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          if (dx * dx + dy * dy <= 9) {
            int px = cx + dx, py = cy + dy;
            if (px >= 0 && px < vis.width() && py >= 0 && py < vis.height()) {
              vis(px, py, 0) = 0;
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

Buffer<uint8_t>
visualize_clusters(const Buffer<uint8_t> &input,
                   const std::vector<std::vector<Point2D>> &clusters) {
  Buffer<uint8_t> vis(input.width(), input.height(), 3);

  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      uint8_t v = (input.channels() == 1) ? input(x, y)
                                          : (uint8_t)(0.299f * input(x, y, 0) +
                                                      0.587f * input(x, y, 1) +
                                                      0.114f * input(x, y, 2));
      vis(x, y, 0) = vis(x, y, 1) = vis(x, y, 2) = v;
    }
  }

  for (const auto &cluster : clusters) {
    for (int i = 0; i < cluster.size(); ++i) {
      int cx = (int)cluster[i].x;
      int cy = (int)cluster[i].y;
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          if (dx * dx + dy * dy <= 9) {
            int px = cx + dx, py = cy + dy;
            if (px >= 0 && px < vis.width() && py >= 0 && py < vis.height()) {
              vis(px, py, 0) = 0;
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

Buffer<uint8_t> visualize_quads(const Buffer<uint8_t> &input,
                                const std::vector<Quad> &quads,
                                bool valid_only = true) {
  Buffer<uint8_t> vis(input.width(), input.height(), 3);

  for (int y = 0; y < input.height(); ++y) {
    for (int x = 0; x < input.width(); ++x) {
      uint8_t v = (input.channels() == 1) ? input(x, y)
                                          : (uint8_t)(0.299f * input(x, y, 0) +
                                                      0.587f * input(x, y, 1) +
                                                      0.114f * input(x, y, 2));
      vis(x, y, 0) = vis(x, y, 1) = vis(x, y, 2) = v;
    }
  }

  for (const auto &q : quads) {
    if (valid_only ^ q.is_valid())
      continue;
    for (int i = 0; i < 4; ++i) {
      draw_line(vis, q.corners[i], q.corners[(i + 1) % 4], 255, 0, 0);
    }

    for (int i = 0; i < 4; ++i) {
      int cx = (int)q.corners[i].x;
      int cy = (int)q.corners[i].y;
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
          if (dx * dx + dy * dy <= 9) {
            int px = cx + dx, py = cy + dy;
            if (px >= 0 && px < vis.width() && py >= 0 && py < vis.height()) {
              vis(px, py, 0) = 0;
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

Buffer<uint8_t> visualize_edges(const Buffer<uint8_t> &edges) {
  Buffer<uint8_t> vis(edges.width(), edges.height(), 3);

  for (int y = 0; y < edges.height(); ++y) {
    for (int x = 0; x < edges.width(); ++x) {
      uint8_t v = edges(x, y);
      vis(x, y, 0) = v;
      vis(x, y, 1) = v;
      vis(x, y, 2) = v;
    }
  }

  return vis;
}

// =============================================================================
// Main
// =============================================================================

std::tuple<std::vector<Quad>, std::vector<std::vector<Point2D>>,
           std::vector<std::vector<Point2D>>>
run_pipeline(Buffer<uint8_t> &input, Buffer<uint8_t> &binary,
             std::vector<std::pair<std::string, double>> &timings) {
  using Clock = std::chrono::steady_clock;
  auto to_ms = [](Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };

  const auto program_start = Clock::now();

  // =================================================================
  // Stage 1: GPU - Grayscale + Adaptive Threshold (binary image)
  // Grayscale conversion is now fused into the Halide pipeline
  // =================================================================

  // Warm call - shaders cached, context ready
  auto stage_start = Clock::now();
  int result = greyscale_and_adaptive_threshold(input, 4, 60.0f, binary);
  auto stage_end = Clock::now();
  double warm_time = to_ms(stage_end - stage_start);
  timings.emplace_back("GPU warm (cached)", warm_time);

  if (result != 0) {
    throw std::runtime_error("Halide pipeline failed: " +
                             std::to_string(result));
  }

  // Copy to host
  stage_start = Clock::now();
  binary.copy_to_host();
  stage_end = Clock::now();
  timings.emplace_back("copy_to_host", to_ms(stage_end - stage_start));

  // =================================================================
  // Stage 2: CPU - Connected Components + Quad Fitting
  // =================================================================
  // Min/max area based on expected tag sizes (adaptive to image size)
  int img_area = binary.width() * binary.height();
  int min_area = img_area / 60000; // Tags should be at least 0.05% of image
  int max_area = img_area / 8;     // Tags should be at most 25% of image

  stage_start = Clock::now();
  auto retval = find_quads_fast(timings, binary, min_area, max_area, 1, 4);
  stage_end = Clock::now();
  timings.emplace_back("quad_detect (CPU)", to_ms(stage_end - stage_start));

  return retval;
}

int main(int argc, char **argv) {
  try {
    using Clock = std::chrono::steady_clock;
    auto to_ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };

    std::vector<std::pair<std::string, double>> timings;
    const auto program_start = Clock::now();

    const char *input_path = (argc > 1) ? argv[1] : "../apriltags.png";

    std::cout << "=== AprilTag Adaptive Threshold Pipeline ===" << std::endl;
    std::cout << "Loading: " << input_path << std::endl;
    std::cout << "(No manual tuning required - adapts to image automatically)"
              << std::endl;

    auto stage_start = Clock::now();
    Buffer<uint8_t> input = load_image(input_path);
    auto stage_end = Clock::now();
    timings.emplace_back("load_image", to_ms(stage_end - stage_start));
    std::cout << "Image dimensions: " << input.width() << "x" << input.height()
              << "x" << input.channels() << std::endl;

    Buffer<uint8_t> binary(input.width(), input.height());
    for (int i = 0; i < 10; i++) {
      run_pipeline(input, binary, timings);
    }

    timings.clear();

    FindQuadsResult quads;
    for (int i = 0; i < 100; i++) {
      quads = run_pipeline(input, binary, timings);
    }

    std::cout << "Found " << quads.quads.size() << " quads" << std::endl;

    // =================================================================
    // Output
    // =================================================================
    for (size_t i = 0; i < quads.quads.size(); ++i) {
      const auto &q = quads.quads[i];
      if (!q.is_valid())
        continue;
      std::cout << "Quad " << i << ": [";
      for (int j = 0; j < 4; ++j) {
        std::cout << "(" << (int)q.corners[j].x << ", " << (int)q.corners[j].y
                  << ")";
        if (j < 3)
          std::cout << ", ";
      }
      std::cout << "] perimeter=" << (int)q.perimeter()
                << " area=" << (int)q.area()
                << " confidence=" << (int)q.confidence
                << " parallelness=" << q.parallelness() << std::endl;
    }

    for (size_t i = 0; i < quads.quads.size(); ++i) {
      const auto &q = quads.quads[i];
      if (q.is_valid())
        continue;
      if (q.parallelness() > 0.7f)
        continue;
      std::cout << "Quad " << i << ": [";
      for (int j = 0; j < 4; ++j) {
        std::cout << "(" << (int)q.corners[j].x << ", " << (int)q.corners[j].y
                  << ")";
        if (j < 3)
          std::cout << ", ";
      }
      std::cout << "] perimeter=" << (int)q.perimeter()
                << " area=" << (int)q.area()
                << " confidence=" << (int)q.confidence
                << " parallelness=" << q.parallelness() << std::endl;
    }

    // Save visualizations
    stage_start = Clock::now();
    Buffer<uint8_t> binary_vis = visualize_edges(binary);
    save_image(binary_vis, "binary_output.png");
    stage_end = Clock::now();
    timings.emplace_back("save_binary_output", to_ms(stage_end - stage_start));
    std::cout << "Saved: binary_output.png" << std::endl;

    // Visualize clusters
    stage_start = Clock::now();
    Buffer<uint8_t> clusters_vis = visualize_clusters(input, quads.clusters);
    save_image(clusters_vis, "clusters_output.png");
    stage_end = Clock::now();
    timings.emplace_back("save_clusters_output",
                         to_ms(stage_end - stage_start));
    std::cout << "Saved: clusters_output.png" << std::endl;

    // Visualize hulls
    stage_start = Clock::now();
    Buffer<uint8_t> hulls_vis = visualize_hulls(input, quads.hulls);
    save_image(hulls_vis, "hulls_output.png");
    stage_end = Clock::now();
    timings.emplace_back("save_hulls_output", to_ms(stage_end - stage_start));
    std::cout << "Saved: hulls_output.png" << std::endl;

    // Visualize quads
    stage_start = Clock::now();
    Buffer<uint8_t> quads_vis = visualize_quads(input, quads.quads);
    save_image(quads_vis, "quads_output.png");
    stage_end = Clock::now();
    timings.emplace_back("save_quads_output", to_ms(stage_end - stage_start));
    std::cout << "Saved: quads_output.png" << std::endl;

    // Visualize quads
    stage_start = Clock::now();
    Buffer<uint8_t> invalid_quads_vis =
        visualize_quads(input, quads.quads, false);
    save_image(invalid_quads_vis, "quads_output_invalid.png");
    stage_end = Clock::now();
    timings.emplace_back("save_quads_output_invalid",
                         to_ms(stage_end - stage_start));
    std::cout << "Saved: quads_output_invalid.png" << std::endl;

    // for (size_t i = 0; i < quads.size(); ++i) {
    //     const auto& q = quads[i];
    //     if (!q.is_valid()) continue;
    //     stage_start = Clock::now();
    //     Buffer<uint8_t> quads_vis = visualize_quads(input, {q});
    //     save_image(quads_vis, "quads_output_" + std::to_string(i) + ".png");
    //     stage_end = Clock::now();
    //     timings.emplace_back("save_quads_output_" + std::to_string(i),
    //     to_ms(stage_end - stage_start)); std::cout << "Saved: quads_output_"
    //     << i << ".png" << std::endl;
    // }

    // Cleanup
    binary.device_free();

    // Calculate sum, mean, and stdev for 'convert_to_grayscale',
    // 'atag_edge_detect (GPU)', 'quad_detect (CPU)'
    auto compute_sum_mean_stdev =
        [](const std::vector<std::pair<std::string, double>> &timings,
           const std::string &key) {
          std::vector<double> values;
          for (const auto &entry : timings) {
            if (entry.first == key) {
              values.push_back(entry.second);
            }
          }
          double sum = 0.0;
          for (double v : values)
            sum += v;
          double mean = (values.empty()) ? 0.0 : sum / values.size();
          double stdev = 0.0;
          if (!values.empty()) {
            for (double v : values)
              stdev += (v - mean) * (v - mean);
            stdev = std::sqrt(stdev / values.size());
          }
          return std::make_tuple(sum, mean, stdev);
        };

    auto [grayscale_sum, grayscale_mean, grayscale_stdev] =
        compute_sum_mean_stdev(timings, "GPU warm (cached)");
    auto [edge_sum, edge_mean, edge_stdev] =
        compute_sum_mean_stdev(timings, "copy_to_host");
    auto [quad_sum, quad_mean, quad_stdev] =
        compute_sum_mean_stdev(timings, "quad_detect (CPU)");
    auto [setup_sum, setup_mean, setup_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_setup");
    auto [uf_pass_1_sum, uf_pass_1_mean, uf_pass_1_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_uf_pass_1");
    auto [uf_flatten_sum, uf_flatten_mean, uf_flatten_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_uf_flatten");
    auto [cluster_sum, cluster_mean, cluster_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_cluster");
    auto [hull_and_fit_quads_sum, hull_and_fit_quads_mean,
          hull_and_fit_quads_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_hull_and_fit_quads");
    auto [total_sum, total_mean, total_stdev] =
        compute_sum_mean_stdev(timings, "find_quads_fast_total");

    std::cout << "GPU warm (cached): sum=" << grayscale_sum
              << ", mean=" << grayscale_mean << ", stdev=" << grayscale_stdev
              << std::endl;
    std::cout << "Copy to Host:      sum=" << edge_sum << ", mean=" << edge_mean
              << ", stdev=" << edge_stdev << std::endl;
    std::cout << "Quad:              sum=" << quad_sum << ", mean=" << quad_mean
              << ", stdev=" << quad_stdev << std::endl;
    std::cout << "    Setup:             sum=" << setup_sum
              << ", mean=" << setup_mean << ", stdev=" << setup_stdev
              << std::endl;
    std::cout << "    UF Pass 1:        sum=" << uf_pass_1_sum
              << ", mean=" << uf_pass_1_mean << ", stdev=" << uf_pass_1_stdev
              << std::endl;
    std::cout << "    UF Flatten:       sum=" << uf_flatten_sum
              << ", mean=" << uf_flatten_mean << ", stdev=" << uf_flatten_stdev
              << std::endl;
    std::cout << "    Cluster:          sum=" << cluster_sum
              << ", mean=" << cluster_mean << ", stdev=" << cluster_stdev
              << std::endl;
    std::cout << "    Hull and Fit Quads: sum=" << hull_and_fit_quads_sum
              << ", mean=" << hull_and_fit_quads_mean
              << ", stdev=" << hull_and_fit_quads_stdev << std::endl;
    std::cout << "    Total:             sum=" << total_sum
              << ", mean=" << total_mean << ", stdev=" << total_stdev
              << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}