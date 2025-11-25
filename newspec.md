### Architecture Spec: Halide-Accelerated AprilTag 2 (Tile-Aggregation)

**Goal:** Detect quad tags without pixel-level connected components.
**Strategy:** "Coarse-to-Fine" reduction. GPU compresses pixels into "Edge Tiles"; CPU connects tiles.

---

### 1. Data Structures (Interface)

**Input:**
* `Image`: $W \times H$ Grayscale (uint8).

**Intermediate (GPU -> CPU):**
* `TileMap`: Struct of Arrays (SoA), dimensions $(W/8) \times (H/8)$.
    * `mag_sum` (float): Sum of gradient magnitudes (Confidence).
    * `vec_x` (float): $\sum (dx \cdot |mag|)$ (Dominant X direction).
    * `vec_y` (float): $\sum (dy \cdot |mag|)$ (Dominant Y direction).
    * `mom_x` (float): $\sum (x \cdot |mag|) / \sum |mag|$ (Local center of mass X).
    * `mom_y` (float): $\sum (y \cdot |mag|) / \sum |mag|$ (Local center of mass Y).

---

### 2. Pipeline Stage 1: The "Cruncher" (Halide/GPU)

**Kernel Logic:**
1.  **Clamp:** `BoundaryConditions::repeat_edge`.
2.  **Blur:** $3 \times 3$ Gaussian (separable) to reduce sensor noise.
3.  **Gradients:**
    * $dx = I(x+1, y) - I(x-1, y)$
    * $dy = I(x, y+1) - I(x, y-1)$
    * $mag = \sqrt{dx^2 + dy^2}$
4.  **Decimation (Reduction):**
    * Input: $8 \times 8$ pixel block.
    * Operation: Parallel reduction (Summation).
    * Filter: If $mag < \text{GlobalThreshold}$, contribution is 0.

**Halide Schedule Strategy:**
* `compute_root()` on the final TileMap.
* `gpu_tile()`: Map output tiles $(x, y)$ to GPU threads.
* `vectorize()`: Inner loop of the reduction.
* **Optimization:** Unroll the $8 \times 8$ loops; keep intermediate sums in registers.

---

### 3. Pipeline Stage 2: The "Linker" (CPU)

**Algorithm: Sparse Union-Find (DSU)**
The data volume is now $1/64$th of the original.

1.  **Iterate** over `TileMap`.
2.  **Discard** tiles where `mag_sum` < `TileThreshold`.
3.  **Connect** Tile $A$ and Tile $B$ (Neighbors 4-way or 8-way) if:
    * $\text{DotProduct}(Vec_A, Vec_B) > \cos(30^\circ)$ (Orientation alignment).
4.  **Output:** A list of `Component` objects. Each component contains a list of $(x, y)$ coordinates (derived from `mom_x/y` + tile offset).

---

### 4. Pipeline Stage 3: The "Solver" (CPU)

1.  **Fit Quads:**
    * For each `Component`, use Least Squares/RANSAC to fit 4 line segments.
    * Compute 4 intersection points (Corners).
2.  **Homography:**
    * Compute $H$ matrix mapping Tag coordinates $(0,0) \dots (1,1)$ to Image coordinates.

---

### 5. Pipeline Stage 4: The "Sampler" (Halide/GPU - Optional)

If high throughput is required, offload sampling back to GPU.

**Input:** Original Image, List of Homographies ($H_1 \dots H_n$).
**Kernel Logic:**
1.  **Grid Gen:** Generate $u, v$ coordinates for bit centers (e.g., $6 \times 6$ grid).
2.  **Transform:** $x, y = H \cdot (u, v)$.
3.  **Sample:** Bilinear interpolate $I(x, y)$.
4.  **Decode:** Compare sample vs. local min/max to determine Bit 0/1.

**Output:** `TagID` and `HammingDistance`.

---

### Implementation Notes

* **Memory:** The `TileMap` output is small. For 1080p, it is approx $240 \times 135$ elements. Transferring this to CPU is negligible (~500KB).
* **Edge Case:** The "Gradient Vector Sum" in Stage 1 might vanish if a tile contains a sharp corner (vectors cancel out).
    * *Fix:* Also track `max_mag` in the reduction. If `mag_sum` is low but `max_mag` is high, pass the tile to CPU for "corner logic" or simply treat it as a valid node with undefined orientation.
* **Precision:** Use `float16` (half) on NPU/GPU for the `TileMap` to halve bandwidth. Precision loss is acceptable for detection.