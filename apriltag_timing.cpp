
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <numeric>

extern "C" {
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/tag25h9.h"
#include "apriltag/tag16h5.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/pjpeg.h"
#include "apriltag/common/pnm.h"
#include "apriltag/common/zarray.h"
#include "apriltag/common/string_util.h"
#include <zlib.h>
}

#ifdef APRILTAG_HAVE_HALIDE
extern "C" image_u8_t *halide_threshold(apriltag_detector_t *td, image_u8_t *im);

static void warm_halide_threshold_pipeline()
{
    static bool warmed = false;
    if (warmed) {
        return;
    }

    apriltag_detector_t *td = apriltag_detector_create();
    if (!td) {
        return;
    }
    td->use_halide = true;

    image_u8_t *dummy = image_u8_create(128, 128);
    if (dummy) {
        image_u8_t *result = halide_threshold(td, dummy);
        if (result) {
            image_u8_destroy(result);
        }
        image_u8_destroy(dummy);
    }

    apriltag_detector_destroy(td);
    warmed = true;
}
#endif

image_u8_t* image_u8_create_from_png(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open PNG file: %s\n", path);
        return NULL;
    }

    uint8_t png_sig[8];
    if (fread(png_sig, 1, 8, fp) != 8) {
        fprintf(stderr, "Error: PNG file too small\n");
        fclose(fp);
        return NULL;
    }

    static const uint8_t expected_sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (memcmp(png_sig, expected_sig, 8) != 0) {
        fprintf(stderr, "Error: Invalid PNG signature\n");
        fclose(fp);
        return NULL;
    }

    uint32_t width = 0, height = 0;
    uint8_t bit_depth = 0, color_type = 0;
    uint8_t *image_data = NULL;
    uint32_t image_data_size = 0;

    while (!feof(fp)) {
        uint8_t len_bytes[4];
        if (fread(len_bytes, 1, 4, fp) != 4) break;

        uint32_t chunk_len = ((uint32_t)len_bytes[0] << 24) |
                              ((uint32_t)len_bytes[1] << 16) |
                              ((uint32_t)len_bytes[2] << 8) |
                              ((uint32_t)len_bytes[3]);

        uint8_t chunk_type[4];
        if (fread(chunk_type, 1, 4, fp) != 4) break;

        if (strncmp((const char*)chunk_type, "IHDR", 4) == 0) {
            uint8_t ihdr[13];
            if (fread(ihdr, 1, 13, fp) != 13) break;

            width = ((uint32_t)ihdr[0] << 24) | ((uint32_t)ihdr[1] << 16) |
                    ((uint32_t)ihdr[2] << 8) | ((uint32_t)ihdr[3]);
            height = ((uint32_t)ihdr[4] << 24) | ((uint32_t)ihdr[5] << 16) |
                     ((uint32_t)ihdr[6] << 8) | ((uint32_t)ihdr[7]);
            bit_depth = ihdr[8];
            color_type = ihdr[9];

            uint32_t crc;
            fread(&crc, 1, 4, fp);
        } else if (strncmp((const char*)chunk_type, "IDAT", 4) == 0) {
            uint8_t *new_data = (uint8_t*)realloc(image_data, image_data_size + chunk_len);
            if (new_data == NULL) {
                fprintf(stderr, "Error: Out of memory for PNG data\n");
                free(image_data);
                fclose(fp);
                return NULL;
            }
            image_data = new_data;

            if (fread(image_data + image_data_size, 1, chunk_len, fp) != chunk_len) {
                fprintf(stderr, "Error: Failed to read PNG IDAT chunk\n");
                free(image_data);
                fclose(fp);
                return NULL;
            }
            image_data_size += chunk_len;

            uint32_t crc;
            fread(&crc, 1, 4, fp);
        } else if (strncmp((const char*)chunk_type, "IEND", 4) == 0) {
            uint32_t crc;
            fread(&crc, 1, 4, fp);
            break;
        } else {
            uint8_t *chunk_data = (uint8_t*)malloc(chunk_len + 4);
            if (chunk_data) {
                fread(chunk_data, 1, chunk_len + 4, fp);
                free(chunk_data);
            }
        }
    }

    fclose(fp);

    if (image_data == NULL || width == 0 || height == 0) {
        fprintf(stderr, "Error: Invalid PNG file structure\n");
        free(image_data);
        return NULL;
    }

    unsigned long decomp_size = height * (width + 1) * 4;
    uint8_t *decomp_data = (uint8_t*)malloc(decomp_size);
    if (decomp_data == NULL) {
        fprintf(stderr, "Error: Out of memory for decompressed PNG data\n");
        free(image_data);
        return NULL;
    }

    int ret = uncompress(decomp_data, &decomp_size, image_data, image_data_size);
    free(image_data);

    if (ret != Z_OK) {
        fprintf(stderr, "Error: Failed to decompress PNG data (zlib error: %d)\n", ret);
        free(decomp_data);
        return NULL;
    }

    image_u8_t *im = image_u8_create(width, height);
    if (im == NULL) {
        fprintf(stderr, "Error: Failed to allocate image\n");
        free(decomp_data);
        return NULL;
    }

    uint32_t offset = 0;
    for (uint32_t y = 0; y < height; y++) {
        uint8_t filter_type = decomp_data[offset++];
        (void)filter_type;

        for (uint32_t x = 0; x < width; x++) {
            uint8_t pixel = 0;

            if (color_type == 0) {
                if (bit_depth == 8) {
                    pixel = decomp_data[offset++];
                }
            } else if (color_type == 2) {
                uint8_t r = decomp_data[offset++];
                uint8_t g = decomp_data[offset++];
                uint8_t b = decomp_data[offset++];
                pixel = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            } else if (color_type == 3) {
                pixel = decomp_data[offset++];
            } else if (color_type == 4) {
                pixel = decomp_data[offset++];
                offset++;
            } else if (color_type == 6) {
                uint8_t r = decomp_data[offset++];
                uint8_t g = decomp_data[offset++];
                uint8_t b = decomp_data[offset++];
                offset++;
                pixel = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
            }

            im->buf[y * im->stride + x] = pixel;
        }
    }

    free(decomp_data);
    return im;
}

struct DetectionSummary {
    int id = 0;
    int hamming = 0;
    double decision_margin = 0.0;
    double center_x = 0.0;
    double center_y = 0.0;
    std::array<std::array<double, 2>, 4> corners{};
};

struct DetectorRun {
    bool use_halide = false;
    double detect_time_ms = 0.0;
    uint32_t nquads = 0;
    uint32_t nsegments = 0;
    uint32_t nedges = 0;
    float quad_decimate = 1.0f;
    float quad_sigma = 0.0f;
    int threads = 1;
    std::vector<DetectionSummary> detections;
};

struct TimingStats {
    double mean = 0.0;
    double stddev = 0.0;
};

static std::vector<DetectionSummary> snapshot_detections(zarray_t *detections)
{
    std::vector<DetectionSummary> out;
    const int n = zarray_size(detections);
    out.reserve(n);

    for (int i = 0; i < n; i++) {
        apriltag_detection_t *det = NULL;
        zarray_get(detections, i, &det);
        if (!det) {
            continue;
        }

        DetectionSummary summary;
        summary.id = det->id;
        summary.hamming = det->hamming;
        summary.decision_margin = det->decision_margin;
        summary.center_x = det->c[0];
        summary.center_y = det->c[1];
        for (int j = 0; j < 4; j++) {
            summary.corners[j][0] = det->p[j][0];
            summary.corners[j][1] = det->p[j][1];
        }
        out.push_back(summary);
    }

    return out;
}

static DetectorRun run_detector(apriltag_family_t *tf,
                                image_u8_t *im,
                                double decimate,
                                double blur,
                                int num_threads,
                                bool use_halide)
{
    DetectorRun run;
    run.use_halide = use_halide;

    apriltag_detector_t *td = apriltag_detector_create();
    if (!td) {
        fprintf(stderr, "Error: Failed to create AprilTag detector\n");
        return run;
    }

    apriltag_detector_add_family_bits(td, tf, 2);
    td->quad_decimate = decimate;
    td->quad_sigma = blur;
    td->nthreads = num_threads;
    td->refine_edges = true;
#ifdef APRILTAG_HAVE_HALIDE
    td->use_halide = use_halide;
#else
    (void)use_halide;
#endif

    auto detect_start = std::chrono::high_resolution_clock::now();
    zarray_t *detections = apriltag_detector_detect(td, im);
    auto detect_end = std::chrono::high_resolution_clock::now();
    run.detect_time_ms = std::chrono::duration<double, std::milli>(detect_end - detect_start).count();

    run.detections = snapshot_detections(detections);
    run.nquads = td->nquads;
    run.nsegments = td->nsegments;
    run.nedges = td->nedges;
    run.quad_decimate = td->quad_decimate;
    run.quad_sigma = td->quad_sigma;
    run.threads = td->nthreads;

    apriltag_detections_destroy(detections);
    apriltag_detector_destroy(td);

    return run;
}

static void print_run_summary(const char *label, const DetectorRun &run)
{
    printf("%s\n", label);
    printf("  Detection time: %.3f ms\n", run.detect_time_ms);
    printf("  Detections: %zu\n", run.detections.size());
    printf("  Quads considered: %u\n", run.nquads);
    printf("  Threads used: %d\n", run.threads);
    printf("  Decimate factor: %.1f\n", run.quad_decimate);
    printf("  Blur (sigma): %.2f\n", run.quad_sigma);

    for (size_t i = 0; i < run.detections.size(); i++) {
        const DetectionSummary &det = run.detections[i];
        printf("    Detection %zu:\n", i);
        printf("      Tag ID:        %d\n", det.id);
        printf("      Hamming dist:  %d\n", det.hamming);
        printf("      Decision mrg:  %.3f\n", det.decision_margin);
        printf("      Center (x,y):  (%.2f, %.2f)\n", det.center_x, det.center_y);
        printf("      Corners:\n");
        for (int j = 0; j < 4; j++) {
            printf("        [%d] (%.2f, %.2f)\n", j, det.corners[j][0], det.corners[j][1]);
        }
    }
}

static TimingStats compute_stats(const std::vector<double> &values)
{
    TimingStats stats;
    if (values.empty()) {
        return stats;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = sum / static_cast<double>(values.size());

    if (values.size() > 1) {
        double variance = 0.0;
        for (double v : values) {
            const double diff = v - stats.mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(values.size());
        stats.stddev = std::sqrt(variance);
    }

    return stats;
}

static std::unordered_map<int, std::vector<const DetectionSummary*>>
index_by_id(const std::vector<DetectionSummary> &detections)
{
    std::unordered_map<int, std::vector<const DetectionSummary*>> index;
    for (const DetectionSummary &det : detections) {
        index[det.id].push_back(&det);
    }
    return index;
}

// ai generated
// seems to work tho
static void report_diff(const DetectorRun &baseline, const DetectorRun &halide)
{
    constexpr double kCenterTolerance = 0.25;
    constexpr double kCornerTolerance = 0.50;
    constexpr double kMarginTolerance = 0.05;

    printf("\n[COMPARE] Detection count: baseline %zu | halide %zu\n",
           baseline.detections.size(), halide.detections.size());
    printf("[COMPARE] Detection time delta (Halide - Baseline): %.3f ms\n",
           halide.detect_time_ms - baseline.detect_time_ms);

    auto baseline_index = index_by_id(baseline.detections);
    auto halide_index = index_by_id(halide.detections);

    bool any_diff = false;

    for (const auto &entry : baseline_index) {
        int id = entry.first;
        const auto &baseline_group = entry.second;
        auto it = halide_index.find(id);

        if (it == halide_index.end()) {
            printf("[COMPARE] Tag %d missing from Halide run.\n", id);
            any_diff = true;
            continue;
        }

        const auto &halide_group = it->second;
        if (halide_group.size() != baseline_group.size()) {
            printf("[COMPARE] Tag %d count mismatch (baseline %zu vs halide %zu).\n",
                   id, baseline_group.size(), halide_group.size());
            any_diff = true;
        }

        const size_t count = std::min(baseline_group.size(), halide_group.size());
        for (size_t i = 0; i < count; i++) {
            const DetectionSummary *base_det = baseline_group[i];
            const DetectionSummary *halide_det = halide_group[i];

            double dx = base_det->center_x - halide_det->center_x;
            double dy = base_det->center_y - halide_det->center_y;
            double center_err = std::hypot(dx, dy);

            double max_corner_err = 0.0;
            for (int c = 0; c < 4; c++) {
                double cx = base_det->corners[c][0] - halide_det->corners[c][0];
                double cy = base_det->corners[c][1] - halide_det->corners[c][1];
                double dist = std::hypot(cx, cy);
                if (dist > max_corner_err) {
                    max_corner_err = dist;
                }
            }

            double margin_diff = std::abs(base_det->decision_margin - halide_det->decision_margin);
            bool hamming_diff = base_det->hamming != halide_det->hamming;

            if (center_err > kCenterTolerance ||
                max_corner_err > kCornerTolerance ||
                margin_diff > kMarginTolerance ||
                hamming_diff) {
                printf("[COMPARE] Tag %d differs (center err %.3f px, max corner err %.3f px, margin diff %.3f, hamming %d vs %d).\n",
                       id, center_err, max_corner_err, margin_diff,
                       base_det->hamming, halide_det->hamming);
                any_diff = true;
            }
        }
    }

    for (const auto &entry : halide_index) {
        int id = entry.first;
        if (baseline_index.find(id) == baseline_index.end()) {
            printf("[COMPARE] Tag %d detected only by Halide run.\n", id);
            any_diff = true;
        }
    }

    if (!any_diff) {
        printf("[COMPARE] Detected tags match within tolerances (center <= %.2f px, corners <= %.2f px).\n",
               kCenterTolerance, kCornerTolerance);
    }
}

void print_usage(const char *program_name) {
    printf("Usage: %s [options] <image_path>\n", program_name);
    printf("Options:\n");
    printf("  -f, --family <name>    Tag family (tag36h11, tag25h9, tag16h5) [default: tag36h11]\n");
    printf("  -d, --decimate <val>   Decimate input image by this factor [default: 2.0]\n");
    printf("  -b, --blur <val>       Apply blur to input; negative sharpens [default: 0.0]\n");
    printf("  -t, --threads <n>      Number of CPU threads to use [default: 1]\n");
    printf("  -r, --runs <n>         Repeat detections n times (averages timings) [default: 1]\n");
    printf("      --compare-halide   Run both baseline and Halide hybrid detectors\n");
    printf("      --halide-only      Run only the Halide hybrid detector\n");
    printf("  -h, --help             Show this help message\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *image_path = NULL;
    const char *family_name = "tag36h11";
    double decimate = 1.0;
    double blur = 0.0;
    int num_threads = 8;
    int runs = 1;
    bool compare_halide = false;
    bool halide_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--family") == 0) {
            if (i + 1 < argc) {
                family_name = argv[++i];
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--decimate") == 0) {
            if (i + 1 < argc) {
                decimate = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--blur") == 0) {
            if (i + 1 < argc) {
                blur = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                num_threads = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--runs") == 0) {
            if (i + 1 < argc) {
                runs = std::max(1, atoi(argv[++i]));
            }
        } else if (strcmp(argv[i], "--compare-halide") == 0) {
            compare_halide = true;
        } else if (strcmp(argv[i], "--halide-only") == 0) {
            halide_only = true;
        } else if (argv[i][0] != '-') {
            image_path = argv[i];
        }
    }

    if (halide_only) {
        compare_halide = false;
    }

    if (image_path == NULL) {
        fprintf(stderr, "Error: No image path provided\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("[INFO] Loading image: %s\n", image_path);
    auto img_load_start = std::chrono::high_resolution_clock::now();

    image_u8_t *im = NULL;
    if (str_ends_with(image_path, "pnm") || str_ends_with(image_path, "PNM") ||
        str_ends_with(image_path, "pgm") || str_ends_with(image_path, "PGM")) {
        im = image_u8_create_from_pnm(image_path);
    } else if (str_ends_with(image_path, "jpg") || str_ends_with(image_path, "JPG")) {
        int err = 0;
        pjpeg_t *pjpeg = pjpeg_create_from_file(image_path, 0, &err);
        if (pjpeg == NULL) {
            fprintf(stderr, "Error: Failed to load JPEG image: %s (error %d)\n", image_path, err);
            return 1;
        }
        im = pjpeg_to_u8_baseline(pjpeg);
        pjpeg_destroy(pjpeg);
    } else if (str_ends_with(image_path, "png") || str_ends_with(image_path, "PNG")) {
        im = image_u8_create_from_png(image_path);
    }

    if (im == NULL) {
        fprintf(stderr, "Error: Could not load image: %s\n", image_path);
        return 1;
    }

    auto img_load_end = std::chrono::high_resolution_clock::now();
    double img_load_time = std::chrono::duration<double, std::milli>(img_load_end - img_load_start).count();

    printf("[INFO] Image loaded: %dx%d pixels\n", im->width, im->height);
    printf("[TIMING] Image load time: %.3f ms\n", img_load_time);

    printf("[INFO] Creating tag family: %s\n", family_name);
    apriltag_family_t *tf = NULL;

    if (strcmp(family_name, "tag36h11") == 0) {
        tf = tag36h11_create();
    } else if (strcmp(family_name, "tag25h9") == 0) {
        tf = tag25h9_create();
    } else if (strcmp(family_name, "tag16h5") == 0) {
        tf = tag16h5_create();
    } else {
        fprintf(stderr, "Error: Unknown tag family: %s\n", family_name);
        fprintf(stderr, "       Supported families: tag36h11, tag25h9, tag16h5\n");
        image_u8_destroy(im);
        return 1;
    }

    bool run_baseline = !halide_only;
    bool run_halide = compare_halide || halide_only;

#ifdef APRILTAG_HAVE_HALIDE
    const bool halide_supported = true;
#else
    const bool halide_supported = false;
#endif

    if (run_halide && !halide_supported) {
        fprintf(stderr, "[WARN] Halide acceleration not available in this build. Rebuild with USE_HALIDE=1.\n");
        if (halide_only) {
            fprintf(stderr, "[WARN] Falling back to baseline detector only.\n");
            run_baseline = true;
        }
        run_halide = false;
        compare_halide = false;
        halide_only = false;
    }

    DetectorRun baseline_run;
    DetectorRun halide_run;
    bool baseline_valid = false;
    bool halide_valid = false;
    std::vector<double> baseline_times;
    std::vector<double> halide_times;
    if (run_baseline) {
        baseline_times.reserve(runs);
    }
    if (run_halide) {
        halide_times.reserve(runs);
    }

    if (run_baseline) {
        printf("[INFO] Running baseline AprilTag detection...\n");
        for (int i = 0; i < runs; i++) {
            DetectorRun run = run_detector(tf, im, decimate, blur, num_threads, false);
            baseline_times.push_back(run.detect_time_ms);
            baseline_run = std::move(run);
        }
        baseline_valid = true;
    }

#ifdef APRILTAG_HAVE_HALIDE
    if (run_halide) {
        warm_halide_threshold_pipeline();
    }

    if (run_halide) {
        printf("[INFO] Running Halide-accelerated AprilTag detection...\n");
        for (int i = 0; i < runs; i++) {
            DetectorRun run = run_detector(tf, im, decimate, blur, num_threads, true);
            halide_times.push_back(run.detect_time_ms);
            halide_run = std::move(run);
        }
        halide_valid = true;
    }
#endif

    printf("\n========== DETECTION RESULTS ==========\n");
    if (baseline_valid) {
        print_run_summary("Baseline (UMich reference)", baseline_run);
    }
#ifdef APRILTAG_HAVE_HALIDE
    if (halide_valid) {
        print_run_summary("Halide hybrid (Halide threshold, UMich sparse)", halide_run);
    }
#endif

    if (compare_halide && baseline_valid && halide_valid) {
        report_diff(baseline_run, halide_run);
    }

    printf("\n========== TIMING SUMMARY ==========\n");
    printf("Image load time:   %.3f ms\n", img_load_time);
    if (baseline_valid) {
        const TimingStats stats = compute_stats(baseline_times);
        printf("Baseline detect:   mean %.3f ms, std %.3f ms over %d run(s)\n",
               stats.mean, stats.stddev, runs);
    }
#ifdef APRILTAG_HAVE_HALIDE
    if (halide_valid) {
        const TimingStats stats = compute_stats(halide_times);
        printf("Halide detect:     mean %.3f ms, std %.3f ms over %d run(s)\n",
               stats.mean, stats.stddev, runs);
    }
    if (baseline_valid && halide_valid) {
        const TimingStats base_stats = compute_stats(baseline_times);
        const TimingStats halide_stats = compute_stats(halide_times);
        printf("Time delta (Halide - Baseline): %.3f ms\n",
               halide_stats.mean - base_stats.mean);
    }
#endif
    double total_time = img_load_time;
    if (baseline_valid) {
        const TimingStats stats = compute_stats(baseline_times);
        total_time += stats.mean;
    }
    if (halide_valid) {
        const TimingStats stats = compute_stats(halide_times);
        total_time += stats.mean;
    }
    printf("Total time:        %.3f ms\n", total_time);

    // Cleanup
    if (strcmp(family_name, "tag36h11") == 0) {
        tag36h11_destroy(tf);
    } else if (strcmp(family_name, "tag25h9") == 0) {
        tag25h9_destroy(tf);
    } else if (strcmp(family_name, "tag16h5") == 0) {
        tag16h5_destroy(tf);
    }

    image_u8_destroy(im);

    return 0;
}
