cmake -S . -B cmake-build -DUSE_HALIDE=ON

cmake --build cmake-build

./cmake-build/apriltag_timing apriltags.jpg --compare-halide --runs 3