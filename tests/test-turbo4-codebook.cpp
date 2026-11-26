// turbo4's 16-level codebook must be the Lloyd-Max quantizer of N(0, 1/128), the distribution of a unit block's
// coordinates after the Hadamard rotation. The table shipped before this test was about 0.6 x those levels: a tenth
// of the coordinates saturated at the outer level and the round-trip error on Gaussian data was 1.8 % of the energy
// against the 0.95 % a 16-level Lloyd-Max quantizer gives. The test quantizes Gaussian blocks, dequantizes them
// (rotated domain), applies the inverse rotation and bounds the relative squared error.

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size);

static double relative_error(ggml_type type, const std::vector<float> & x, int block) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    const int64_t n = (int64_t) x.size();
    std::vector<uint8_t> q(ggml_row_size(type, n));
    std::vector<float> y(x.size());
    traits->from_float_ref(x.data(), q.data(), n);
    traits->to_float(q.data(), y.data(), n);
    for (int64_t i = 0; i < n; i += block) {
        turbo_cpu_fwht_inverse(y.data() + i, block);
    }
    double err = 0.0, energy = 0.0;
    for (int64_t i = 0; i < n; i++) {
        err    += ((double) x[i] - y[i]) * ((double) x[i] - y[i]);
        energy += (double) x[i] * x[i];
    }
    return err / energy;
}

int main() {
    const int block  = 128;
    const int blocks = 512;
    std::mt19937 rng(20261126);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x((size_t) block * blocks);
    for (int b = 0; b < blocks; b++) {
        const float scale = 0.05f + 4.0f * (float) b / blocks;   // block norms from about 0.6 to 46
        for (int i = 0; i < block; i++) {
            x[(size_t) b * block + i] = scale * gauss(rng);
        }
    }

    const double e4 = relative_error(GGML_TYPE_TURBO4_0, x, block);
    const double e3 = relative_error(GGML_TYPE_TURBO3_0, x, block);
    printf("turbo4 relative squared error: %.5f (16-level Lloyd-Max on a Gaussian: 0.0095)\n", e4);
    printf("turbo3 relative squared error: %.5f\n", e3);

    int failed = 0;
    if (!(e4 < 0.0120)) {
        printf("FAIL: turbo4 error %.5f is not below 0.0120 (the mis-scaled table gave 0.018)\n", e4);
        failed = 1;
    }
    if (!(e4 < e3)) {
        printf("FAIL: turbo4 (%.5f) is not more accurate than turbo3 (%.5f)\n", e4, e3);
        failed = 1;
    }
    if (!failed) {
        printf("OK\n");
    }
    return failed;
}
