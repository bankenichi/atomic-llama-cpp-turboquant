// S1_32, the sign type of moeq's binary-factor experts: its contract and its CPU dot product.
//
//  - traits: 32 values per 4-byte block, quantized, named s1_32;
//  - layout: bit j of the block's uint32 (bit j%8 of byte j/8) is element j, 1 = +1, 0 = -1 (Q1_0's convention);
//  - quantization stores sign(x) (x >= 0, -0 and NaN give +1), dequantization gives exactly +-1, and the round trip
//    quantize(dequantize(bits)) returns the same bits; the CPU quantizer equals the reference;
//  - the CPU dot with Q8_0 activations equals the exact sum over blocks of d_b * sum_j(+-q_j), computed here
//    independently in double, to float rounding;
//  - every bit pattern validates.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAILED %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++n_failed; } } while (0)

static float q8_0_scale(const uint8_t * block) {
    uint16_t h;
    std::memcpy(&h, block, 2);
    return ggml_fp16_to_fp32(h);
}

int main() {
    ggml_cpu_init(); // the CPU kernels read fp16 scales through tables it fills

    const ggml_type type = GGML_TYPE_S1_32;
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    const ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(type);

    // traits
    CHECK(std::strcmp(ggml_type_name(type), "s1_32") == 0, "type name %s", ggml_type_name(type));
    CHECK(ggml_blck_size(type) == 32 && ggml_type_size(type) == 4, "block %lld size %zu",
          (long long) ggml_blck_size(type), ggml_type_size(type));
    CHECK(ggml_is_quantized(type), "not quantized");
    CHECK(ggml_row_size(type, 480) == 60, "row size of 480 signs = %zu", ggml_row_size(type, 480));
    CHECK(tc->vec_dot_type == GGML_TYPE_Q8_0, "vec_dot_type %s", ggml_type_name(tc->vec_dot_type));

    // layout: element 5 negative, all others non-negative -> every bit set but bit 5
    {
        std::vector<float> x(32, 0.5f);
        x[5] = -2.0f;
        uint32_t bits = 0;
        tt->from_float_ref(x.data(), &bits, 32);
        CHECK(bits == 0xFFFFFFDFu, "bits %08x", bits);
        // byte order: element 9 is bit 1 of byte 1
        x.assign(32, 1.0f);
        x[9] = -1.0f;
        uint8_t bytes[4];
        tt->from_float_ref(x.data(), bytes, 32);
        CHECK(bytes[0] == 0xFF && bytes[1] == 0xFD && bytes[2] == 0xFF && bytes[3] == 0xFF, "bytes %02x %02x %02x %02x",
              bytes[0], bytes[1], bytes[2], bytes[3]);
    }

    // sign rule, exact dequantization, round trip, CPU quantizer == reference, validation, quantize_chunk
    std::mt19937 rng(42);
    {
        const int n = 32*40;
        std::uniform_real_distribution<float> u(-3.0f, 3.0f);
        std::vector<float> x(n);
        for (float & v : x) {
            v = u(rng);
        }
        x[0] = 0.0f;
        x[1] = -0.0f;
        x[2] = NAN;
        x[3] = -1e-30f;
        x[4] = INFINITY;
        x[5] = -INFINITY;

        std::vector<uint8_t> q(ggml_row_size(type, n)), q_cpu(q.size()), q2(q.size()), q_chunk(q.size());
        std::vector<float> y(n);
        tt->from_float_ref(x.data(), q.data(), n);
        tc->from_float(x.data(), q_cpu.data(), n);
        CHECK(q == q_cpu, "CPU quantizer differs from the reference");
        tt->to_float(q.data(), y.data(), n);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const float want = x[i] < 0.0f ? -1.0f : 1.0f;
            bad += y[i] != want;
        }
        CHECK(bad == 0, "%d values do not dequantize to sign(x)", bad);
        tt->from_float_ref(y.data(), q2.data(), n);
        CHECK(q == q2, "quantize(dequantize(bits)) != bits");
        const size_t written = ggml_quantize_chunk(type, x.data(), q_chunk.data(), 0, 4, n/4, nullptr);
        CHECK(written == q.size() && q_chunk == q, "ggml_quantize_chunk: %zu bytes, same bits %d", written, (int) (q_chunk == q));
        CHECK(ggml_validate_row_data(type, q.data(), q.size()), "row data rejected");
        std::vector<uint8_t> all_ones(q.size(), 0xFF), all_zeros(q.size(), 0x00);
        CHECK(ggml_validate_row_data(type, all_ones.data(), all_ones.size()) &&
              ggml_validate_row_data(type, all_zeros.data(), all_zeros.size()), "a bit pattern was rejected");
    }

    // dot with Q8_0 activations against the exact integer sum
    {
        const ggml_type_traits_cpu * t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int n : { 32, 64, 320, 416, 480, 1024, 2048, 4096 }) {
            for (int trial = 0; trial < 20; ++trial) {
                std::vector<float> w(n), a(n);
                for (int i = 0; i < n; ++i) {
                    w[i] = nd(rng);
                    a[i] = nd(rng) * (trial % 3 == 0 ? 100.0f : 1.0f);
                }
                if (trial == 1) {
                    a[n/2] = 3000.0f; // a massive activation in one block
                }
                std::vector<uint8_t> qw(ggml_row_size(type, n)), qa(ggml_row_size(GGML_TYPE_Q8_0, n));
                tc->from_float(w.data(), qw.data(), n);
                t8->from_float(a.data(), qa.data(), n);

                double ref = 0.0, mag = 0.0;
                for (int b = 0; b < n/32; ++b) {
                    const uint8_t * blk = qa.data() + (size_t) b*34;
                    const int8_t  * qs  = (const int8_t *) (blk + 2);
                    uint32_t bits;
                    std::memcpy(&bits, qw.data() + (size_t) b*4, 4);
                    long long sumi = 0, sumabs = 0;
                    for (int j = 0; j < 32; ++j) {
                        sumi   += ((bits >> j) & 1) ? qs[j] : -qs[j];
                        sumabs += std::llabs(qs[j]);
                    }
                    ref += (double) q8_0_scale(blk) * (double) sumi;
                    mag += (double) q8_0_scale(blk) * (double) sumabs;
                }
                float got = NAN;
                tc->vec_dot(n, &got, 0, qw.data(), 0, qa.data(), 0, 1);
                const double err = std::fabs((double) got - ref);
                CHECK(err <= 1e-6 * mag + 1e-30, "n=%d trial=%d: dot %.9g, exact %.9g (|err| %.3g, scale %.3g)",
                      n, trial, (double) got, ref, err, mag);
            }
        }
    }

    if (n_failed == 0) {
        std::printf("test-s1-32: all checks passed\n");
    }
    return n_failed == 0 ? 0 : 1;
}
