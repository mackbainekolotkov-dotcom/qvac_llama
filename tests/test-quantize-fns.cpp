// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;

static const char* RESULT_STR[] = {"ok", "FAILED"};


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

// Total quantization error on test data
static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

// Total quantization error on test data
static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    // FIXME: why is done twice?
    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

// Total dot product error
static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

static int test_vec_dot_f32(bool verbose) {
    const auto * f32 = ggml_get_type_traits_cpu(GGML_TYPE_F32);
    int num_failed = 0;
    for (int n : {1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 33, 63, 67, 127, 129, 193, 255, 1023}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        generate_data(0.0, n, a.data());
        generate_data(1.0, n, b.data());

        float result = 0.0f;
        f32->vec_dot(n, &result, 0, a.data(), 0, b.data(), 0, 1);
        const float ref = dot_product(a.data(), b.data(), n);
        const float error = fabsf(result - ref) / n;

        const bool failed = !(error < MAX_QUANTIZATION_REFERENCE_ERROR);
        num_failed += failed;
        if (failed || verbose) {
            printf(" f32 vec_dot n=%4d:                 %s (ref=%f got=%f err=%f)\n",
                   n, RESULT_STR[failed], ref, result, error);
        }
    }
    return num_failed;
}

static int test_vec_dot_q(bool verbose) {
    int num_failed = 0;

    const size_t test_size = 32 * 128;

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        // deprecated - skip
        if (qfns->blck_size == 0) {
            continue;
        }

        const ggml_type ei = (ggml_type)i;

        printf("Testing %s\n", ggml_type_name((ggml_type) i));
        ggml_quantize_init(ei);

        if (qfns_cpu->from_float && qfns->to_float) {
            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            const float max_quantization_error =
                type == GGML_TYPE_Q1_0    ? MAX_QUANTIZATION_TOTAL_ERROR_BINARY :
                type == GGML_TYPE_TQ1_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_TQ2_0   ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_0    ? MAX_QUANTIZATION_TOTAL_ERROR_TERNARY :
                type == GGML_TYPE_Q2_K    ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_IQ2_S   ? MAX_QUANTIZATION_TOTAL_ERROR_2BITS :
                type == GGML_TYPE_Q3_K    ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_S   ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS :
                type == GGML_TYPE_IQ3_XXS ? MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS :
                type == GGML_TYPE_NVFP4   ? MAX_QUANTIZATION_TOTAL_ERROR_FP4 : MAX_QUANTIZATION_TOTAL_ERROR;
            bool failed = !(total_error < max_quantization_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data.data());
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data.data(), test_data2.data());
            const float max_allowed_error = type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS ||
                type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S
                ? MAX_DOT_PRODUCT_ERROR_LOWBIT
                : type == GGML_TYPE_Q1_0
                ? MAX_DOT_PRODUCT_ERROR_BINARY
                : type == GGML_TYPE_TQ1_0 || type == GGML_TYPE_TQ2_0 || type == GGML_TYPE_Q2_0
                ? MAX_DOT_PRODUCT_ERROR_TERNARY
                : type == GGML_TYPE_NVFP4
                ? MAX_DOT_PRODUCT_ERROR_FP4
                : MAX_DOT_PRODUCT_ERROR;
            failed = !(vec_dot_error < max_allowed_error);
            num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
        }
    }

    return num_failed;
}

// synthetic blocks that exercise the q4_hqq numerical edge cases
static const char * q4_hqq_case_name(int c) {
    switch (c) {
        case 0: return "uniform noise";
        case 1: return "gaussian";
        case 2: return "all zero";
        case 3: return "constant non-zero";
        case 4: return "single outlier";
        case 5: return "extreme dynamic range";
        case 6: return "denormals";
        case 7: return "narrow band far from zero";
        case 8: return "huge magnitude (scale underflow)";
    }
    return "?";
}

static void q4_hqq_fill_case(int c, size_t n, float * dst) {
    std::mt19937 rng(1234 + c);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        switch (c) {
            case 0: dst[i] = uni(rng); break;
            case 1: dst[i] = gauss(rng); break;
            case 2: dst[i] = 0.0f; break;
            case 3: dst[i] = 0.37f; break;
            // one large value per block, the rest small
            case 4: dst[i] = (i % 32) == 7 ? 100.0f : 0.01f*gauss(rng); break;
            case 5: dst[i] = (i % 2) == 0 ? 1e-6f : 1e6f; break;
            case 6: dst[i] = 1e-40f*(1.0f + (i % 32)); break;
            // range so small that 15/(max-min) does not fit in fp16
            case 7: dst[i] = 1000.0f + 1e-6f*(i % 32); break;
            // range so wide that 15/(max-min) underflows the f16 grid to zero
            case 8: dst[i] = 1e13f*(1.0f + 0.01f*(i % 32)); break;
        }
    }
}

// the round trip must stay finite and must not lose to q4_1, which has the same block layout
static int test_q4_hqq_edge_cases(bool verbose) {
    const auto * hqq     = ggml_get_type_traits(GGML_TYPE_Q4_HQQ);
    const auto * hqq_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_HQQ);
    const auto * q41     = ggml_get_type_traits(GGML_TYPE_Q4_1);
    const auto * q41_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q4_1);

    const size_t test_size = 32*64;

    std::vector<float>   src(test_size);
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float>   out_hqq(test_size);
    std::vector<float>   out_q41(test_size);

    int num_failed = 0;

    for (int c = 0; c < 9; c++) {
        q4_hqq_fill_case(c, test_size, src.data());

        hqq_cpu->from_float(src.data(), tmp_q.data(), test_size);
        hqq->to_float(tmp_q.data(), out_hqq.data(), test_size);

        q41_cpu->from_float(src.data(), tmp_q.data(), test_size);
        q41->to_float(tmp_q.data(), out_q41.data(), test_size);

        size_t n_bad = 0;
        float  amax  = 0.0f;
        for (size_t i = 0; i < test_size; i++) {
            if (!std::isfinite(out_hqq[i])) {
                n_bad++;
            }
            amax = std::max(amax, fabsf(src[i]));
        }

        const float rmse_hqq = array_rmse(src.data(), out_hqq.data(), test_size);
        const float rmse_q41 = array_rmse(src.data(), out_q41.data(), test_size);

        // 1. the round trip must never produce a non-finite value
        bool failed = n_bad > 0 || !std::isfinite(rmse_hqq);

        // 2. the error must stay bounded by the dynamic range of the input
        failed = failed || !(rmse_hqq <= amax + 1e-9f);

        // 3. must not lose to q4_1 by more than the reciprocal storage costs. skip q4_1 when
        //    it is not finite itself, which happens on case 5
        if (c != 7 && c != 8 && std::isfinite(rmse_q41)) {
            failed = failed || !(rmse_hqq <= rmse_q41*1.10f + 1e-9f);
        }

        // 4. narrow band: absolute error instead. (q-zero)/scale carries an offset of |min|,
        //    so f16 precision floors the error near |min|*2^-11 however narrow the band is
        if (c == 7) {
            float amin = INFINITY;
            for (size_t i = 0; i < test_size; i++) {
                amin = std::min(amin, fabsf(src[i]));
            }
            failed = failed || !(rmse_hqq <= amin*(1.0f/1024.0f) + 1e-9f);
        }

        // case 8 gets assertions 1 and 2 only: a block spanning 3e12 needs a scale f16 cannot
        // hold, so dequantize_row_q4_hqq only promises finiteness through its s != 0 guard

        num_failed += failed;
        if (failed || verbose) {
            printf("q4_hqq edge case %-34s %s (non-finite=%zu rmse=%g q4_1 rmse=%g)\n",
                   q4_hqq_case_name(c), RESULT_STR[failed], n_bad, rmse_hqq, rmse_q41);
        }
    }

    return num_failed;
}

int main(int argc, char * argv[]) {
    bool verbose = false;

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    ggml_cpu_init();

    int num_failed = 0;

    num_failed += test_vec_dot_f32(verbose);
    num_failed += test_vec_dot_q(verbose);
    num_failed += test_q4_hqq_edge_cases(verbose);

    if (num_failed || verbose) {
        printf("%d tests failed\n", num_failed);
    }

    return num_failed > 0;
}
