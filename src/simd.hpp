#pragma once

#include "isa.hpp"

#include <memory>

using T = int;

namespace kernel {

T* transpose(const T* b, int M, int P);
void auto_simd(const T* a, const T* b, T* c, int N, int M, int P);

inline void simd(const T* a, const T* b, T* c, int N, int M, int P) {
    std::unique_ptr<T[]> b_tr(transpose(b, M, P));

    if constexpr (isa::check::has_neon()) {
        const int simd_width = 4;
        const int unroll_factor = 8;

#define SIMD_LOAD_AND_MULTIPLY(idx)                                                  \
    do {                                                                             \
        int32x4_t a_vec##idx = vld1q_s32(&a[i * M + k + simd_width * (idx - 1)]);    \
        int32x4_t b_vec##idx = vld1q_s32(&b_tr[j * M + k + simd_width * (idx - 1)]); \
        sum_vec##idx = vmlaq_s32(sum_vec##idx, a_vec##idx, b_vec##idx);              \
    } while (0)

#define DECLARE_SUM_VEC(idx) int32x4_t sum_vec##idx = vdupq_n_s32(0)

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < P; ++j) {
                // 声明8个累加器
                DECLARE_SUM_VEC(1);
                DECLARE_SUM_VEC(2);
                DECLARE_SUM_VEC(3);
                DECLARE_SUM_VEC(4);
                DECLARE_SUM_VEC(5);
                DECLARE_SUM_VEC(6);
                DECLARE_SUM_VEC(7);
                DECLARE_SUM_VEC(8);
                int k = 0;

                // 循环展开：每次处理8个SIMD向量
                for (; k <= M - simd_width * unroll_factor; k += simd_width * unroll_factor) {
                    SIMD_LOAD_AND_MULTIPLY(1);
                    SIMD_LOAD_AND_MULTIPLY(2);
                    SIMD_LOAD_AND_MULTIPLY(3);
                    SIMD_LOAD_AND_MULTIPLY(4);
                    SIMD_LOAD_AND_MULTIPLY(5);
                    SIMD_LOAD_AND_MULTIPLY(6);
                    SIMD_LOAD_AND_MULTIPLY(7);
                    SIMD_LOAD_AND_MULTIPLY(8);
                }

                // 合并累加器：分层合并减少延迟
                sum_vec1 = vaddq_s32(sum_vec1, sum_vec2);
                sum_vec3 = vaddq_s32(sum_vec3, sum_vec4);
                sum_vec5 = vaddq_s32(sum_vec5, sum_vec6);
                sum_vec7 = vaddq_s32(sum_vec7, sum_vec8);

                sum_vec1 = vaddq_s32(sum_vec1, sum_vec3);
                sum_vec5 = vaddq_s32(sum_vec5, sum_vec7);

                sum_vec1 = vaddq_s32(sum_vec1, sum_vec5);

                // 处理剩余的完整SIMD向量
                for (; k <= M - simd_width; k += simd_width) {
                    int32x4_t a_vec = vld1q_s32(&a[i * M + k]);
                    int32x4_t b_vec = vld1q_s32(&b_tr[j * M + k]);
                    sum_vec1 = vmlaq_s32(sum_vec1, a_vec, b_vec);
                }

                T sum = vaddvq_s32(sum_vec1);

                // 处理剩余的标量元素
                for (; k < M; ++k) {
                    sum += a[i * M + k] * b_tr[j * M + k];
                }

                c[i * P + j] = sum;
            }
        }

#undef SIMD_LOAD_AND_MULTIPLY
#undef DECLARE_SUM_VEC
    } else {
        return auto_simd(a, b, c, N, M, P);
    }
}

}  // namespace kernel
