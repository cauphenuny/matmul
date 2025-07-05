#pragma once

#include "isa.hpp"

#include <memory>

using T = int;

namespace kernel {

// 前向声明
T* transpose(const T* b, int M, int P);
void auto_simd(const T* a, const T* b, T* c, int N, int M, int P);

inline void simd(const T* a, const T* b, T* c, int N, int M, int P) {
    std::unique_ptr<T[]> b_tr(transpose(b, M, P));

    if constexpr (isa::check::has_neon()) {
        const int simd_width = 4;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < P; ++j) {
                int32x4_t sum_vec = vdupq_n_s32(0);
                int k = 0;

                for (; k <= M - simd_width; k += simd_width) {
                    int32x4_t a_vec = vld1q_s32(&a[i * M + k]);
                    int32x4_t b_vec = vld1q_s32(&b_tr[j * M + k]);
                    sum_vec = vmlaq_s32(sum_vec, a_vec, b_vec);
                }

                T sum = vaddvq_s32(sum_vec);

                for (; k < M; ++k) {
                    sum += a[i * M + k] * b_tr[j * M + k];
                }

                c[i * P + j] = sum;
            }
        }
    } else {
        return auto_simd(a, b, c, N, M, P);
    }
}

}  // namespace kernel
