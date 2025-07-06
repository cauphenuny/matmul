#include "isa.hpp"
#include "simd.hpp"
#include "typedef.h"

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;

namespace py = pybind11;

namespace kernel {

void trivial(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            T sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += a[i * M + k] * b[k * P + j];
            }
            c[i * P + j] = sum;
        }
    }
}

void transpose_iter(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            auto tmp = a[i * M + k];
            for (int j = 0; j < P; ++j) {
                c[i * P + j] += tmp * b[k * P + j];
            }
        }
    }
}

inline void multithread(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            auto tmp = a[i * M + k];
            for (int j = 0; j < P; ++j) {
                c[i * P + j] += tmp * b[k * P + j];
            }
        }
    }
}

void chunk(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
    const int chunk_size = 64;
    for (int ii = 0; ii < N; ii += chunk_size) {
        for (int kk = 0; kk < M; kk += chunk_size) {
            for (int jj = 0; jj < P; jj += chunk_size) {
                for (int i = ii; i < std::min(ii + chunk_size, N); ++i) {
                    for (int k = kk; k < std::min(kk + chunk_size, M); ++k) {
                        T tmp = a[i * M + k];
                        for (int j = jj; j < std::min(jj + chunk_size, P); ++j) {
                            c[i * P + j] += tmp * b[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

void multithread_chunk(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
    const int chunk_size = 64;
#pragma omp parallel for
    for (int ii = 0; ii < N; ii += chunk_size) {
        for (int kk = 0; kk < M; kk += chunk_size) {
            for (int jj = 0; jj < P; jj += chunk_size) {
                for (int i = ii; i < std::min(ii + chunk_size, N); ++i) {
                    for (int k = kk; k < std::min(kk + chunk_size, M); ++k) {
                        T tmp = a[i * M + k];
                        for (int j = jj; j < std::min(jj + chunk_size, P); ++j) {
                            c[i * P + j] += tmp * b[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

T* transpose(const T* b, int M, int P) {
    T* b_tr = new T[M * P];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            b_tr[j * M + i] = b[i * P + j];
        }
    }
    // printf("M = %d, P = %d\n", M, P);
    // for (int i = 0; i < M * P; i++) {
    //     printf("%d ", b[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < M * P; i++) {
    //     printf("%d ", b_tr[i]);
    // }
    // printf("\n");
    return b_tr;
}

void transpose_data(const T* a, const T* b, T* c, int N, int M, int P) {
    memset(c, 0, N * P * sizeof(T));
    std::unique_ptr<T[]> b_tr(transpose(b, M, P));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            T sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += a[i * M + k] * b_tr[j * M + k];
            }
            c[i * P + j] = sum;
        }
    }
}

}  // namespace kernel

template <auto impl> py::array_t<T> np_matmul(py::array_t<T> a, py::array_t<T> b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("Incompatible shapes for matrix multiplication");
    }
    const auto N = a_shape[0], M = a_shape[1], P = b_shape[1];

    auto c = py::array_t<T>({a_shape[0], b_shape[1]});
    auto a_ptr = a.mutable_data();
    auto b_ptr = b.mutable_data();
    auto c_ptr = c.mutable_data();

    impl(a_ptr, b_ptr, c_ptr, N, M, P);
    return c;
}

template <auto impl> void c_matmul(int N, int** matrixA, int** matrixB, int** matrixC) {
    std::unique_ptr<int[]> a(new int[N * N]);
    std::unique_ptr<int[]> b(new int[N * N]);
    std::unique_ptr<int[]> c(new int[N * N]);
    for (int i = 0; i < N; ++i) {
        memcpy(a.get() + i * N, matrixA[i], N * sizeof(int));
        memcpy(b.get() + i * N, matrixB[i], N * sizeof(int));
    }
    impl(a.get(), b.get(), c.get(), N, N, N);
    for (int i = 0; i < N; ++i) {
        memcpy(matrixC[i], c.get() + i * N, N * sizeof(int));
    }
}

extern "C" {
void matrixmultiply(int N, int** matrixA, int** matrixB, int** matrixC) {
    return c_matmul<kernel::auto_simd>(N, matrixA, matrixB, matrixC);
}
}

PYBIND11_MODULE(libmatmul, m) {
    m.doc() = "matrix multiplication library";

    m.def(
        "get_target_isa", &isa::get_target_info,
        "Get target instruction set architecture information");
    m.def("get_compiler_info", &isa::get_compiler_info, "Get compiler information");
    m.def("get_build_info", &isa::get_build_info, "Get build configuration information");
    m.def("get_full_info", &isa::get_full_info, "Get complete system information");

    m.def("has_sse2", &isa::check::has_sse2, "Check if SSE2 is available");
    m.def("has_avx", &isa::check::has_avx, "Check if AVX is available");
    m.def("has_avx2", &isa::check::has_avx2, "Check if AVX2 is available");
    m.def("has_avx512", &isa::check::has_avx512, "Check if AVX512 is available");
    m.def("has_neon", &isa::check::has_neon, "Check if NEON is available");
    m.def("has_fma", &isa::check::has_fma, "Check if FMA is available");

    auto bind = [&m](const char* name, auto func, const char* desc) {
        m.def(name, func, desc, py::arg("a"), py::arg("b"));
    };
    bind(
        "trivial", &np_matmul<kernel::trivial>,
        "Matrix multiplication using a trivial implementation");
    bind(
        "transpose_iter", &np_matmul<kernel::transpose_iter>,
        "Matrix multiplication using a transposed loop iterator implementation");
    bind(
        "multithread", &np_matmul<kernel::multithread>,
        "Matrix multiplication using a multithreaded implementation");
    bind(
        "multithread_chunk", &np_matmul<kernel::multithread_chunk>,
        "Matrix multiplication using a multithreaded chunked implementation");
    bind(
        "auto_simd", &np_matmul<kernel::auto_simd>,
        "Matrix multiplication using a SIMD implementation (auto generated by libomp)");
    bind("chunk", &np_matmul<kernel::chunk>, "Matrix multiplication using a chunked implementation");
    bind(
        "transpose", &np_matmul<kernel::transpose_data>,
        "Matrix multiplication using a transposed implementation");
    bind(
        "multithread_simd", &np_matmul<kernel::multithread_simd>,
        "Matrix multiplication using a multithreaded SIMD implementation");
    bind(
        "simd", &np_matmul<kernel::simd>,
        "Matrix multiplication using a SIMD implementation (manually generated)");
    bind(
        "simd_optimized", &np_matmul<kernel::simd_optimized>,
        "Matrix multiplication using an optimized SIMD implementation with prefetch");
    bind(
        "simd_arm_sme", &np_matmul<kernel::simd_arm_sme>,
        "Matrix multiplication using ARM SME instructions");
}
