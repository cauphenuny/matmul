#include "isa.hpp"
#include "simd.hpp"

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;

namespace py = pybind11;

using T = int;

namespace kernel {

void trivial(const T* a, const T* b, T* c, int N, int M, int P) {
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

void multithread(const T* a, const T* b, T* c, int N, int M, int P) {
#pragma omp parallel for
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

void chunk(const T* a, const T* b, T* c, int N, int M, int P) {
    const int chunk_size = 32;
    for (int i = 0; i < N; i += chunk_size) {
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < M; ++k) {
                T sum = 0;
                for (int ii = i; ii < i + chunk_size && ii < N; ++ii) {
                    sum += a[ii * M + k] * b[k * P + j];
                }
                c[i * P + j] = sum;
            }
        }
    }
}

void multithread_chunk(const T* a, const T* b, T* c, int N, int M, int P) {
    const int chunk_size = 32;
#pragma omp parallel for
    for (int i = 0; i < N; i += chunk_size) {
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < M; ++k) {
                T sum = 0;
                for (int ii = i; ii < i + chunk_size && ii < N; ++ii) {
                    sum += a[ii * M + k] * b[k * P + j];
                }
                c[i * P + j] = sum;
            }
        }
    }
}

T* transpose(const T* b, int M, int P) {
    T* b_tr = new T[M * P];
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            b_tr[i * P + j] = b[j * M + i];
        }
    }
    return b_tr;
}

void auto_simd(const T* a, const T* b, T* c, int N, int M, int P) {
    std::unique_ptr<T[]> b_tr(transpose(b, M, P));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            T sum = 0;
#pragma omp simd
            for (int k = 0; k < M; ++k) {
                sum += a[i * M + k] * b_tr[j * P + k];
            }
            c[i * P + j] = sum;
        }
    }
}

void multithread_simd(const T* a, const T* b, T* c, int N, int M, int P) {
    std::unique_ptr<T[]> b_tr(transpose(b, M, P));
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            T sum = 0;
#pragma omp simd
            for (int k = 0; k < M; ++k) {
                sum += a[i * M + k] * b_tr[j * P + k];
            }
            c[i * P + j] = sum;
        }
    }
}

}  // namespace kernel

template <auto impl> py::array_t<T> matmul(py::array_t<T> a, py::array_t<T> b) {
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

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(libmatmul, m) {
    m.doc() = "matrix multiplication library";

    m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b"));

    // 指令集信息函数
    m.def(
        "get_target_isa", &isa::get_target_info,
        "Get target instruction set architecture information");
    m.def("get_compiler_info", &isa::get_compiler_info, "Get compiler information");
    m.def("get_build_info", &isa::get_build_info, "Get build configuration information");
    m.def("get_full_info", &isa::get_full_info, "Get complete system information");

    // 指令集检查函数
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
        "trivial", &matmul<kernel::trivial>,
        "Matrix multiplication using a trivial implementation");
    bind(
        "multithread", &matmul<kernel::multithread>,
        "Matrix multiplication using a multithreaded implementation");
    bind("multithread_chunk", &matmul<kernel::multithread_chunk>,
        "Matrix multiplication using a multithreaded chunked implementation");
    bind(
        "auto_simd", &matmul<kernel::auto_simd>,
        "Matrix multiplication using a SIMD implementation (auto generated by libomp)");
    bind("chunk", &matmul<kernel::chunk>, "Matrix multiplication using a chunked implementation");
    bind(
        "multithread_simd", &matmul<kernel::multithread_simd>,
        "Matrix multiplication using a multithreaded SIMD implementation");
    bind(
        "simd", &matmul<kernel::simd>,
        "Matrix multiplication using a SIMD implementation (manually generated)");
}
