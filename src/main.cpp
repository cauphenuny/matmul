#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
using namespace pybind11::literals;

namespace py = pybind11;

using T = int;

namespace kernal{

void trivial(int* a, int* b, int* c, int N, int M, int P) {
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

void multithread(int* a, int* b, int* c, int N, int M, int P) {
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

}

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
    m.doc() = "Simple test module";

    m.def("add", &add, "A function that adds two numbers", py::arg("a"), py::arg("b"));

    m.def(
        "trivial", &matmul<kernal::trivial>, "Matrix multiplication using a trivial implementation",
        py::arg("a"), py::arg("b"));

    m.def(
        "multithread", &matmul<kernal::multithread>, "Matrix multiplication using a multithreaded implementation",
        py::arg("a"), py::arg("b"));
}
