import libmatmul as matmul
import os
import numpy as np
from timeit import timeit


def test_hello():
    pass

def check(func):
    a = np.array([
        [1, 2, 3, 4], [5, 6, 7, 8]
    ])
    b = np.array([
        [1, 2], [3, 4], [5, 6], [7, 8]
    ])
    # a = np.array([[1, 2, 3, 4]])
    # b = np.array([[1], [2], [3], [4]])
    c = func(a, b)
    c_ans = np.matmul(a, b)
    print(f"Result:\n{c}")
    print(f"Expected:\n{c_ans}")
    assert np.allclose(c, c_ans), "Matrix multiplication result is incorrect."

def test_trivial():
    check(matmul.trivial)

def test_multithread():
    check(matmul.multithread)

def test_chunk():
    check(matmul.chunk)

def test_autosimd():
    check(matmul.auto_simd)

def test_simd():
    check(matmul.simd)

def test_transpose():
    check(matmul.transpose)

def test_simd_optimized():
    check(matmul.simd_optimized)
    pass

if __name__ == "__main__":
    test_hello()
    test_trivial()
    test_multithread()
    test_chunk()
    test_transpose()
    test_autosimd()
    test_simd()
    test_simd_optimized()