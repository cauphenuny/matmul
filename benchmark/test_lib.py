import libmatmul as matmul
import os
import numpy as np
from timeit import timeit


def test_hello():
    pass

def check(func):
    N = 1024
    a = np.random.randint(-10, 10, size=(N, N)).astype(np.int32)
    b = np.random.randint(-10, 10, size=(N, N)).astype(np.int32)
    c = func(a, b)
    c_ans = np.matmul(a, b)
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