import libmatmul as matmul
from typing import Any, Callable
import numpy as np
from timeit import repeat


def run(
    func: Callable[..., Any], n_warmup: int = 2, n_iters: int = 10
) -> float:
    nums = 10
    times = repeat(func, number=nums, repeat=n_iters)
    return np.mean(times[n_warmup:]).item() / nums


def benchmark(func: Callable[[np.ndarray, np.ndarray], np.ndarray], name: str):
    M = 1024
    a = np.random.randint(-10, 10, size=(M, M), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(M, M), dtype=np.int32)
    t0 = run(lambda: func(a, b))
    t1 = run(lambda: a @ b)
    print(t0, t1)
    print(f'{name} over numpy: {(t1/t0):.4f}\n')


def main():
    benchmark(matmul.trivial, "trivial matmul")
    benchmark(matmul.multithread, "multi-threaded matmul")
    benchmark(matmul.chunk, "chunked matmul")
    benchmark(matmul.simd, "SIMD matmul")
    benchmark(matmul.multithread_simd, "multi-threaded SIMD matmul")

if __name__ == "__main__":
    main()
