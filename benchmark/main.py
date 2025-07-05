import libmatmul as matmul
from typing import Any, Callable, TypeAlias
import numpy as np
from timeit import repeat
from numpy.typing import NDArray

# Type aliases
IntArray: TypeAlias = NDArray[np.int32]
MatmulFunc: TypeAlias = Callable[[IntArray, IntArray], IntArray]


def run(func: Callable[..., Any], n_warmup: int = 1, n_iters: int = 5) -> float:
    nums = 10
    times = repeat(func, number=nums, repeat=n_iters)
    return np.mean(times[n_warmup:]).item() / nums


def benchmark(func: MatmulFunc, name: str, N: int = 1024) -> float:
    print(f"benchmarking {name} with N={N}...", end=" ")
    a = np.random.randint(-10, 10, size=(N, N), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(N, N), dtype=np.int32)
    t0 = run(lambda: func(a, b)) * 1000
    print(f"{t0:.4f}ms")
    return t0


def main() -> None:
    funcs: list[tuple[MatmulFunc, str]] = [
        (np.matmul, "numpy"),
        (matmul.trivial, "trivial"),
        (matmul.multithread, "multi-thread"),
        (matmul.chunk, "chunk"),
        (matmul.multithread_chunk, "chunk, multi-thread"),
        (matmul.auto_simd, "SIMD (auto)"),
        (matmul.simd, "SIMD (manual)"),
        (matmul.multithread_simd, "SIMD, multi-thread"),
    ]

    results: list[tuple[str, float, float]] = []

    for func, name in funcs:
        time_ms = benchmark(func, name, 1024)
        results.append((name, time_ms, 0.0))
    
    for i, (name, time_ms, _) in enumerate(results):
        speedup = results[0][1] / time_ms
        results[i] = (name, time_ms, speedup)

    print("\nResult")
    print("| Method              | Time(ms) | Speedup over `np.matmul` |")
    print("|---------------------|----------|--------------------------|")
    for name, time_ms, speedup in results:
        print(f"| {name:<19} | {time_ms:8.4f} | {speedup:8.4f}x                |")


if __name__ == "__main__":
    main()
