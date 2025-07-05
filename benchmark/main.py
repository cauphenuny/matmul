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


def benchmark(func: MatmulFunc, name: str, N: int = 512) -> tuple[float, float]:
    a = np.random.randint(-10, 10, size=(N, N), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(N, N), dtype=np.int32)
    t0 = run(lambda: func(a, b))
    t1 = run(lambda: a @ b)
    print(t0, t1)
    print(f"{name} over numpy: {(t1 / t0):.4f}\n")
    return t0 * 1000, t1 / t0


def main() -> None:
    funcs: list[tuple[MatmulFunc, str]] = [
        (matmul.trivial, "trivial"),
        (matmul.multithread, "multi-thread"),
        (matmul.chunk, "chunk"),
        (matmul.auto_simd, "SIMD (auto)"),
        (matmul.multithread_simd, "multi-thread SIMD"),
    ]

    results: list[tuple[str, float, float]] = []

    for func, name in funcs:
        time_ms, speedup = benchmark(func, name)
        results.append((name, time_ms, speedup))

    print("## Result")
    print("| Method              | Time(ms) | Speedup over `np.matmul` |")
    print("|---------------------|----------|--------------------------|")
    for name, time_ms, speedup in results:
        print(f"| {name:<19} | {time_ms:8.4f} | {speedup:8.4f}x          |")


if __name__ == "__main__":
    main()
