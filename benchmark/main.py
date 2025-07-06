import libmatmul as matmul
from typing import Any, Callable, TypeAlias
import numpy as np
import time
from timeit import repeat
from numpy.typing import NDArray
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Type aliases
IntArray: TypeAlias = NDArray[np.int32]
MatmulFunc: TypeAlias = Callable[[IntArray, IntArray], IntArray]


def py_matmul(a: IntArray, b: IntArray) -> IntArray:
    n = a.shape[0]
    assert a.shape[1] == b.shape[0], "Incompatible shapes for matrix multiplication"
    m = a.shape[1]
    p = b.shape[1]
    c = np.zeros((n, p), dtype=np.int32)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                c[i, j] += a[i, k] * b[k, j]
    return c


def run(func: Callable[..., Any], n_warmup: int = 1, n_iters: int = 3) -> float:
    nums = 2
    times = repeat(func, number=nums, repeat=n_iters)
    return np.mean(times[n_warmup:]).item() / nums


def benchmark(func: MatmulFunc, name: str, n: int = 512) -> float:
    print(f"benchmarking {name} with N={n}...", end=" ")
    a = np.random.randint(-10, 10, size=(n, n), dtype=np.int32)
    b = np.random.randint(-10, 10, size=(n, n), dtype=np.int32)
    t0 = run(lambda: func(a, b)) * 1000
    print(f"{t0:.4f}ms")
    return t0


def visualize_markdown(
    funcs: list[tuple[MatmulFunc, str]], n: int = 512, gap: int = 10
) -> None:
    results: list[tuple[str, float, float]] = []

    for func, name in funcs:
        time_ms = benchmark(func, name, n)
        results.append((name, time_ms, 0.0))
        time.sleep(gap)

    for i, (name, time_ms, _) in enumerate(results):
        speedup = results[0][1] / time_ms
        results[i] = (name, time_ms, speedup)

    print("\nResult")
    print("| Method              | Time(ms) | Speedup over `np.matmul` |")
    print("|---------------------|----------|--------------------------|")
    for name, time_ms, speedup in results:
        print(f"| {name:<19} | {time_ms:8.4f} | {speedup:8.4f}x                |")


def visualize_matplotlib(funcs: list[tuple[MatmulFunc, str]], n: int, gap: int) -> None:
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1: 不同矩阵大小的耗时曲线
    sizes: list[int] = [64]
    while sizes[-1] < n:
        sizes.append(sizes[-1] * 2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(funcs)))

    print("Benchmarking different matrix sizes...")
    for i, (func, name) in enumerate(funcs):
        times = []
        for size in sizes:
            time_ms = benchmark(func, name, size)
            times.append(time_ms)

        ax1.plot(sizes, times, marker="o", label=name,
                 color=colors[i], linewidth=2)

        time.sleep(gap)

    ax1.set_xlabel("Matrix Size (N)", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Matrix Multiplication Performance vs Size", fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")

    # 图2: 特定大小时的柱状图
    print(f"\nBenchmarking all methods with size {n}...")
    names = []
    times = []

    for func, name in funcs:
        time_ms = benchmark(func, name, n)
        names.append(name)
        times.append(time_ms)
        time.sleep(gap)

    bars = ax2.bar(range(len(names)), times,
                   color=colors[: len(names)], alpha=0.7)
    ax2.set_xlabel("Method", fontsize=12)
    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title(f"Matrix Multiplication Performance (Size={n})", fontsize=14)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # 在柱状图上显示数值
    for bar, time_ms in zip(bars, times):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time_ms:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()

    # 计算并打印加速比
    print(f"\nSpeedup over numpy (size={n}):")
    baseline_time = times[0]  # numpy 的时间
    for name, time_ms in zip(names, times):
        speedup = baseline_time / time_ms
        print(f"  {name:<19}: {speedup:.4f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matrix multiplication benchmark")
    parser.add_argument(
        "--size", type=int, default=512, help="size of the matrices (default: 512)"
    )
    parser.add_argument(
        "--gap", type=int, default=10, help="time gap between methods (default: 10)"
    )
    parser.add_argument(
        "--matplotlib",
        action="store_true",
        default=False,
        help="use matplotlib to visualize result",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        default=False,
        help="add benchmark for pure Python implementation",
    )
    args = parser.parse_args()

    funcs: list[tuple[MatmulFunc, str]] = [
        (np.matmul, "numpy"),
        (matmul.trivial, "trivial"),
        (matmul.transpose_iter, "transpose loop iter"),
        (matmul.multithread, "multi-thread"),
        (matmul.chunk, "chunk"),
        (matmul.multithread_chunk, "chunk, multi-thread"),
        (matmul.transpose, "transpose matrix B"),
        (matmul.auto_simd, "SIMD (auto)"),
        (matmul.simd, "SIMD (manual)"),
        (matmul.simd_optimized, "SIMD (optimized)"),
        (matmul.multithread_simd, "SIMD, multi-thread"),
    ]
    if args.python:
        funcs.append((py_matmul, "pure Python"))

    if args.matplotlib:
        visualize_matplotlib(funcs, n=args.size, gap=args.gap)
    else:
        visualize_markdown(funcs, n=args.size, gap=args.gap)


if __name__ == "__main__":
    main()
