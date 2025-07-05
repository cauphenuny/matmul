# Matmul

> Assignment for assembly course in UCAS

## Build

```
uv venv .venv
uv sync
xmake
```

## Run

```
uv run benchmark/main.py
```

---

## Benchmark

($N=512$)

| Method              | Time(ms) | Speedup over `np.matmul` |
|---------------------|----------|--------------------------|
| trivial             | 117.2691 |   0.7026x          |
| multi-thread        |  17.5354 |   4.7365x          |
| chunk               |  23.7355 |   3.5019x          |
| SIMD (auto)         |   7.5846 |  11.1124x          |
| multi-thread SIMD   |   1.8076 |  46.3657x          |