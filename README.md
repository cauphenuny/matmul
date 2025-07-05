# Matmul

> Assignment for assembly course in UCAS

## Build

```
xmake # depends on uv the python package manager
```

## Run

```
uv run benchmark/main.py
```

---

## Benchmark

($N=1024$)

| Method              | Time(ms) | Speedup over `np.matmul` |
|---------------------|----------|--------------------------|
| numpy               | 974.9938 |   1.0000x                |
| trivial             | 852.5763 |   1.1436x                |
| multi-thread        | 221.4695 |   4.4024x                |
| chunk               | 236.0463 |   4.1305x                |
| chunk, multi-thread |  91.8529 |  10.6147x                |
| SIMD (auto)         |  59.2121 |  16.4661x                |
| SIMD (manual)       |  55.7972 |  17.4739x                |
| SIMD, multi-thread  |  15.1485 |  64.3623x                |
