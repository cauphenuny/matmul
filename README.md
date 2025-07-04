# Matmul

> Assignment for assembly course in UCAS

Build

```
xmake
uv venv .venv
uv sync
```

Run

```
uv run benchmark/main.py
```

---

|Method|Result ($N=512$)|Compare to `numpy.matmul`|
|----|----|----|
|trivial|120.711ms|0.70|
|multi-thread|17.190ms|4.94|
|chunk|9.041ms|9.19|
|SIMD|6.984ms|11.88|
|multi-thread, SIMD|1.791ms|48.58|