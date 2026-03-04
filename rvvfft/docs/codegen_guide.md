# Code Generator Guide

The rvvfft **code generator** (`codegen/`) automatically produces optimised RVV butterfly kernel C source files from a high-level description.  This eliminates the need to hand-write dozens of near-identical kernel variants for different radix / dtype / VLEN / LMUL / layout combinations.

---

## Architecture

```
codegen/
├── dag.py            DFT computation DAG — radix-2/4/8 butterfly builders
├── simplify.py       Symbolic simplification pass (twiddle elimination)
├── verify_math.py    SymPy-based mathematical verification
├── generator.py      Main script: DAG → Jinja2 → .c files
└── templates/
    ├── butterfly.c.j2    RVV intrinsic kernel template
    └── twiddle.c.j2      Twiddle table initialisation template
```

### Processing Pipeline

```
       Radix, N, group
           │
           ▼
    dag.py: build_radix{2,4,8}_butterfly()
    ─ constructs DFT computation DAG ─
           │
           ▼
    simplify.py: simplify_dag()
    ─ eliminates trivial twiddle multiplies ─
    ─ verified by SymPy (verify_math.py)   ─
           │
           ▼
    generator.py: build_template_context()
    ─ maps DAG + config → Jinja2 variables ─
           │
           ▼
    Jinja2: butterfly.c.j2
    ─ renders RVV intrinsic C code ─
           │
           ▼
    src/kernel/generated/<func_name>.c
```

---

## Running the Generator

```bash
cd rvvfft/codegen

# Generate all default kernel variants (radix 2/4/8, f32/f64, VLEN 256/1024, both layouts)
python generator.py

# Run mathematical verification first, then generate
python generator.py --verify

# Generate only radix-2 f32 kernels for VLEN=256
python generator.py --radix 2 --dtype f32 --vlen 256 --layout split interleaved

# Dry run: print what would be generated without writing files
python generator.py --dry-run

# Generate and write registry to custom location
python generator.py --registry /path/to/codelet_registry.h
```

### Via CMake

```bash
cmake -B build
cmake --build build --target rvvfft_codegen          # generate kernel sources
cmake --build build --target rvvfft_codegen_verify   # verify math rules only
```

---

## Configuration Parameters

| Parameter | Values | Description |
|---|---|---|
| `--radix` | 2, 4, 8, 16, 32 | Butterfly radix |
| `--dtype` | f32, f64 | Scalar data type (SEW=32 or 64) |
| `--vlen` | 256, 1024 | Hardware VLEN in bits |
| `--lmul` | 1, 2, 4, 8 | RVV LMUL (auto-selected if omitted) |
| `--layout` | interleaved, split | Input/output data layout |

**LMUL auto-selection:** the generator picks the largest LMUL that keeps register pressure within 32 vector registers (the RVV architectural limit).  For radix-8 this is typically LMUL=2; for radix-2 it can be LMUL=8.

---

## Generated Kernel Naming Convention

```
rvvfft_butterfly_r{radix}_{dtype}_vlen{vlen}_m{lmul}_{layout[:5]}
```

Examples:
- `rvvfft_butterfly_r2_f32_vlen256_m8_split`
- `rvvfft_butterfly_r4_f32_vlen1024_m2_inter`
- `rvvfft_butterfly_r8_f64_vlen1024_m1_split`

---

## Mathematical Verification

Every twiddle-factor simplification rule in `simplify.py` is verified by `verify_math.py` using **SymPy**:

```
python verify_math.py
```

Verification covers:
- W(0,N) = 1 (identity — no multiply needed)
- W(N/2,N) = -1 (negate — no multiply needed)
- W(N/4,N) = -j (pure imaginary rotation — swap + negate, no FMA needed)
- W(3N/4,N) = +j (conjugate rotation)
- Complex multiply formula correctness
- Radix-4 butterfly numerical correctness (4-point DFT agreement)
- FLOP count formula: 5 × N × log₂(N)
- Algebraic identities: x×1=x, x×0=0, Neg(Neg(x))=x, etc.

All tests must pass before any generated kernel is merged into the repository.

---

## Adding a New Radix

1. Implement `build_radixN_butterfly(group, n)` in `dag.py`.
2. Add an entry in `_build_dag()` in `generator.py`.
3. Add a template block for `radix == N` in `butterfly.c.j2`.
4. Add correctness tests in `verify_math.py`.
5. Run `python verify_math.py` — all tests must pass.
6. Run `python generator.py --radix N --dry-run` to confirm output.

---

## Adding a New Instruction Backend

The generator is designed with a pluggable instruction backend for future AI-extension instructions:

```python
# In generator.py build_template_context():
ctx["instruction_backend"] = "rvv_standard"   # default
# Future: ctx["instruction_backend"] = "rvv_ai_ext"
```

When your chip gains fused complex-multiply instructions:
1. Add a new template variant (`butterfly_ai.c.j2`) using the new intrinsics.
2. Add `--backend rvv_ai_ext` CLI option to `generator.py`.
3. The planner automatically selects the right codelet at runtime via the registry.

---

## Codelet Registry

After generation, `src/kernel/generated/codelet_registry.h` enumerates all kernels:

```c
#include "codelet_registry.h"

// Access the kernel table
for (const rvvfft_codelet_entry_t *e = rvvfft_codelet_table; e->fn != NULL; ++e) {
    printf("radix=%d dtype=%s vlen=%d lmul=%d layout=%s\n",
           e->radix, e->dtype, e->vlen, e->lmul, e->layout);
}
```

The planner uses this table to select the best codelet for each FFT stage given the hardware VLEN.

---

## Dependencies

| Package | Purpose | Required? |
|---|---|---|
| Python ≥ 3.9 | Generator runtime | Yes |
| Jinja2 | Template rendering | Yes (`pip install jinja2`) |
| SymPy | Math verification | Yes (`pip install sympy`) |
| NumPy | Golden data generation | Tests only (`pip install numpy`) |

Install all at once:
```bash
pip install jinja2 sympy numpy
```
