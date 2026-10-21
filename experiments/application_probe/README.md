# Application probe (scoping, 1–2 days): relative-phase rewrites on crypto/arithmetic circuits

**Question.** Is there a crypto/arithmetic family where the relative-phase
rewrite gives a *new, publishable* resource result (T-count / T-depth / ancilla
/ spacetime) that the first paper (adders, ladders, Grover) does not already
cover?

## Families and results

`relT` = T-count of the all-relative-phase circuit (4 T per block);
`FR` = pyZX `full_reduce`; `L3` = Qiskit `optimization_level=3` in the
Clifford+T basis. Legality: dense oracle for N ≤ 11, tracker otherwise.

| family | q | blocks | in domain | legality | relT | relT-depth | FR | FR T-depth | L3 |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| CDKM adder 3 | 8 | 6 | yes | legal | 24 | 24 | **24** | **9** | 36 |
| **VBE adder 3** | 10 | 10 | yes | legal | **40** | 26 | **24** | **9** | 56 |
| comparator 3 | 6 | 3 | yes | legal | **12** | 10 | 15 | **6** | 19 |
| modular adder 3 | 6 | 0 | **no** | n/a | – | – | 16 | 7 | 16 |
| Draper / QFT / multiplier | – | – | **no** | – | – | – | – | – | – |
| modpow / Shor-like | – | – | **not constructed** | – | – | – | – | – | – |

## GO / NO-GO

**NO-GO.**

- The only **new** in-domain family is the VBE adder, and it is **negative**:
  relative-phase needs **40 T** vs FR's **24 T**, and its T-depth is **26 vs 9**.
- CDKM ties FR (24 = 24) — already in the first paper.
- Comparator wins by **3 T** (12 vs 15) — already the first paper's "+3 T"
  result, and FR still has the better T-depth (6 vs 10).
- **No family shows a T-depth or ancilla advantage**; FR dominates T-depth in
  every in-domain case.

## Why some families are missing (not forced)

- **Draper adder / QFT / HRSCumulativeMultiplier**: they contain `cp(θ)` with
  angles finer than π/4 (e.g. π/8). Such angles are **not exactly representable
  in Clifford+T**, so an exact T-count does not exist without an extra
  approximation/synthesis step. The probe records the transpiler error instead
  of forcing a number.
- **Modular adder**: the library gate decomposes with **0 `ccx` blocks**, so the
  block-level tracker has nothing to certify.
- **modpow / Shor-like**: no library primitive; composing a controlled modular
  multiplier is algorithm work, outside this 1–2 day scope.

## Safety proxy (crude, not a security estimate)

`physical_qubits = qubits · 2·d²` with `d = 15`, and
`spacetime = T-count · physical_qubits` (qubit-cycles).
This ignores routing, magic-state distillation and code surgery; it is only a
labelled proxy for comparing families, not a security claim.

## Reproduce

```bash
.venv/bin/python experiments/application_probe/run_probe.py
.venv/bin/python -m pytest tests/test_application_probe.py -q
```
