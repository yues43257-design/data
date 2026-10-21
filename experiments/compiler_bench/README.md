# Compiler bug benchmark — frozen protocol v1

`protocol.py` freezes the definitions:

    PROTOCOL_VERSION  = compiler-bench-protocol-v1
    protocol_fingerprint = 16-hex hash of {L, S, C definitions, DENSE_LIMIT}

Every `results.json` carries both. **Classification is frozen; bump the version
to change it.**

## Suite (400 cases, generated)

| axis | values |
|---|---|
| family | `two_block`, CDKM `adder2`/`adder3`, `vbe2`/`vbe3`, `comparator3` |
| layout | `plain`, `with_h` |
| seed | 0–3 |
| attack | legal, h_on_support, block_substitution, tail_h, u3, cert_forge, cert_replay |
| D1 sweep | `hh_restore` with blocks ∈ {1,2,3} × H-pairs ∈ {1,2,4} (9 cases) |
| chain | 10 substitution + 45 forged-hash = 55 |

## Three questions (frozen)

| question | metric | verifiers |
|---|---|---|
| **L** | soundness / completeness | dense oracle (N≤8), **QCEC Z-commutation (all)** |
| **S** | exact equivalence | dense `Operator`, **QCEC plain equivalence** |
| **C** | positive / illegal_rewrite / binding / tamper / chain | bound certificate |

`n/a` = abstention (not a false positive); comparisons use the common decided
subset; every rate has a Wilson 95% interval.

## Measured (400 cases, full QCEC)

| metric | value |
|---|---|
| **L / dense** | decided 249, coverage 0.623, soundness **1.000** [0.972, 1.000], completeness **1.000** [0.951, 1.000] |
| **L / QCEC** | decided **345**, coverage **0.863**, soundness **1.000** [0.981, 1.000], completeness **1.000** [0.960, 1.000] |
| **S / QCEC vs dense** | common **286**, accuracy **1.000** [0.984, 1.000] |
| **C / positive** | 43 cases, 34 correct → **0.791** [0.646, 0.888] |
| C / illegal_rewrite | 62, **1.000** [0.930, 1.000] |
| C / binding | 144, **1.000** [0.969, 1.000] |
| C / tamper | 96, **1.000** [0.954, 1.000] |
| **C / chain (split)** | substitution **10/10 = 1.000**; **forgery 0/45 = 0.000** [0.000, 0.094] |
| cost | QCEC ≈ 0.19 s (L) / 0.17 s (S); certificate ≈ 1.35 ms; **543 B** |

## D1 — certificate false negatives, reproduced across parameters

`positive_by_family`:

| family | positive cases | accepted | rate |
|---|---:|---:|---:|
| two_block | 4 | 4 | 1.000 |
| adder | 10 | 10 | 1.000 |
| vbe | 13 | 13 | 1.000 |
| comparator | 7 | 7 | 1.000 |
| **hh_restore** | **9** | **0** | **0.000** |

Mechanism: `H … H` after a downgraded block conjugates the correction back to
diagonal, so **dense and QCEC say legal**; the tracker rejects at the first H.
The sweep (blocks 1–3 × H-pairs 1,2,4) shows this is **not a single-instance
artifact** — all 9 reproduce, while every other family is 1.000.

## D2 — chain forgeries

45 forged-hash cases (wrong `C2`, correct hashes) are **accepted** (0/45
detected), while all 10 substitution cases are rejected. The attestation gap is
therefore not a counting artifact of one combined rate.

## Not done (honest)

- compiler-version axis (Qiskit/pyZX/TODD variants) absent;
- QASMBench/RevLib not in this suite (domain census: 4/42 in report §A.1);
- parameter sweep covers blocks × H-pairs; H density / width sweeps not
  systematic.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec   # ~150 s
.venv/bin/python -m pytest tests/test_compiler_bench.py -q              # 10 tests
```
