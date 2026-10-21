# Compiler bug benchmark — frozen protocol v1

`protocol.py` freezes `PROTOCOL_VERSION = compiler-bench-protocol-v1` and a
16-hex `protocol_fingerprint()` over the L/S/C definitions and `DENSE_LIMIT`.
Every `results.json` carries both. Change the classification ⇒ bump the version.

## Suite (416 cases)

| axis | values |
|---|---|
| family | `two_block`, CDKM `adder2`/`adder3`, `vbe2`/`vbe3`, `comparator3` |
| layout | `plain`, `with_h` |
| seed | 0–3 |
| attack | legal, h_on_support, block_substitution, tail_h, u3, cert_forge, cert_replay |
| D1 sweep | `hh_restore`: blocks {1,2,3} × H-pairs {1,2,4} = 9 |
| **real circuits** | **QASMBench in-domain 4** (`fredkin_n3`, `sat_n7`, `simon_n6`, `toffoli_n3`) × 4 attacks = 16 |
| chain | 10 substitution + 45 forged-hash = 55 |

## Three questions (frozen)

| question | metric | verifiers |
|---|---|---|
| **L** | soundness / completeness | dense (N≤8), QCEC Z-commutation (all) |
| **S** | exact equivalence | dense `Operator`, QCEC plain equivalence |
| **C** | positive / illegal_rewrite / binding / tamper / chain | bound certificate |

`n/a` = abstention; comparisons use the common decided subset; Wilson 95% CI.

## Measured (416 cases, full QCEC)

| metric | value |
|---|---|
| **L / dense** | decided 249, coverage 0.599, soundness **1.000** [0.972, 1.000], completeness **1.000** [0.951, 1.000] |
| **L / QCEC** | decided **361**, coverage **0.868**, soundness **1.000** [0.981, 1.000], completeness **1.000** [0.962, 1.000] |
| **S / QCEC vs dense** | common **285**, accuracy **1.000** [0.984, 1.000] |
| **C / positive** | 45 cases, 36 correct → **0.800** [0.660, 0.893] |
| C / illegal_rewrite | 64, **1.000** [0.932, 1.000] |
| C / binding | 148, **1.000** [0.970, 1.000] |
| C / tamper | 104, **1.000** [0.957, 1.000] |
| **C / chain split** | substitution **10/10 = 1.000**; forgery **0/45 = 0.000** [0.000, 0.094] |
| cost | QCEC ≈ 0.19 s (L) / 0.17 s (S); cert ≈ 1.39 ms; **554 B** |

## D1 — certificate false negatives, and its scope

| family | positive cases | accepted | rate |
|---|---:|---:|---:|
| two_block | 4 | 4 | 1.000 |
| adder | 10 | 10 | 1.000 |
| vbe | 13 | 13 | 1.000 |
| comparator | 7 | 7 | 1.000 |
| **hh_restore** | **9** | **0** | **0.000** |
| **real_qasmbench** | **2** | **2** | **1.000** |

**Reading.** D1 reproduces across all 9 `hh_restore` parameter combos
(blocks × H-pairs) but **does not reproduce on the 4 real in-domain QASMBench
circuits** (2/2 accepted). So the certificate false negative is
**mechanism-specific** (an H hit that is later conjugated back), not a general
property of the tracker — an important scoping result, not a stronger claim.

## D2 — chain forgeries

45 forged-hash cases (wrong `C2`, correct hashes) are **all accepted** (0/45
detected); the 10 substitution cases are all rejected. D2 is structural and
independent of the circuit family.

**Framing correction (frozen).** D2 is a *known trust-root limitation* of hash
chains, not a new vulnerability. The contribution of this work is the
**protocol + suite + measurement**, not the discovery of D2. D2 is used as a
calibration point that the protocol can see.

## H-density / width sweep (`h_density_sweep.py`)

Every mask of every `WideInstance` cell, dense oracle, bound certificate:

| h | masks | legal | cert-accepted | **FN** | **UNSOUND** | FN/legal |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 112 | 112 | 112 | **0** | 0 | 0.000 |
| **0.1** | 112 | 50 | 46 | **4** | 0 | **0.080** |
| 0.3 | 112 | 32 | 32 | 0 | 0 | 0.000 |
| 0.5 | 112 | 28 | 28 | 0 | 0 | 0.000 |

Reading: certificate false negatives are **H-density specific** — they appear
only at low H density (8% of legal masks at h=0.1), and the tracker is **sound
everywhere** (0 illegal accepts). This bounds D1 to a narrow regime rather than
a general claim.

## Not done

- compiler-version axis (Qiskit/pyZX/TODD variants);
- RevLib (QASMBench in-domain 4 are now included);
- width sweep beyond N=8 (dense limit; QCEC covers larger N in the main suite).

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec   # ~155 s
.venv/bin/python -m pytest tests/test_compiler_bench.py -q              # 11 tests
```
