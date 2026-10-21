# Compiler bug benchmark — L/S/C protocol (v3)

**Generated suite.** `generator.py` enumerates

    family × layout × seed × attack    (336)  +  5 chain-substitution + 20 chain-forgery  = 361

| axis | values |
|---|---|
| family | `two_block`, CDKM `adder2`/`adder3`, `vbe2`/`vbe3`, `comparator3` |
| layout | `plain`, `with_h` |
| seed | 0, 1, 2, 3 |
| attack | legal, h_on_support, block_substitution, tail_h, u3, cert_forge, cert_replay |
| chain | 5 substitution (detected) + 20 forged-hash (the structural gap) |

## Three questions, never merged

| question | metric | verifiers |
|---|---|---|
| **L** legality | **soundness** (no illegal accepted) and **completeness** (legal accepted) | dense oracle (N≤8), **QCEC Z-commutation (all cases)** |
| **S** semantics | exact equivalence up to global phase | dense `Operator`, **QCEC plain equivalence** |
| **C** certificate | named properties: `positive` / `illegal_rewrite` / `binding` / `tamper` / `chain` | bound certificate |

A legal relative-phase rewrite has `L = True, S = False` — the relaxation
changes the semantics by a diagonal, so the axes must stay separate.

## Scoring protocol

- each verifier answers exactly one question;
- `n/a` is an **abstention**, never a false positive;
- cross-verifier comparisons use the **common decided subset**;
- every rate has a Wilson 95% interval;
- `undecided_legal` marks cases whose L is not decidable densely; **QCEC
  resolves them** in this run (0 remain).

## Measured (361 cases, full QCEC)

| metric | value |
|---|---|
| **L / dense** | decided 240, coverage 0.665, soundness **1.000** [0.972, 1.000], completeness **1.000** [0.946, 1.000] |
| **L / QCEC (all)** | decided **336**, coverage **0.931**, soundness **1.000** [0.981, 1.000], completeness **1.000** [0.956, 1.000] |
| **S / QCEC vs dense** | common subset **279**, accuracy **1.000** [0.984, 1.000] |
| **C / positive** | 34 cases, **1.000** [0.879, 1.000] |
| **C / illegal_rewrite** | 62 cases, **1.000** [0.930, 1.000] |
| **C / binding** | 144 cases, **1.000** [0.969, 1.000] |
| **C / tamper** | 96 cases, **1.000** [0.954, 1.000] |
| **C / chain** | **25 cases, 5 correct** → **0.200** [0.084, 0.396] |
| cost | QCEC ≈ **0.20 s**/case (L), ≈ **0.18 s** (S); certificate ≈ **1.2 ms**; cert **≈ 533 B** |

The 25 chain cases are 5 substitutions (rejected by the hash chain) and 20
forgeries that write **correct hashes** for a wrong `C2` — the structural
attestation gap, now sampled 20 times instead of once.

## What is still missing (honest)

- **no compiler-version axis** (Qiskit only; pyZX/TODD are used in other probes);
- **no QASMBench/RevLib circuits** in this suite (the QASMBench domain census is
  in `reports/strategy_b_experiments.md` §A.1 — 4/42 enter the domain);
- only one two-pass chain shape; parameter sweeps over H density / block count /
  width are not yet systematic;
- the L/S verifiers agree on every decided case, so apart from the chain gap
  there is **no tool disagreement** found yet.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec   # ~150 s
.venv/bin/python -m pytest tests/test_compiler_bench.py -q
```
