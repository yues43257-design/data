# Compiler bug benchmark — scoring protocol v3

**Generated suite, three separate questions.** `generator.py` enumerates

    family × layout × seed × attack   → 194 cases (+2 two-pass)

| axis | values |
|---|---|
| family | `two_block`, CDKM `adder2`/`adder3`, `vbe2`/`vbe3`, `comparator3` |
| layout | `plain`, `with_h` |
| seed | 0, 1 (changes the inter-block CNOT / the H position) |
| attack | legal, block_substitution, nonblock_x, tail_h, u3, measure, cert_forge, cert_replay |

## The three questions (never merged)

### L — legality (soundness / completeness)

`truth_L` = the residual is diagonal (dense `relative_phase_ok`, N ≤ 8).
Dense is the oracle; QCEC (Z-commutation) is the independent L verifier on a
subset.

### S — semantics (exact equivalence)

`truth_S` = the emitted circuit equals the intended one up to **global phase**
(dense `Operator`, N ≤ 8). QCEC (plain equivalence) is the S verifier.

> A legal relative-phase rewrite is `L = True, S = False` — that is the whole
> relaxation, and why the two axes must be scored separately.

### C — certificate (four named properties)

| property | expected | meaning |
|---|---|---|
| `positive` | accept | a genuine certificate for a legal rewrite |
| `illegal_rewrite` | reject | the rewrite is illegal (L = False) |
| `binding` | reject | the circuit changed (block/X/tail H/u3/measure) |
| `tamper` | reject | forged or replayed certificate |
| `chain` | reject | wrong C2 in a two-pass chain |

`undecided_legal` cases (N > dense limit) are **not scored** — they are
reported separately.

## Scoring protocol

- each verifier answers exactly one question;
- `n/a` is an **abstention**, never a false positive;
- QCEC and certificate are compared on the **common decided subset**, not on
  different denominators;
- every rate comes with a Wilson 95% interval.

## Measured (194 cases; QCEC on the first 48)

| metric | value |
|---|---|
| **L / dense** (oracle) | decided 120, coverage **0.619**, soundness **1.000** [0.939, 1.000], completeness **1.000** [0.912, 1.000] |
| **L / QCEC** | decided 32, coverage **0.165**, soundness **1.000** [0.810, 1.000], completeness **1.000** [0.718, 1.000] |
| **S / QCEC vs dense** | common subset **41**, accuracy **1.000** [0.898, 1.000] |
| **C / positive** | 16 cases, **1.000** [0.773, 1.000] |
| **C / illegal_rewrite** | 4 cases, **1.000** [0.454, 1.000] |
| **C / binding** | 120 cases, **1.000** [0.963, 1.000] |
| **C / tamper** | 48 cases, **1.000** [0.912, 1.000] |
| **C / chain** | 2 cases, **1 of 2** [0.095, 0.905] |
| cost | QCEC ≈ 0.17 s, certificate **≈ 0.61 ms**, cert size **≈ 528 B** |

## The one structural gap

`pass2/pass2_forgery`: a compromised pass2 emits a wrong `C2` but writes
**correct hashes**, so the hash chain is self-consistent (C/chain 1/2).

## Coverage and honest caveats

- dense is limited to **N ≤ 8** (runtime); the 4 `vbe3` legal cases are
  `undecided_legal` and excluded from C scoring;
- QCEC runs on a **48-case subset** (coverage 0.165) because it costs ~0.17 s
  per case; the common-subset rule prevents comparing unlike denominators;
- only one pass chain and one attack parameter per type;
- no compiler-version axis yet (Qiskit only; pyZX/TODD are used elsewhere).

## GO / NO-GO

**GO at scoping level**: 194 generated cases, three orthogonal ground truths,
per-property certificate scoring, common-subset comparisons, and a measurable
gap (chain 1/2).

**Not yet a paper**: the family/width/gate-set coverage is still small, and the
compiler-version axis is missing. The protocol is reusable as-is.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec --qcec-limit 48
.venv/bin/python -m pytest tests/test_compiler_bench.py -q
```
