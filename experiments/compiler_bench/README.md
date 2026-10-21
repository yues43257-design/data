# Compiler correctness / safety benchmark (scoping probe)

**Question.** Can we build ≥10 reproducible cases with ground truth where
`dense exact`, `MQT QCEC`, and the **bound certificate** produce *distinguishable*
results — and is there a verifier gap worth a benchmark/evaluation paper?

## Cases (11) and verifier outputs

`L` = expected legality of the emitted circuit, `C` = expected certificate
verdict. `–` = verifier not applicable (non-unitary, width, or a different
question).

| case | expected | dense | QCEC | certificate | ground truth |
|---|---|---|---|---|---|
| legal_control | L=T C=T | T | T | T | none |
| illegal_rtof_mask | L=F C=F | F | F | F | layer 1: H(2) hits block 0 support |
| block_substitution | L=F C=F | F | F | F | block 0 remapped `[0,1,2]→[0,2,1]` |
| nonblock_x | L=F C=F | F | F | F | layer 1: X(3) |
| tail_h | L=F C=F | F | F | F | after block 1: H(2) |
| gate_smuggle_u3 | L=F C=F | – | F | F | layer 1: `u3` |
| gate_smuggle_measure | L=F C=F | – | – | F | after block 1: `measure` |
| cert_forge | L=T C=F | T | T | **F** | cert `accepted`/`final_support` tampered |
| cert_replay | L=T C=F | T | T | **F** | mask-1 cert replayed for mask 3 |
| pass2_substitution | C=F | – | – | F | C2 gets X(0) after full_reduce |
| **pass2_forgery** | C=F | – | – | **T (miss)** | C2 gets X(0) + forged cert2 with correct hashes |

## Metrics (measured)

| verifier | applicable | correct | rate | false positives |
|---|---:|---:|---:|---:|
| dense exact (N ≤ 10) | 7 | 7 | **1.000** | 0 |
| MQT QCEC (Z-commutation) | 8 | 8 | **1.000** | 0 |
| bound certificate | 11 | 10 | **0.909** | 0 |

Runtime: dense ≤ 10 ms, QCEC ≈ 0.16–0.24 s, certificate **0.2–0.5 ms**
(pass2 chain check ≈ 4.7 ms). Certificate size: 363 B (RP cases).

## The gap

- **dense / QCEC** answer "is the emitted circuit legal?" completely (on these
  widths) but are blind to certificate tampering (`cert_forge`, `cert_replay`).
- **certificate** is fast and catches every circuit substitution and every
  certificate tampering, but **cannot catch a compromised pass that writes
  correct hashes for a wrong circuit** (`pass2_forgery`).
- So the trade-off is real: *fast but incomplete* vs *slow but complete on
  legality*, plus different blind spots (cert validity vs semantics).

## GO / NO-GO

**GO (scoping level).** 11 reproducible cases, all with an explicit expected
verdict and a named ground-truth location, and at least one measurable verifier
gap (`pass2_forgery`; plus dense/QCEC blindness to certificate tampering).

**Caveats before calling it a paper:** every case is built on the *same* 5-qubit
two-block base; there is no diversity across widths, gate sets or compiler
versions; 11 cases is a seed set, not a suite. A benchmark paper would need
many more families, a threat model, and a scoring protocol.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py
.venv/bin/python -m pytest tests/test_compiler_bench.py -q
```
