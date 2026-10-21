# Compiler bug benchmark — frozen protocol v1

`protocol.py` freezes the definitions; every result carries
`protocol_version = compiler-bench-protocol-v1` and a 16-hex fingerprint.
**Do not change the classification without bumping the version.**

## Suite (447 cases, generated)

| axis | values |
|---|---|
| family | `two_block`, **`hh_restore`**, CDKM `adder2`/`adder3`, `vbe2`/`vbe3`, `comparator3` |
| layout | `plain`, `with_h` |
| seed | 0–3 |
| attack | legal, h_on_support, block_substitution, tail_h, u3, cert_forge, cert_replay |
| chain | 10 substitution + 45 forged-hash = **55** |

## Three questions (frozen)

| question | metric | verifiers |
|---|---|---|
| **L** | soundness / completeness | dense oracle (N≤8), **QCEC Z-commutation (all)** |
| **S** | exact equivalence | dense `Operator`, **QCEC plain equivalence** |
| **C** | properties: positive / illegal_rewrite / binding / tamper / chain | bound certificate |

`n/a` is an abstention, never a false positive; comparisons use the common
decided subset; every rate carries a Wilson 95% interval.

## Measured (447 cases, full QCEC)

| metric | value |
|---|---|
| **L / dense** | decided 288, coverage 0.644, soundness **1.000** [0.975, 1.000], completeness **1.000** [0.958, 1.000] |
| **L / QCEC (all)** | decided **392**, coverage **0.877**, soundness **1.000** [0.983, 1.000], completeness **1.000** [0.964, 1.000] |
| **S / QCEC vs dense** | common **334**, accuracy **1.000** [0.986, 1.000] |
| **C / positive** | 42 cases, **34 correct → 0.810** [0.664, 0.903] |
| C / illegal_rewrite | 70, **1.000** [0.938, 1.000] |
| C / binding | 168, **1.000** [0.973, 1.000] |
| C / tamper | 112, **1.000** [0.960, 1.000] |
| **C / chain** | **55 cases, 10 correct → 0.182** [0.100, 0.305] |
| cost | QCEC ≈ 0.20 s (L) / 0.17 s (S); certificate ≈ 1.25 ms; **536 B** |

## Two tool divergences (the main finding)

### D1 — certificate false negatives (8/42 positives)

Family `hh_restore`: the circuit contains `H … H` after a downgraded block. The
two Hadamards conjugate the correction back to diagonal, so **dense and QCEC say
legal**, but the tracker rejects at the **first** H and never sees the second.
This is the same conservativeness measured in the main report (§5.62): the
certificate is sound but **incomplete**, and the missing recall is a *tool
divergence*, not noise.

### D2 — chain forgeries (45/55 chain cases wrong)

A compromised pass2 emits a wrong `C2` but writes **correct hashes**; the hash
chain is self-consistent and accepts it (10 substitution cases are correctly
rejected). This is the attestation gap from the multipass probe.

These are the two divergences the benchmark is built to expose; every other
category is at 1.000.

## Honest gaps (not done)

- **compiler-version axis** (Qiskit/pyZX/TODD variants) absent;
- **QASMBench/RevLib** not in this suite (QASMBench domain census: 4/42 in
  `reports/strategy_b_experiments.md` §A.1);
- parameter sweeps limited to 4 seeds; H density / block count / width sweeps
  are not systematic yet.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec   # ~165 s
.venv/bin/python -m pytest tests/test_compiler_bench.py -q              # 8 tests
```
