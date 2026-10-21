# Compiler correctness / safety benchmark suite

**Generated, not hand-written.** `generator.py` enumerates

    family × n × layout × attack      (66 cases)

| axis | values |
|---|---|
| family | `two_block`, CDKM `adder2`/`adder3`, `comparator3` |
| width n | 2, 3 (adders/comparator); fixed 5-qubit two-block base |
| layout | `plain`, `with_h` (H on a supported qubit → makes the all-relative mask illegal) |
| attack | legal, block_substitution, nonblock_x, tail_h, gate_smuggle_u3, gate_smuggle_measure, cert_forge, cert_replay (+ 2 two-pass cases) |
| certificate version | **verifier under test**, not a case axis: `v1` unbound vs `v2` bound |

## Three ground truths per case

| truth | meaning | oracle |
|---|---|---|
| **L** legality | residual is diagonal (relative-phase criterion) | dense `relative_phase_ok` (N ≤ 10) |
| **S** semantics | exactly equal to the intended circuit (global phase) | dense `Operator` comparison (N ≤ 10) |
| **C** certificate | a correct certificate verifier should accept | normative: `attack == legal and L == True` |

Note the deliberate split: a legal relative-phase rewrite has **L = True but
S = False** (it differs by a diagonal) — that is the whole point of the
relaxation, and it is why "legal" and "semantically identical" must be scored
separately.

## Scoring protocol

Each verifier answers **exactly one** question:

| verifier | axis | notes |
|---|---|---|
| dense | L | oracle |
| MQT QCEC (Z-commutation) | L | optional subset (`--with-qcec --qcec-limit`) |
| `cert_v1` (unbound) | C | core fields only — the pre-fix semantics |
| `cert_v2` (bound) | C | circuit/schedule/block hashes + witness |

`n/a` (non-unitary circuit, width out of range, or wrong question) is an
**abstention**, never a false positive. Accuracy is computed over decided
answers only.

## Measured (66 cases, QCEC on the first 24)

| verifier | decided | abstain | correct | wrong | accuracy |
|---|---:|---:|---:|---:|---:|
| dense | 48 | 18 | 48 | 0 | **1.000** |
| QCEC | 16 | 50 | 16 | 0 | **1.000** |
| cert_v1 (unbound) | 64 | 2 | 44 | **20** | **0.688** |
| **cert_v2 (bound)** | 66 | 0 | 65 | **1** | **0.985** |

### The one remaining miss

`pass2/pass2_forgery`: a compromised pass2 emits a wrong `C2` but writes
**correct hashes** for it, so the hash chain is self-consistent. This is the
attestation gap measured in `experiments/multipass_attestation/`.

### Reading

- `cert_v1` (schedule-only) accepts **20** circuit attacks — the measurable cost
  of not binding the certificate to the circuit.
- `cert_v2` (hash binding) removes all 20 and leaves exactly one structural
  gap (compromised pass with correct hashes).
- dense/QCEC are complete on L but answer a different question; they are blind
  to certificate tampering (`cert_forge`, `cert_replay`), which the protocol
  does **not** count as a false positive.

## GO / NO-GO

**GO at scoping level**: 66 generated cases, three explicit ground truths each,
and a measurable verifier gap (`cert_v1` 20 misses; `cert_v2` exactly 1).

**Before this is a paper:** the suite still rests on a small number of families
(two-block, CDKM adder, comparator), N ≤ 10, one pass chain, and one attack per
type. It needs more families, widths, gate sets and compiler versions; the
scoring protocol above can be reused as-is.

## Reproduce

```bash
.venv/bin/python experiments/compiler_bench/run_bench.py --with-qcec --qcec-limit 24
.venv/bin/python -m pytest tests/test_compiler_bench.py -q
```
