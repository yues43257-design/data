# A Checkable Certificate for Relative-Phase Toffoli Rewrites

**Draft v0 — 2026-10-20 — every number below is traceable to a script + JSON in this repo.**

Status: claims frozen as of commit `a245aaa`. Numbers marked ⚠️ need a citation or a
re-run before submission.

---

## Abstract

Replacing a Toffoli by a 4-T relative-phase gate saves three T gates, but the
replacement is only legal when the induced diagonal correction remains
unobservable. We give a **checkable certificate** for that precondition: the
tracker outputs the support set `S` and every future Hadamard check, so a reader
can verify `S ∩ H = ∅` directly; a one-line lemma then gives "residual is
diagonal", i.e. only phases change. The certificate is **sound but incomplete**
and costs `O(gates)` time / `O(qubits)` space. We validate the implementation
end-to-end (not just the lemma): 52 executable assertions, 2278 accepted masks
against the true residual support, 3456 adversarial masks, a 200-circuit tail
fuzz and an ancilla-layout fuzz. We then show how to *use* the certificate:
as a pre-filter with an exact fallback (best T saved vs cost), with a decision
table for four proposal budgets. The pass reduces T-count by **33–37%** versus
Qiskit L3 on H-free/low-H block-structured arithmetic, but only **14–26 T
(0.7–1.3%)** once Hadamards are interleaved.

---

## 1. Contributions (four, and only four)

1. **Certificate schema + verifier + a four-layer trust chain.**
2. **Implementation-fidelity acceptance** (52 / 2278 / 3456 / fuzz / Grover).
3. **Hybrid pre-filter + Pareto + decision table** (enum / one-shot / 2-per-group / greedy).
4. **Limitations**: ccx-block domain, H-free vs H-interleaved, standard benchmarks
   need a re-synthesis front-end.

Everything else (learned selection, the pricing law, window selection, H-migration)
is **appendix / limitations**, not a contribution.

---

## 2. Setting and the certificate

### 2.1 Declared domain

Circuits over `{X, Z, S, H, CX, C^nX}` with **Toffoli/MCX as block atoms**.
No ancilla, no mid-circuit measurement, no classical control. Anything else
raises `UnsupportedGateError` rather than being silently modelled.

### 2.2 Criterion

`LEGAL(candidate | ideal) ⟺ C = U_cand · U_ideal†` is diagonal with unit-modulus
entries (equivalently: Z-basis statistics of the data register are unchanged).

### 2.3 Lemma (the theorem)

Conjugating a diagonal operator by a basis permutation keeps it diagonal, and
diagonal gates commute with it. Hence the accumulated correction stays diagonal
**iff no Hadamard ever acts on a qubit in its support**. (One-line proof; the
tracker's support propagation is the algorithmic content, not the lemma.)

### 2.4 Certificate

`certify(schedule, mask)` returns

```
{ schema: "diagonal-certificate-v1",
  accepted, final_support: S,
  h_checks: [{stage, qubit, in_support, support_before}, ...],
  violations: [ ... ] }        # empty  <=>  accepted
```

A reader checks `S ∩ {H qubits} = ∅` directly. `verify_certificate` re-runs the
replay and compares every claim; tests show it rejects a forged `accepted=True`
or a cleared `h_checks`.

### 2.5 The four-layer trust chain (do not conflate)

| layer | what it guarantees | evidence |
|---|---|---|
| 1 | `S ∩ H = ∅` | the certificate's own record |
| 2 | disjoint ⇒ phase-only | the lemma |
| 3 | true support ⊆ `S` | 2278 masks + fuzz (§3) |
| 4 | certificate not tampered | `verify_certificate` — **re-execution, not an independent implementation** |

---

## 3. Implementation fidelity

| check | result |
|---|---|
| executable assertions (frozen CI) | **52 / 52 PASS** |
| accepted masks: true support ⊆ tracked support | **2278 / 2278** |
| adversarial masks (§5.23 + §5.31) | **3456**, 0 counterexamples |
| random tail fuzz, 200 circuits | old **844** unsound accepts → new **0** |
| ancilla-layout fuzz, 40 circuits | old **80** → new **0** |
| Grover 3-qubit | old tracker accepted 1 illegal mask; new rejects it |

The last two rows are the reason this section exists: the tail-layer bug
(`layers[-1]` never applied) was **found by external review**, not by our own
tests, because no generator ever put a Hadamard after the last block.

---

## 4. Using the certificate: hybrid + decision table

Pre-filter semantics: **accept → certificate, stop; reject → exact check only
for that mask**, up to a width threshold (default 11).

### 4.1 Quality (not recall) by proposal budget — homogeneous population

| budget | proposals | legal_in_prop | best_T | enum_best | /enum | tracker-only | exact_calls | exact_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all (enum) | 48 | 36 | **30** | 30 | 1.00 | 12 | 40 | 1.4 ms |
| 2/group | 6 | 5 | **27** | 30 | 0.90 | 12 | 3 | 0.1 ms |
| 1/group (one-shot) | 3 | 2 | 24 | 30 | 0.80 | 0 | 3 | 0.2 ms |
| greedy (tracker only) | 3 | 3 | 12 | 30 | 0.40 | 12 | 0 | 0 |

**Recall denominators differ by budget**; that is why the quality column is
`best_T / enum_best_T`.

### 4.2 Multi-seed check of 2-per-group (5 seeds)

`2/group ≥ one-shot` in **5/5** seeds (gaps 0, 3, 3, 6, 9; mean **+4.2 T**), at
0.07–0.32 ms. The **+3 T** seen on the first seed is **not** a stable magnitude —
report dominance, not a constant.

### 4.3 Decision table (the paper's "instructions")

| scenario | use | best T | cost |
|---|---|---:|---:|
| homogeneous, max quality | enum + hybrid | 30 | 1.7 ms |
| homogeneous, cheap | **2/group + hybrid** | ≥ one-shot (5/5) | 0.1–0.3 ms |
| homogeneous, zero exact | greedy | 12 | 0.1 ms |
| wide / H-interleaved, max | **greedy** | 6 | 0.1 ms |
| wide / H-interleaved, avoid | one-shot (0) / enum+exact (6) | 0 / 6 | 0.53 s / 7.9 s |

### 4.4 Against an exact checker

QCEC (`alternating`), deciding `C` diagonal via `[C, Z_q] = 0`:

| N | single query | full check legal | full check illegal | tracker |
|---:|---:|---:|---:|---:|
| 18 | 0.032 s | 0.80 s | 0.27 s | 0.047 ms |
| 56 | 0.057 s | 10.73 s | 5.44 s | ~0.2 ms |
| 106 | 0.080 s | **14.63 s** | **4.24 s** | **0.5 ms** |

⇒ the certificate is **~10⁴× faster** (range 8.5×10³–5×10⁴) for the whole
decision, and remains useful where exact checking is not interactive.
**Do not write** "exact cannot scale": it decides N=106.

---

## 5. Limitations

1. **ccx-block domain.** The rewrite unit is a Toffoli/MCX block. Circuits that
   already expose only `h/t/tdg/cx` (or `cu1`) have no block sites.
2. **H-free vs H-interleaved.** H-free: −33…−37% vs Qiskit L3. H-interleaved:
   **14–26 T (0.7–1.3%)** only.
3. **Standard benchmarks.** 5 real QASMBench circuits: **0/5 in domain** —
   pre-decomposed and containing `measure`. A re-synthesis front-end is required;
   this paper does not do it, and reports **no standard-benchmark performance**.
4. **Certificate is sound, not complete.** Rejected masks are not necessarily illegal;
   the hybrid closes that gap only up to N≈11 (12–18 is an optional slow path).

---

## 6. Security considerations (trust boundary)

The certificate is a statement about a **schedule**, not about the circuit that
will run. In the original API, `verify_certificate(schedule, mask, cert)`
re-executes the schedule and never looks at the emitted circuit. If the
adversary controls the compiler pass, it can keep the schedule and emit a
different circuit:

| attack | circuit legal | unbound cert | bound cert |
|---|---|---|---|
| block substitution (re-map a block) | no | **accepts** | rejects |
| non-block X inserted | no | **accepts** | rejects |
| tail Hadamard appended | no | **accepts** | rejects |
| smuggled `u3` / `measure` | no | rejects (`UnsupportedGateError`) | rejects |
| forged certificate | yes | rejects | rejects |
| replayed certificate (other mask) | yes | rejects | rejects |

**Defense.** The certificate now carries

* `circuit_hash` — canonical gate list of the emitted circuit,
* `schedule_hash` — canonical block/layer description,
* `block_hashes` — per-block decomposition hashes,

and `verify_certificate(..., circuit=..., block_ranges=...)` recomputes all three
and re-derives the witness. With the binding, **7/7 attacks are detected and 0
succeed**; the certificate grows from 147 B to 363 B and verification stays
sub-millisecond.

**Take-away.** A schedule-level certificate is only meaningful when the verifier
can bind it to the artifact it will execute. We therefore ship the binding as
part of the tool and recommend it whenever the compiler pass is untrusted.

---

## Appendix (pointers, not contributions)

- Learned selection: no gain at full budget (0.00 T); only +5–13 T with an exact
  verifier and a tight budget. Not the main line.
- Pricing law `FR = 4m + 3k`: holds on the controlled CDKM series (14/14),
  **fails** on two chained adders (k=0 but 20–32 T behind pyZX).
- Window selection: closed in that space (Δ≡0; floor `4m` is 20–32 T above pyZX),
  not a global impossibility.
- H-migration (b2): **OPEN** — the original "0 free-unlockable" was a vacuous
  predicate; no verdict is claimed.

---

## Reproduction

```
.venv/bin/python frozen_ci.py            # 4 steps, 155 s, one exit code
.venv/bin/python hybrid_check.py         # quality table + Pareto data
.venv/bin/python plot_pareto.py          # core figure
.venv/bin/python qcec_boundary.py --qcec-python /tmp/qcec-venv/bin/python
```
