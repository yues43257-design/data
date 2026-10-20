# Cross-block phase-merge probe (independent, not a paper claim)

**Question.** On 2–5 block, H-interleaved, N ≤ 10 circuits, can merging a
relative-phase block's residual into a later block's phase polynomial beat the
"7 T or 4 T per block" selection?

## Method (what was actually run)

1. Small instances: `WideInstance` with m = 2…5 blocks, 2–4 controls, N ≤ 10,
   `h ∈ {0, 0.3, 0.7}`, 2 seeds → 42 instances.
2. Every mask (2^m) is classified by the **dense** oracle
   `relative_phase_ok` (exact, no tracker).
3. Baselines: all-exact (7m), all-relative (4m), tracker greedy, pyZX
   `full_reduce`, TODD (`phase_block_optimize`).
4. "Merge" = segment-wise phase-polynomial re-synthesis of the legal target
   circuit, using the repo's existing TODD entry point
   (`realworld_benchmark.todd_t_count`). We did **not** write a separate
   polynomial merger; TODD is the strongest available proxy for "merge the
   terms and re-synthesise".

The target mask is the all-relative mask when it is dense-legal, otherwise the
best dense-legal mask. `full_reduce`/TODD preserve the unitary, so the merged
circuit is legal exactly when the target is.

## Result

| quantity | value |
|---|---:|
| instances | 42 |
| all-relative mask legal | 15 / 42 |
| **merge (TODD) < 4m** | **4 / 42** |
| **full_reduce ≤ TODD** | **42 / 42** |
| TODD ≥ 4m | 38 / 42 |

### The four gains

| N | m | h | 4m | FR | TODD | all-relative legal |
|---:|---:|---:|---:|---:|---:|---|
| 6 | 2 | 0.0 | 8 | **4** | 4 | yes |
| 6 | 3 | 0.0 | 12 | **4** | 4 | yes |
| 6 | 3 | 0.0 | 12 | **4** | 4 | yes |
| 6 | 3 | 0.3 | 12 | **7** | 7 | **no** (target = best legal mask, T=15) |

## Reading (negative)

1. **Gains exist, but only at N = 6.** At N ≥ 7 both `full_reduce` and TODD are
   *worse* than 4m (38/42): on tiny circuits with ancillas the pyZX extractor
   returns a larger circuit than the input. This is an extraction artefact, not
   a property of the merge idea.
2. **`full_reduce` dominates TODD in 42/42.** Whatever the four gains are, the
   existing strong baseline already captures them; there is no evidence that a
   *distinct* cross-block merger beats the current toolchain.
3. The one H-interleaved gain (7 < 12, with all-relative illegal) is real but is
   again matched by `full_reduce`. It does not establish a new mechanism.

**Verdict: not worth continuing as a separate research line.** A genuine
cross-block optimizer would have to beat `full_reduce`, not just 4m — and this
probe gives no such evidence. (A phase-polynomial *tracker* for recall is a
different goal; see the main report §5.61–§5.62.)

## Reproduce

```bash
.venv/bin/python experiments/cross_block/cross_block_probe.py
```

Writes `results.json` (all 42 instances, per-mask legality, all baselines).
No files outside `experiments/cross_block/` are modified.
