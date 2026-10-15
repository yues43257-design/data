# Lineage table: every number in the report -> script -> setup

`setup_version = canonical-setup-v2` (see `experiment_setup.py`):
controls `(2,3,4,5)`, `blocks = 6`, `seeds = 3`, `seed = index*7 + 11`,
full enumeration of `2**6` masks, exact oracle = `symbolic_verifier`.

**Important:** the canonical seed rule was introduced on 2026-10-15. Figures
measured before it use `index*5 + 3` and are marked `superseded` below.

| report figure | script | setup | status |
|---|---|---|---|
| §5.1 verifier bench (times / speed-ups) | `collect_figure_data.py --benchmarks` | n/a | current |
| §5.2 greedy optimality (500 circuits) | `oracle_vs_greedy.py` | its own seed 20261014 | current |
| §5.3 h_prob sweep ("cliff") | `hprob_sweep.py` | seed 20261012 | **superseded by §5.6** |
| §5.5 tracker audit (precision/recall) | `tracker_verification.py` | `index*7+11` | current |
| §5.6 matched h curve | `h_curve_matched.py` | matched shapes, own seed | current |
| §5.7 relaxation root cause (56-77%) | `tracker_relaxation_diag.py` | `index*5+3` | **superseded (pre-seed-freeze)** |
| §5.8 v2 payoff | `tracker_bailout_check.py` | `index*5+3` | **superseded (pre-fix + pre-freeze)** |
| §5.9/§5.10 support-width scan | `support_width_scan.py` | `index*5+3` | **superseded (Part A was a ceiling artefact)** |
| §6.x real circuits | `realworld_benchmark.py` | size-map, seed 20261015 | current |
| §7 wide scale | `wide_scale_stress.py` | seed 20261015 | current |
| §8 ordering ablation | `ablation_ordering.py` | seed 20261015 | current |

**Open item:** §5.7 / §5.8 / §5.9 were re-run under the frozen seed rule on
2026-10-15 (see the follow-up commit); their previous numbers stay here as
`superseded` for traceability rather than being silently replaced.
