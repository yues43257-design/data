#!/usr/bin/env python3
"""Independent probe: can cross-block phase merging beat 4 T per block?

Scope (deliberately small, N <= 10): m = 2..5 Toffoli/MCX blocks, 2-4 controls,
H-interleaved layers.  Legality is decided by the **dense** criterion
``relative_phase_ok``; nothing here touches the tracker or the paper's claims.

What "merge" means in this probe
  * the residual of a relative-phase block is a diagonal phase polynomial;
  * inside an H-free segment those terms can be merged and re-synthesised;
  * we use the repo's existing TODD entry point (``phase_block_optimize``), which
    is exactly segment-wise phase-polynomial re-synthesis, as the merged result.

Baselines: all-exact (7m), all-relative (4m), tracker greedy, pyZX full_reduce.

Usage
-----
    .venv/bin/python experiments/cross_block/cross_block_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC        # noqa: E402
import realworld_benchmark as RB              # noqa: E402
import support_tracker as ST                  # noqa: E402
import ultimate_scalability_test as ULT       # noqa: E402
import wide_scale_stress as WS                # noqa: E402

VERSION = "cross-block-v1"
EXACT_T, REL_T = 7, 4


def run_instance(width: int, blocks: int, h_prob: float, seed: int) -> Dict[str, Any]:
    inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                            h_prob=h_prob, ancilla_pool=2)
    if inst.total > 10:
        return {"skipped": f"N={inst.total}>10"}
    ideal = inst.circuit(0)
    schedule = WS.build_schedule(inst)
    rows = []
    for mask in range(1 << blocks):
        circuit = inst.circuit(mask)
        rows.append({
            "mask": mask,
            "upgrades": bin(mask).count("1"),
            "T": EXACT_T * (blocks - bin(mask).count("1")) + REL_T * bin(mask).count("1"),
            "legal": bool(RPC.relative_phase_ok(ideal, circuit)),
        })
    legal_rows = [r for r in rows if r["legal"]]
    best_legal = min(legal_rows, key=lambda r: r["T"]) if legal_rows else None
    greedy = ST.tracker_greedy(schedule, [0.0] * blocks)
    greedy_row = next(r for r in rows if r["mask"] == greedy)

    all_rel_mask = (1 << blocks) - 1
    all_rel = next(r for r in rows if r["mask"] == all_rel_mask)
    target_mask = all_rel_mask if all_rel["legal"] else (
        best_legal["mask"] if best_legal else 0)
    target_circuit = inst.circuit(target_mask)
    fr = RB.pyzx_full_reduce(target_circuit, timeout=30.0)
    todd = RB.todd_t_count(target_circuit, timeout=30.0)
    fr_T = fr[0]["t_count"] if fr and fr[0] else None
    return {
        "width": width, "N": inst.total, "blocks": blocks, "h_prob": h_prob,
        "seed": seed,
        "T_exact_7m": EXACT_T * blocks,
        "T_relative_4m": REL_T * blocks,
        "all_relative_legal": all_rel["legal"],
        "n_legal_masks": len(legal_rows),
        "best_legal_mask": best_legal["mask"] if best_legal else None,
        "best_legal_T": best_legal["T"] if best_legal else None,
        "tracker_greedy_mask": greedy,
        "tracker_greedy_T": greedy_row["T"],
        "tracker_greedy_legal": greedy_row["legal"],
        "merge_target_mask": target_mask,
        "merge_target_legal": next(r for r in rows if r["mask"] == target_mask)["legal"],
        "full_reduce_T": fr_T,
        "full_reduce_seconds": fr[1] if fr else None,
        "todd_T": todd[0] if todd else None,
        "todd_seconds": todd[1] if todd else None,
        "merge_T": todd[0] if todd else None,
        "merge_beats_4m": bool(todd and todd[0] < REL_T * blocks),
        "merge_beats_best_legal": bool(todd and best_legal and todd[0] < best_legal["T"]),
        "full_reduce_beats_merge": bool(fr_T is not None and todd and todd[0] is not None
                                        and fr_T <= todd[0]),
        "dense_counterexamples": sum(1 for r in rows if r["legal"] is None),
        "rows": rows,
    }


def main() -> int:
    results: List[Dict[str, Any]] = []
    for width, blocks in ((4, 2), (4, 3), (5, 2), (5, 3), (5, 4), (6, 5), (6, 3)):
        for h_prob in (0.0, 0.3, 0.7):
            for seed in (1, 2):
                row = run_instance(width, blocks, h_prob, seed)
                if "skipped" not in row:
                    results.append(row)
    gains = [r for r in results if r["merge_beats_4m"]]
    print("=" * 104)
    print(f"  cross-block merge probe  {VERSION}  instances={len(results)}")
    print("=" * 104)
    print(f"  {'N':>3}{'m':>3}{'h':>5}{'4m':>4}{'bestLeg':>8}{'greedy':>7}"
          f"{'FR':>5}{'TODD':>6}{'merge<4m':>9}{'FR<=TODD':>9}")
    for r in results:
        print(f"  {r['N']:>3}{r['blocks']:>3}{r['h_prob']:>5.1f}{r['T_relative_4m']:>4}"
              f"{str(r['best_legal_T']):>8}{r['tracker_greedy_T']:>7}"
              f"{str(r['full_reduce_T']):>5}{str(r['todd_T']):>6}"
              f"{str(r['merge_beats_4m']):>9}{str(r['full_reduce_beats_merge']):>9}")
    print("-" * 104)
    print(f"  merge < 4m: {len(gains)}/{len(results)}")
    print(f"  FR <= TODD: {sum(1 for r in results if r['full_reduce_beats_merge'])}/{len(results)}")
    print(f"  instances whose all-relative mask is legal: "
          f"{sum(1 for r in results if r['all_relative_legal'])}/{len(results)}")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({"version": VERSION, "instances": results}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
