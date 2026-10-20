#!/usr/bin/env python3
"""Item 2: tracker as a pre-filter, exact check only on the rejects.

    accept  ->  certified by the tracker (certificate attached), stop
    reject  ->  if N <= threshold, run the exact criterion on that one mask
    reject and N > threshold -> "unresolved" (the tracker cannot certify it and
                                exact is too expensive here)

The point is the cost profile: the tracker is sound, so an *accept* never needs
verification; only rejects pay for exact checking, and only up to the threshold.
On small N this gives recall 1.0 at a measurable price, which is exactly the
speed/recall trade-off the paper's Pareto figure should show.

Usage
-----
    .venv/bin/python hybrid_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC      # noqa: E402
import support_tracker as ST                # noqa: E402
import ultimate_scalability_test as ULT     # noqa: E402
import wide_scale_stress as WS              # noqa: E402

HYBRID_VERSION = "hybrid-v1"
DENSE_LIMIT = 11          # dense Operator is affordable here


def check_mask(schedule: Dict[str, Any],
               ideal: Any,
               build_candidate: Callable[[int], Any],
               mask: int, n_qubits: int,
               threshold: int = 18) -> Dict[str, Any]:
    """One mask through the hybrid pipeline."""
    started = time.perf_counter()
    certificate = ST.certify(schedule, mask)
    if certificate["accepted"]:
        return {"mask": mask, "verdict": "accepted", "route": "tracker",
                "seconds": time.perf_counter() - started,
                "certificate": certificate}
    if n_qubits <= min(threshold, DENSE_LIMIT):
        legal = bool(RPC.relative_phase_ok(ideal, build_candidate(mask)))
        return {"mask": mask, "verdict": "legal" if legal else "illegal",
                "route": "exact", "seconds": time.perf_counter() - started,
                "certificate": certificate}
    return {"mask": mask, "verdict": "unresolved", "route": "none",
            "seconds": time.perf_counter() - started,
            "certificate": certificate}


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    totals = {"masks": 0, "legal": 0, "tracker_accept": 0, "hybrid_legal": 0,
              "exact_calls": 0, "tracker_s": 0.0, "exact_s": 0.0}
    for width, blocks, h_prob in ((4, 3, 0.1), (4, 3, 0.5), (4, 3, 0.7),
                                  (6, 3, 0.1), (6, 3, 0.7), (8, 2, 0.7),
                                  (4, 4, 0.5)):
        inst = ULT.WideInstance(width=width, blocks=blocks, seed=20261020,
                                h_prob=h_prob, ancilla_pool=2)
        if inst.total > 10:
            continue
        ideal = inst.circuit(0)
        schedule = WS.build_schedule(inst)
        legal = accept = hybrid = exact_calls = 0
        t_tracker = t_exact = 0.0
        for mask in range(1 << blocks):
            truth = bool(RPC.relative_phase_ok(ideal, inst.circuit(mask)))
            result = check_mask(schedule, ideal, lambda m: inst.circuit(m),
                                mask, inst.total, threshold=18)
            legal += truth
            accept += int(result["verdict"] == "accepted")
            hybrid += int(result["verdict"] in ("accepted", "legal"))
            exact_calls += int(result["route"] == "exact")
            if result["route"] == "exact":
                t_exact += result["seconds"]
            else:
                t_tracker += result["seconds"]
        rows.append({"N": inst.total, "h_prob": h_prob, "blocks": blocks,
                     "masks": 1 << blocks, "legal": legal,
                     "tracker_accept": accept, "hybrid_legal": hybrid,
                     "exact_calls": exact_calls,
                     "tracker_s": round(t_tracker, 4),
                     "exact_s": round(t_exact, 4),
                     "tracker_recall": round(accept / legal, 3) if legal else None,
                     "hybrid_recall": round(hybrid / legal, 3) if legal else None})
        for key in ("masks", "legal", "tracker_accept", "hybrid_legal"):
            pass
        totals["masks"] += 1 << blocks
        totals["legal"] += legal
        totals["tracker_accept"] += accept
        totals["hybrid_legal"] += hybrid
        totals["exact_calls"] += exact_calls
        totals["tracker_s"] += t_tracker
        totals["exact_s"] += t_exact
    return {"rows": rows, "totals": totals}


def run_homogeneous() -> Dict[str, Any]:
    """The population where the tracker is known to be conservative (§5.5)."""
    import experiment_setup as SETUP
    import symbolic_verifier as SYM

    rows: List[Dict[str, Any]] = []
    agg = {"masks": 0, "legal": 0, "tracker_accept": 0, "hybrid_legal": 0,
           "exact_calls": 0, "tracker_s": 0.0, "exact_s": 0.0}
    for h_prob in (0.1, 0.3, 0.7):
        for num_controls in (2, 3, 4):
            for seed_index in range(2):
                seed = SETUP.seed_for(seed_index)
                blocks = 4
                verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks, seed, h_prob)
                schedule = ST.homogeneous_schedule(num_controls, blocks, seed, h_prob)
                legal = accept = hybrid = exact_calls = 0
                t_trk = t_ex = 0.0
                for mask in range(1 << blocks):
                    truth = bool(verifier.is_safe(mask))
                    t0 = time.perf_counter()
                    cert = ST.certify(schedule, mask)
                    if cert["accepted"]:
                        t_trk += time.perf_counter() - t0
                        accept += 1
                        hybrid += int(truth)
                    else:
                        t0 = time.perf_counter()
                        ok = bool(verifier.is_safe(mask))
                        t_ex += time.perf_counter() - t0
                        exact_calls += 1
                        hybrid += int(ok)
                    legal += truth
                rows.append({"controls": num_controls, "h_prob": h_prob,
                             "seed": seed, "masks": 1 << blocks, "legal": legal,
                             "tracker_accept": accept, "hybrid_legal": hybrid,
                             "exact_calls": exact_calls,
                             "tracker_s": round(t_trk, 5),
                             "exact_s": round(t_ex, 5)})
                agg["masks"] += 1 << blocks
                agg["legal"] += legal
                agg["tracker_accept"] += accept
                agg["hybrid_legal"] += hybrid
                agg["exact_calls"] += exact_calls
                agg["tracker_s"] += t_trk
                agg["exact_s"] += t_ex
    return {"rows": rows, "totals": agg}


def main() -> int:
    started = time.perf_counter()
    print("=" * 100)
    print(f"  HYBRID PRE-FILTER  version={HYBRID_VERSION}")
    print("  tracker accept -> certified; tracker reject -> exact (N <= 18)")
    print("=" * 100)
    data = run_experiment()
    print(f"  {'N':>3}{'h':>5}{'masks':>7}{'legal':>7}{'trackAcc':>10}"
          f"{'hybrid':>8}{'exactCalls':>12}{'trk ms':>9}{'exact ms':>10}")
    for r in data["rows"]:
        print(f"  {r['N']:>3}{r['h_prob']:>5.1f}{r['masks']:>7}{r['legal']:>7}"
              f"{r['tracker_accept']:>10}{r['hybrid_legal']:>8}{r['exact_calls']:>12}"
              f"{r['tracker_s'] * 1e3:>9.2f}{r['exact_s'] * 1e3:>10.1f}")
    t = data["totals"]
    tr = t["tracker_accept"] / t["legal"] if t["legal"] else float("nan")
    hr = t["hybrid_legal"] / t["legal"] if t["legal"] else float("nan")
    print("-" * 100)
    print(f"  pooled: legal={t['legal']}  tracker={t['tracker_accept']} "
          f"(recall {tr:.3f})  hybrid={t['hybrid_legal']} (recall {hr:.3f})")
    print(f"  exact calls={t['exact_calls']}  tracker={t['tracker_s'] * 1e3:.2f} ms  "
          f"exact={t['exact_s'] * 1e3:.1f} ms")
    print(f"  ({time.perf_counter() - started:.1f}s)")
    print("\n" + "=" * 100)
    print("  homogeneous population (§5.5): where the tracker IS conservative")
    print("=" * 100)
    hom = run_homogeneous()
    ht = hom["totals"]
    htr = ht["tracker_accept"] / ht["legal"] if ht["legal"] else float("nan")
    hhr = ht["hybrid_legal"] / ht["legal"] if ht["legal"] else float("nan")
    print(f"  {'ctrl':>5}{'h':>5}{'masks':>7}{'legal':>7}{'trackAcc':>10}"
          f"{'hybrid':>8}{'exactCalls':>12}")
    for r in hom["rows"]:
        print(f"  {r['controls']:>5}{r['h_prob']:>5.1f}{r['masks']:>7}{r['legal']:>7}"
              f"{r['tracker_accept']:>10}{r['hybrid_legal']:>8}{r['exact_calls']:>12}")
    print(f"  pooled: legal={ht['legal']} tracker={ht['tracker_accept']} "
          f"(recall {htr:.3f}) hybrid={ht['hybrid_legal']} (recall {hhr:.3f})")
    print(f"  tracker={ht['tracker_s'] * 1e3:.2f} ms  exact={ht['exact_s'] * 1e3:.1f} ms")
    out = ROOT / "reports" / "figure_data" / "hybrid_check.json"
    out.write_text(json.dumps({"version": HYBRID_VERSION,
                               "threshold": 18, "dense_limit": DENSE_LIMIT,
                               **data, "homogeneous": hom}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
