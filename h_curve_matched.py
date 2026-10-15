#!/usr/bin/env python3
"""Matched-instance h_prob curve + oracle-provenance audit.

Why this exists
---------------
The report's §5.3 "H cliff" (h=0 -> 100%, h>=0.1 -> ~5%) was measured with the
**exact** reduced verifier (``mixed_stress.MixedVerifier``), not with the
tracker -- so it is not a tracker artefact.  But two problems remain:

* the headline number is a *greedy capture* on ONE seed per h; the curve shape
  therefore mixes the h effect with instance-to-instance variance;
* it was measured on the *mixed shared-register* family only, yet is written up
  as a universal "physical ceiling".

This script re-measures it properly: **paired instances** (the control counts and
block count are drawn once per seed and held fixed across every h), 20 seeds,
and reports mean +/- std for metrics that are unambiguous under full enumeration:

    legal-fraction      |{m : exact(m)}| / 2**k
    single-block rate   |{i : exact(1<<i)}| / k
    greedy capture      greedy_saving / oracle_saving     (ordering-free for exact, see S8)
    all-relative safe   exact(2**k - 1)

Everything is measured with the exact verifier (``d = 2**(n_max+1)``), so the
numbers are oracle facts, not tracker conservativeness.

Usage
-----
    .venv/bin/python h_curve_matched.py
    .venv/bin/python h_curve_matched.py --seeds 20 --blocks 12 --controls 2,3,4,5
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


MS = _load("hc_mixed", "mixed_stress.py")
GEN = MS.GEN

# ---------------------------------------------------------------------------
# oracle provenance (which verifier produced which report number)
# ---------------------------------------------------------------------------

PROVENANCE = [
    ("§5.1 verifier bench", "exact simulation vs reduced symbolic", "exact"),
    ("§5.2 greedy optimality (500 circuits)", "MixedVerifier.deviation_table", "exact"),
    ("§5.3 h_prob sweep (the 'cliff')", "mixed_stress.run_stress -> MixedVerifier", "exact"),
    ("§5.4 conservative-criterion loss", "support_tracker vs exact", "tracker vs exact"),
    ("§5.5 tracker audit", "ReducedSymbolicVerifier vs support_tracker", "exact vs tracker"),
    ("§6 / §7 real circuits", "<=11q exact, >11q support_tracker", "mixed"),
    ("§8 ordering ablation A1/A2", "MixedVerifier (exact)", "exact"),
    ("§8 ordering ablation B", "support_tracker", "tracker"),
]


def measure(counts: Sequence[int], seed: int, h_prob: float, blocks: int) -> Dict[str, Any]:
    instance = MS.MixedInstance(list(counts), seed, h_prob)
    verifier = instance.verifier
    table = verifier.deviation_table()
    tol = verifier.tolerance
    k = int(blocks)
    base = verifier.t_count(0)

    legal = [table[m] <= tol for m in range(1 << k)]
    savings = [base - verifier.t_count(m) for m in range(1 << k)]

    deltas = [int(GEN.block_t_counts(int(n))[0] - GEN.block_t_counts(int(n))[1])
              for n in counts]
    order = sorted(range(k), key=lambda i: (-deltas[i], -i))
    accepted = 0
    for block in order:
        candidate = accepted | (1 << block)
        if legal[candidate]:
            accepted = candidate
    greedy_saving = base - verifier.t_count(accepted)

    legal_savings = [s for m, s in enumerate(savings) if legal[m]]
    oracle_saving = max(legal_savings) if legal_savings else 0

    return {
        "legal_fraction": sum(legal) / (1 << k),
        "single_block_rate": sum(legal[1 << i] for i in range(k)) / k,
        "oracle_saving": float(oracle_saving),
        "greedy_saving": float(greedy_saving),
        "capture": (greedy_saving / oracle_saving) if oracle_saving else 0.0,
        "all_relative_safe": bool(legal[(1 << k) - 1]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="matched h_prob curve (exact oracle).")
    parser.add_argument("--controls", default="2,3,4,5")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--h-probs", default="0.0,0.1,0.3,0.7,1.0")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "h_curve_matched.json"))
    args = parser.parse_args(argv)

    controls = [int(c) for c in args.controls.split(",") if c.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]
    k = args.blocks

    print("=" * 96)
    print("  oracle provenance (which verifier produced which number)")
    print("=" * 96)
    for where, how, kind in PROVENANCE:
        print(f"  {where:<38} {kind:<14} {how}")

    print()
    print("=" * 96)
    print(f"  matched-instance h curve  (blocks={k}, controls={controls}, "
          f"seeds={args.seeds}, exact verifier, full 2^{k} enumeration)")
    print("=" * 96)
    print("  controls/blocks drawn ONCE per seed and held FIXED across every h")
    print()
    print(f"  {'h':>5}{'legal frac':>18}{'single-block':>18}{'capture':>16}"
          f"{'oracle T':>16}{'all-rel safe':>14}")

    # one instance shape per seed, reused for every h  -> paired design
    shapes: List[Sequence[int]] = []
    rng = random.Random(20261015)
    for _ in range(args.seeds):
        shapes.append([int(rng.choice(controls)) for _ in range(k)])

    rows: List[Dict[str, Any]] = []
    for h_prob in h_probs:
        per_seed = [measure(shape, 7000 + index, h_prob, k)
                    for index, shape in enumerate(shapes)]
        def col(name):
            return np.array([r[name] for r in per_seed], dtype=float)
        legal, single, capture = col("legal_fraction"), col("single_block_rate"), col("capture")
        oracle = col("oracle_saving")
        allrel = np.array([r["all_relative_safe"] for r in per_seed], dtype=float)
        print(f"  {h_prob:>5.1f}"
              f"  {legal.mean():.3f} ± {legal.std():.3f}".ljust(24)
              + f"  {single.mean():.3f} ± {single.std():.3f}".ljust(20)
              + f"  {capture.mean():.3f} ± {capture.std():.3f}".ljust(18)
              + f"  {oracle.mean():7.1f} ± {oracle.std():5.1f}".ljust(18)
              + f"  {allrel.mean():.2f}")
        rows.append({"h_prob": h_prob, "seeds": args.seeds,
                     "legal_fraction_mean": float(legal.mean()),
                     "legal_fraction_std": float(legal.std()),
                     "single_block_mean": float(single.mean()),
                     "single_block_std": float(single.std()),
                     "capture_mean": float(capture.mean()),
                     "capture_std": float(capture.std()),
                     "oracle_saving_mean": float(oracle.mean()),
                     "all_relative_safe_rate": float(allrel.mean())})

    print("=" * 96)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provenance": PROVENANCE, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
