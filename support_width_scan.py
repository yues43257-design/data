#!/usr/bin/env python3
"""Support-width of masks that tracker-v1 rejects: would a two-level certifier pay?

Plan under test (proposed by the reviewer):

    level 1 : tracker v1  (fast, sound, conservative)
    level 2 : if v1 rejects, take v1's *support set* (a sound over-approximation of
              the correction's support); if |support| <= W, decide diagonality
              exactly on those W qubits (2**W matrix).  Sound by construction.

So the payoff is governed by the **support-width distribution** of the rejected
masks.  Part A measures it where the exact verifier is available (<=11 qubits, so
we can restrict to masks that are *actually legal* -- the true recall losses).
Part B measures it on wide circuits (24/50/100 qubits) where no exact oracle
exists, to see whether the window stays small at scale.

Pre-registered decision rule: if Part A coverage at W<=10 exceeds 30% of the
recoverable masks, build the two-level certifier; otherwise close the direction.

Usage
-----
    .venv/bin/python support_width_scan.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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


ST = _load("sw_tracker", "support_tracker.py")
SYM = _load("sw_symbolic", "symbolic_verifier.py")
ULT = _load("sw_ult", "ultimate_scalability_test.py")
WIDE = _load("sw_wide", "wide_scale_stress.py")


def final_support_size(schedule: Dict[str, Any], mask: int) -> int:
    """Replay tracker-v1's propagation and return |support| at the end."""
    support: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "h":
                continue                                  # support unchanged
            if name == "cx":
                if int(qubits[1]) in support:
                    support.add(int(qubits[0]))
            elif name == "swap":
                a, b = int(qubits[0]), int(qubits[1])
                in_a, in_b = a in support, b in support
                if in_a:
                    support.add(b)
                if in_b:
                    support.add(a)
        if block["target"] in support:
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
        if (mask >> index) & 1:
            support.add(int(block["target"]))
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
    return len(support)


def part_a(controls, blocks, seeds, h_probs, thresholds):
    print("=" * 92)
    print("  PART A  <=11q, ground truth available: support width of masks that are")
    print("          LEGAL (exact) but REJECTED by tracker v1  == the recall losses")
    print("=" * 92)
    print(f"  {'h':>5}{'losses':>9}{'median W':>10}{'max W':>7}"
          + "".join(f"{'<= ' + str(t):>9}" for t in thresholds))
    rows = []
    all_widths: List[int] = []
    for h_prob in h_probs:
        widths: List[int] = []
        for num_controls in controls:
            for seed in range(seeds):
                verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks,
                                                       seed * 5 + 3, h_prob)
                schedule = ST.homogeneous_schedule(num_controls, blocks,
                                                   seed * 5 + 3, h_prob)
                for mask in range(1 << blocks):
                    if verifier.is_safe(mask) and not ST.track_safe(schedule, mask):
                        widths.append(final_support_size(schedule, mask))
        if not widths:
            print(f"  {h_prob:>5.1f}{0:>9}{'n/a':>10}{'n/a':>7}"
                  + "".join(f"{'n/a':>9}" for _ in thresholds))
            continue
        all_widths.extend(widths)
        cov = [sum(1 for w in widths if w <= t) / len(widths) for t in thresholds]
        print(f"  {h_prob:>5.1f}{len(widths):>9}{int(np.median(widths)):>10}"
              f"{max(widths):>7}" + "".join(f"{c:>9.1%}" for c in cov))
        rows.append({"h_prob": h_prob, "losses": len(widths),
                     "median_W": int(np.median(widths)), "max_W": max(widths),
                     "coverage": {str(t): c for t, c in zip(thresholds, cov)}})
    if all_widths:
        cov10 = sum(1 for w in all_widths if w <= 10) / len(all_widths)
        print(f"  pooled losses={len(all_widths)}  coverage(W<=10)= {cov10:.1%}  "
              f"-> pre-registered rule (>30% => build level 2): "
              f"{'BUILD' if cov10 > 0.30 else 'CLOSE'}")
    print()
    return {"rows": rows, "pooled": all_widths}


def part_b(widths_q, blocks, h_prob, thresholds):
    print("=" * 92)
    print("  PART B  wide circuits (>20q, no exact oracle exists): how wide does the")
    print("          support get for masks tracker v1 rejects?")
    print("=" * 92)
    print(f"  {'width':>6}{'blocks':>7}{'tested':>8}{'median W':>10}{'max W':>7}"
          + "".join(f"{'<= ' + str(t):>9}" for t in thresholds))
    rows = []
    for width in widths_q:
        instance = ULT.WideInstance(width=width, blocks=blocks, seed=12345,
                                    h_prob=h_prob)
        schedule = WIDE.build_schedule(instance)
        k = instance.blocks
        masks = [(1 << i) for i in range(k)] + [(1 << k) - 1]
        widths: List[int] = []
        for mask in masks:
            if not ST.track_safe(schedule, mask):
                widths.append(final_support_size(schedule, mask))
        if not widths:
            print(f"  {width:>6}{k:>7}{len(masks):>8}{'n/a':>10}{'n/a':>7}"
                  + "".join(f"{'n/a':>9}" for _ in thresholds))
            continue
        cov = [sum(1 for w in widths if w <= t) / len(widths) for t in thresholds]
        print(f"  {width:>6}{k:>7}{len(masks):>8}{int(np.median(widths)):>10}"
              f"{max(widths):>7}" + "".join(f"{c:>9.1%}" for c in cov))
        rows.append({"width": width, "blocks": k, "tested": len(masks),
                     "median_W": int(np.median(widths)), "max_W": max(widths),
                     "coverage": {str(t): c for t, c in zip(thresholds, cov)}})
    print()
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="support-width scan.")
    parser.add_argument("--blocks", type=int, default=6, help="Part A circuit blocks")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--controls", default="2,3,4,5")
    parser.add_argument("--h-probs", default="0.1,0.3,0.7")
    parser.add_argument("--widths", default="24,50,100")
    parser.add_argument("--wide-blocks", type=int, default=40)
    parser.add_argument("--wide-h", type=float, default=0.5)
    parser.add_argument("--thresholds", default="6,8,10,12")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "support_width_scan.json"))
    args = parser.parse_args(argv)
    controls = [int(c) for c in args.controls.split(",") if c.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]
    widths_q = [int(w) for w in args.widths.split(",") if w.strip()]
    thresholds = [int(t) for t in args.thresholds.split(",") if t.strip()]

    a = part_a(controls, args.blocks, args.seeds, h_probs, thresholds)
    b = part_b(widths_q, args.wide_blocks, args.wide_h, thresholds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"part_a": a, "part_b": b}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
