#!/usr/bin/env python3
"""Strategy-A gate: does the routing-policy spread grow with scale and a sparse
hardware topology?

At 7 qubits (C^4X) all six policies in ``routing_policy.ROUTING_POLICIES`` reach
the same cost on both fully-connected and line topologies, so a learned routing
policy has nothing to win there.  The hypothesis behind strategy A is that at
20-50 qubits on a *sparse* device (grid / heavy-hex-like connectivity) the SWAP
overhead dominates and the policy choice finally matters.

This script measures exactly that: for random phase polynomials at a given size
it routes each policy on each of several architectures and reports the spread of
CX count, CX-depth and T-depth between the best and the worst policy.  The gate is
positive when the best policy beats the worst by a margin (>= ``--threshold``,
default 10%) -- only then is learned routing worth training.

Usage
-----
    .venv/bin/python scale_headroom.py --qubits 16 --terms 12
    .venv/bin/python scale_headroom.py --qubits 32 --terms 16 --architectures full,line,grid
    .venv/bin/python scale_headroom.py --qubits 32 --json /tmp/scale.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import routing_policy as rp  # noqa: E402


def random_phase_terms(n_qubits: int, n_terms: int, seed: int, max_support: int = 5):
    """Random phase polynomial: ``n_terms`` parities with odd/even phases."""
    rng = random.Random(seed)
    terms: "Dict[tuple[int, ...], Any]" = {}
    for _ in range(n_terms):
        size = rng.randint(2, min(max_support, n_qubits))
        qubits = rng.sample(range(n_qubits), size)
        parity = tuple(1 if q in qubits else 0 for q in range(n_qubits))
        phase = rng.choice([1, 2, 3, 5, 6, 7])
        terms[parity] = rp.Fraction((rp.Fraction(terms.get(parity, 0)) + rp.Fraction(phase, 4)) % 2)
        if terms[parity] == 0:
            del terms[parity]
    return {p: v for p, v in terms.items() if v != 0}


def route_costs(terms, n_qubits: int, architecture: str, attempts: int = 1):
    """Cost table of every policy on ``architecture`` (``None`` on failure)."""
    rows: "List[Dict[str, Any]]" = []
    for index, policy in enumerate(rp.ROUTING_POLICIES):
        started = time.time()
        row: "Dict[str, Any]" = {"policy": policy["name"], "architecture": architecture}
        try:
            circuit = rp.route_with_policy(
                terms,
                n_qubits,
                policy,
                attempts=attempts,
                architecture_name=architecture,
            )
            metrics = rp.LINUX_MCX.compute_ftq_metrics(circuit)
            row.update(
                {
                    "ok": True,
                    "t_count": metrics.t_count,
                    "t_depth": metrics.t_depth,
                    "cx_depth": metrics.cx_depth,
                    "cx_count": sum(
                        1 for gate in circuit.gates if gate.name in ("CNOT", "CZ")
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            row.update({"ok": False, "error": str(exc)[:100]})
        row["seconds"] = round(time.time() - started, 2)
        rows.append(row)
    return rows


def spread(values: "Sequence[int]") -> float:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    best, worst = min(values), max(values)
    return 0.0 if worst == 0 else 100.0 * (worst - best) / worst


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy-A gate (scale x topology routing spread).")
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--terms", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--architectures", default="full,line")
    parser.add_argument("--threshold", type=float, default=10.0, help="percent spread that counts as headroom")
    parser.add_argument("--time-budget", type=float, default=600.0, help="seconds before stopping early")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    architectures = [a.strip() for a in args.architectures.split(",") if a.strip()]
    started = time.time()
    report: "List[Dict[str, Any]]" = []
    print(
        f"    size: {args.qubits} qubits, {args.terms} terms | architectures: {architectures} | "
        f"attempts/policy: {args.attempts}"
    )
    print(
        f"{'arch':>8} | {'seed':>4} | {'policies ok':>11} | {'CX min/max':>10} | "
        f"{'CX spread':>9} | {'CXdep spread':>12} | {'Tdep spread':>11}"
    )
    for architecture in architectures:
        for seed in range(args.seeds):
            if time.time() - started > args.time_budget:
                print("    [stop] time budget reached")
                break
            terms = random_phase_terms(args.qubits, args.terms, seed)
            rows = route_costs(terms, args.qubits, architecture, attempts=args.attempts)
            ok_rows = [row for row in rows if row.get("ok")]
            # the real baseline: what the pipeline uses today (best of the
            # Steiner/Meijer attempts, i.e. NOT one fixed policy)
            default_cost = None
            try:
                # same attempt budget as the policies: an apples-to-apples
                # baseline (attempts=3 at 30+ qubits can take minutes per instance)
                default_circuit = rp.LINUX_MCX.route_phase_term_set(
                    terms, args.qubits, attempts=args.attempts
                )
                default_metrics = rp.LINUX_MCX.compute_ftq_metrics(default_circuit)
                default_cost = {
                    "cx_count": sum(
                        1 for gate in default_circuit.gates if gate.name in ("CNOT", "CZ")
                    ),
                    "cx_depth": default_metrics.cx_depth,
                    "t_depth": default_metrics.t_depth,
                }
            except Exception as exc:  # pragma: no cover - defensive
                default_cost = {"error": str(exc)[:80]}
            best_row = min(
                ok_rows,
                key=lambda row: (row["cx_count"], row["cx_depth"], row["t_depth"]),
            ) if ok_rows else None
            gain_vs_default = None
            if default_cost and best_row and "cx_count" in default_cost:
                base = default_cost["cx_count"]
                gain_vs_default = 0.0 if not base else 100.0 * (base - best_row["cx_count"]) / base
            cx = [row["cx_count"] for row in ok_rows]
            cxd = [row["cx_depth"] for row in ok_rows]
            td = [row["t_depth"] for row in ok_rows]
            entry = {
                "architecture": architecture,
                "seed": seed,
                "qubits": args.qubits,
                "terms": len(terms),
                "policies_ok": len(ok_rows),
                "cx_min": min(cx) if cx else None,
                "cx_max": max(cx) if cx else None,
                "cx_spread_pct": spread(cx),
                "cx_depth_spread_pct": spread(cxd),
                "t_depth_spread_pct": spread(td),
                "default_cost": default_cost,
                "best_policy": best_row["policy"] if best_row else None,
                "best_cx_count": best_row["cx_count"] if best_row else None,
                "gain_vs_default_pct": gain_vs_default,
                "rows": rows,
            }
            report.append(entry)
            default_cx = default_cost.get("cx_count") if default_cost else None
            print(
                f"{architecture:>8} | {seed:>4} | {len(ok_rows):>4}/{len(rows):<6} | "
                f"{(str(min(cx)) + '/' + str(max(cx))) if cx else '-':>10} | "
                f"{entry['cx_spread_pct']:>8.1f}% | {entry['cx_depth_spread_pct']:>11.1f}% | "
                f"{entry['t_depth_spread_pct']:>10.1f}% | "
                f"default CX={(default_cx if default_cx is not None else '-')}, "
                f"best={entry['best_policy']}, "
                f"gain={(f'{gain_vs_default:.1f}%' if gain_vs_default is not None else '-')}"
            )
    best = max(
        (entry["cx_spread_pct"] for entry in report if entry["policies_ok"] >= 2),
        default=0.0,
    )
    print(
        f"\n    [gate] largest CX-count spread between policies: {best:.1f}% "
        f"(threshold {args.threshold:.0f}%)"
    )
    if best >= args.threshold:
        print("    [verdict] POSITIVE -> learned/selected routing has measurable room at this scale.")
    else:
        print(
            "    [verdict] no spread yet -> go larger / sparser, or drop the routing-learning "
            "direction."
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"    [json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
