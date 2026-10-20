#!/usr/bin/env python3
"""Core figure: best T saved vs decision cost (log x).

Reads ``reports/figure_data/hybrid_check.json`` and plots, for each population,
four strategies:

    tracker-only   cost = tracker time,   y = best T the tracker accepts
    greedy         cost = tracker time,   y = best T of its one proposal
    one_shot       cost = total,          y = best T if legal
    enum           cost = total,          y = best T over all masks

The y-axis is **best T saved**, not recall: "recall 1.00 with best_T 12" must not
look like a win.

Usage
-----
    .venv/bin/python plot_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "reports" / "figure_data" / "hybrid_check.json"
OUT = ROOT / "reports" / "figures" / "pareto_best_t.pdf"


def main() -> int:
    data = json.loads(DATA.read_text())
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    markers = {"homogeneous": "o", "wide": "s"}
    for population, marker in markers.items():
        enum = data["results"][f"{population}/enum"]["totals"]
        enum_best = max(enum["best_saving"], 1)
        points = []
        points.append(("tracker-only", enum["tracker_s"], enum["tracker_only_saving"]))
        for mode in ("one_shot", "greedy", "enum"):
            tot = data["results"][f"{population}/{mode}"]["totals"]
            cost = tot["tracker_s"] + tot["exact_s"]
            points.append((mode, cost, tot["best_saving"]))
        for name, cost, best in points:
            ax.scatter(max(cost, 1e-6), best, marker=marker, s=42,
                       label=f"{population}: {name}")
            ax.annotate(f"{name}\n{best / enum_best:.2f}×",
                        (max(cost, 1e-6), best), fontsize=6,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("log")
    ax.set_xlabel("decision cost (s, log)")
    ax.set_ylabel("best T saved")
    ax.set_title("best T vs cost (annotation = fraction of enum best)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(OUT.with_suffix(".png"), dpi=160)
    print(f"wrote {OUT} (+ .png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
