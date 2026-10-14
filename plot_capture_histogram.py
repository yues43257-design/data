#!/usr/bin/env python3
"""Per-circuit headroom capture-rate histogram (companion to ``fig4``).

Reads ``reports/figure_data/oracle_vs_greedy.json`` (written by
``oracle_vs_greedy.py``) and plots the distribution of the per-circuit capture
rate

    capture_i = greedy_saved_i / oracle_saved_i        (0 .. 100 %)

The point of the figure is its *shape*: the vast majority of circuits sit
exactly at 100 % (the greedy decoder matches the exhaustive oracle), and a short
long tail of sub-optimal decodes is scattered to the left.  A linear-count
histogram shows the spike at 100 %; a zoomed inset makes the tail readable.

Usage
-----
    .venv/bin/python plot_capture_histogram.py
    .venv/bin/python plot_capture_histogram.py --bin-width 2.0 --out-dir reports/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_rates(data_path: Path) -> Tuple[np.ndarray, int]:
    """Return (capture rates in %, number of circuits skipped for oracle == 0)."""
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    oracle = np.array([r["oracle_saved"] for r in rows], dtype=float)
    greedy = np.array([r["greedy_saved"] for r in rows], dtype=float)
    usable = oracle > 0
    rates = 100.0 * greedy[usable] / oracle[usable]
    skipped = int((~usable).sum())
    return rates, skipped


def plot(rates: np.ndarray, skipped: int, out_dir: Path,
         formats: Sequence[str], bin_width: float) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    bins = np.arange(0.0, 100.0 + bin_width, bin_width)
    counts, _ = np.histogram(rates, bins=bins)
    at_100 = int((rates >= 100.0 - 1e-9).sum())
    below = int((rates < 100.0 - 1e-9).sum())

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    _, _, patches = ax.hist(rates, bins=bins, edgecolor="white", linewidth=0.6)
    # colour the optimal bar (touching 100 %) green, the tail steel-blue
    for patch, right in zip(patches, bins[1:]):
        patch.set_facecolor("#2e7d32" if right >= 100.0 - 1e-9 else "#4c72b0")

    ax.set_xlabel("Headroom capture rate  (greedy saving / oracle saving), %")
    ax.set_ylabel("Count (number of circuits)")
    ax.set_title("Distribution of per-circuit headroom capture rate\n"
                 f"{len(rates)} random mixed circuits, 5-12 blocks", pad=12)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, max(counts.max() * 1.18, 5))
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # headline annotation on the 100 % spike
    ax.annotate(f"{at_100} / {len(rates)} circuits = {at_100 / len(rates):.1%}\n"
                "exactly at 100 %",
                xy=(100.0 - bin_width / 2.0, at_100),
                xytext=(50.0, counts.max() * 1.03),
                fontsize=10, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.4),
                color="#1b5e20")

    stats = (f"mean   = {rates.mean():.2f} %\n"
             f"median = {np.median(rates):.2f} %\n"
             f"min    = {rates.min():.2f} %\n"
             f"below 100 % : {below} / {len(rates)}")
    if skipped:
        stats += f"\n(skipped {skipped} with oracle = 0)"
    ax.text(0.015, 0.97, stats, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.6, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                      edgecolor="#cccccc"))

    # zoomed inset: the sub-optimal tail, linear count so 1-2 circuits are visible
    inset = ax.inset_axes([0.07, 0.30, 0.45, 0.40])
    inset.hist(rates, bins=bins, color="#c0392b", edgecolor="white", linewidth=0.6)
    inset.set_xlim(30.0, 100.0)
    inset.set_ylim(0.0, 4.0)
    inset.set_title("sub-optimal tail (zoom)", fontsize=8.5, pad=3)
    inset.tick_params(labelsize=7.5)
    inset.grid(axis="y", alpha=0.25, linestyle=":")
    inset.set_axisbelow(True)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in formats:
        path = out_dir / f"fig4b_capture_histogram.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Capture-rate histogram (fig4b).")
    parser.add_argument("--data", default=str(ROOT / "reports" / "figure_data" /
                                              "oracle_vs_greedy.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "figures"))
    parser.add_argument("--bin-width", type=float, default=2.5,
                        help="histogram bin width in percentage points")
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args(argv)

    rates, skipped = load_rates(Path(args.data))
    at_100 = int((rates >= 100.0 - 1e-9).sum())
    print(f"circuits          : {len(rates)}  (skipped {skipped} with oracle = 0)")
    print(f"exactly 100 %     : {at_100} / {len(rates)} = {at_100 / len(rates):.1%}")
    print(f"mean / median     : {rates.mean():.2f} % / {np.median(rates):.2f} %")
    print(f"min               : {rates.min():.2f} %")
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for path in plot(rates, skipped, Path(args.out_dir), formats, args.bin_width):
        print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
