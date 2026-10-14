#!/usr/bin/env python3
"""Publication figures (English only) for the strategy-B experiments.

Data sources, in priority order:

1. ``reports/figure_data/*.json`` - freshly re-measured, high-precision numbers
   produced by ``collect_figure_data.py`` and ``hprob_sweep.py --data-dir ...``.
   When present these are used, which is what gives figure 2 its error bars and
   figure 3 its absolute T-counts.
2. A transcription of ``reports/strategy_b_experiments.md`` compiled into this
   file as a fallback (means only - no error bars, no absolute T-counts), so the
   script still runs before the re-measurement has been done.

Usage
-----
    .venv/bin/python plot_results.py                  # -> reports/figures/*.{png,pdf}
    .venv/bin/python plot_results.py --formats png,pdf
    .venv/bin/python plot_results.py --check          # data/consistency guard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "reports" / "figure_data"

try:
    import matplotlib
    matplotlib.use("Agg")                     # headless
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as exc:  # pragma: no cover
    print(f"matplotlib is required to draw the figures ({exc}).\n"
          f"Install it with:  .venv/bin/pip install matplotlib", file=sys.stderr)
    raise SystemExit(2)

plt.rcParams["font.family"] = "DejaVu Sans"     # English labels only
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

COLOR = {
    "sim": "#c0392b",
    "sym": "#1f77b4",
    "greedy": "#2e7d32",
    "oneshot": "#8e44ad",
    "cap": "#e67e22",
    "repair": "#1f77b4",
    "illegal": "#b0b0b0",
    "exact": "#555555",
    "baseline": "#c0392b",
}


# ---------------------------------------------------------------------------
# fallback data (transcribed from reports/strategy_b_experiments.md)
# ---------------------------------------------------------------------------

FALLBACK_BENCH = [
    {"key": "oracle", "label": "C^6X x4 oracle enumeration", "repeats": 3,
     "simulation_mean_s": 447.724, "simulation_std_s": 8.136,
     "symbolic_mean_s": 0.04690, "symbolic_std_s": 0.00737,
     "ratio_mean": 9693.0, "ratio_std": 1408.0},
    {"key": "greedy", "label": "30-block greedy decode", "repeats": 3,
     "simulation_mean_s": 21.428, "simulation_std_s": 0.511,
     "symbolic_mean_s": 0.00202, "symbolic_std_s": 0.00012,
     "ratio_mean": 10641.0, "ratio_std": 413.0},
]

FALLBACK_SWEEP = {
    "h_prob": [0.0, 0.1, 0.3, 0.5, 0.8],
    "greedy_saved": [952.2, 59.1, 54.9, 47.4, 50.4],
    "one_shot_saved": [325.2, 0.0, 0.0, 0.0, 0.0],
    "all_relative_safe": [100.0, 0.0, 0.0, 0.0, 0.0],
}

POOL_TITLES = {
    "shallow": "Shallow (n=16, 2-7 blocks)",
    "deep": "Deep (n=12, 30-50 blocks)",
}


def _load_json(name: str) -> Optional[Any]:
    path = DATA_DIR / name
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] could not read {path}: {exc}", file=sys.stderr)
        return None


def _normalise_bench(rows: Any) -> List[Dict[str, Any]]:
    """Accept both the current mean/std schema and the earlier flat one."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if "simulation_mean_s" in row:
            out.append(row)
            continue
        sim, sym = float(row["simulation_s"]), float(row["symbolic_s"])
        out.append({
            "label": row.get("label", row.get("key", "benchmark")),
            "repeats": int(row.get("repeats", 1)),
            "simulation_mean_s": sim, "simulation_std_s": 0.0,
            "symbolic_mean_s": sym, "symbolic_std_s": 0.0,
            "ratio_mean": float(row.get("ratio", sim / max(sym, 1e-12))), "ratio_std": 0.0,
        })
    return out


def _style(ax) -> None:
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# figure 1 - verifier speed-up
# ---------------------------------------------------------------------------


def figure_speedup(out_dir: Path, formats: Sequence[str]) -> List[Path]:
    measured = _load_json("verifier_bench.json")
    bench = _normalise_bench(measured) if measured else FALLBACK_BENCH
    repeats = max(row.get("repeats", 1) for row in bench)
    labels = [row["label"] for row in bench]
    sim = [row["simulation_mean_s"] for row in bench]
    sim_std = [row.get("simulation_std_s", 0.0) for row in bench]
    sym = [row["symbolic_mean_s"] for row in bench]
    sym_std = [row.get("symbolic_std_s", 0.0) for row in bench]
    ratios = [row["ratio_mean"] for row in bench]
    ratio_std = [row.get("ratio_std", 0.0) for row in bench]

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    x = np.arange(len(labels))
    width = 0.36
    bars_sim = ax.bar(x - width / 2, sim, width, yerr=sim_std, capsize=4,
                      label="Full circuit simulation", color=COLOR["sim"])
    bars_sym = ax.bar(x + width / 2, sym, width, yerr=sym_std, capsize=4,
                      label="Symbolic / reduced verifier", color=COLOR["sym"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock time per run (s, log scale)")
    title = "Verifier speed-up: simulation vs symbolic verification"
    ax.set_title(title + (f"\nmean +/- std over {repeats} runs" if repeats > 1 else ""),
                 pad=12, fontsize=12)
    _style(ax)

    def _fmt(value: float, err: float) -> str:
        text = f"{value:.4g}"
        return f"{text} +/- {err:.3g} s" if err > 0 else f"{text} s"

    for bars, values, errors in ((bars_sim, sim, sim_std), (bars_sym, sym, sym_std)):
        for rect, value, err in zip(bars, values, errors):
            ax.annotate(_fmt(value, err), (rect.get_x() + rect.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=8.5)
    for xi, ratio, err in zip(x, ratios, ratio_std):
        text = f"{ratio:,.0f}x faster" if err <= 0 else f"{ratio:,.0f} +/- {err:,.0f}x faster"
        ax.annotate(text, (xi, max(sim) * 2.2), ha="center", fontsize=12.5, fontweight="bold")
    ax.set_ylim(min(sym) * 0.4, max(sim) * 5.5)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    caption = (f"re-measured, {repeats} repeats" if measured
               else "report transcription (run collect_figure_data.py --benchmarks)")
    fig.text(0.5, -0.02, f"data: {caption}", ha="center", fontsize=8, color="#666666")
    return _save(fig, out_dir, "fig1_verifier_speedup", formats)


# ---------------------------------------------------------------------------
# figure 2 - Hadamard cliff, with error bars
# ---------------------------------------------------------------------------


def _sweep_series() -> Dict[str, Any]:
    """Mean/std of the savings per h_prob, from the sweep JSON when available."""
    sweep = _load_json("hprob_sweep.json")
    if not sweep:
        return {
            "h_prob": FALLBACK_SWEEP["h_prob"],
            "greedy_mean": FALLBACK_SWEEP["greedy_saved"],
            "greedy_std": None,
            "oneshot_mean": FALLBACK_SWEEP["one_shot_saved"],
            "oneshot_std": None,
            "relative_safe": FALLBACK_SWEEP["all_relative_safe"],
            "n": None,
            "live": False,
        }
    per_h = sweep["per_circuit"]
    keys = sorted(per_h, key=float)
    h_values = [float(k) for k in keys]
    greedy_mean, greedy_std, oneshot_mean, oneshot_std, safe = [], [], [], [], []
    n = None
    for key in keys:
        rows = per_h[key]
        g = np.array([r["exact_t"] - r["greedy_t"] for r in rows], dtype=float)
        o = np.array([r["exact_t"] - r["one_shot_t"] for r in rows], dtype=float)
        greedy_mean.append(float(g.mean()))
        greedy_std.append(float(g.std(ddof=1)) if len(g) > 1 else 0.0)
        oneshot_mean.append(float(o.mean()))
        oneshot_std.append(float(o.std(ddof=1)) if len(o) > 1 else 0.0)
        safe.append(100.0 * float(np.mean([1.0 if r["relative_safe"] else 0.0 for r in rows])))
        n = len(rows)
    return {
        "h_prob": h_values,
        "greedy_mean": greedy_mean,
        "greedy_std": greedy_std,
        "oneshot_mean": oneshot_mean,
        "oneshot_std": oneshot_std,
        "relative_safe": safe,
        "n": n,
        "live": True,
    }


def figure_cliff(out_dir: Path, formats: Sequence[str]) -> List[Path]:
    series = _sweep_series()
    h = series["h_prob"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(h, series["greedy_mean"], yerr=series["greedy_std"],
                fmt="o-", color=COLOR["greedy"], lw=2, capsize=4,
                label="Verifier greedy" + ("" if series["live"] else " (mean only)"))
    ax.errorbar(h, series["oneshot_mean"], yerr=series["oneshot_std"],
                fmt="s--", color=COLOR["oneshot"], lw=2, capsize=4,
                label="One-shot (plain fallback; zero for h_prob >= 0.1)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Inter-block Hadamard density  h_prob")
    ax.set_ylabel("T gates saved per circuit (symlog scale)")
    ax.set_title("Hadamard-interference cliff in the safe T-count budget", pad=12)
    _style(ax)
    ax.axvspan(0.0, 0.1, color="#c0392b", alpha=0.08)
    lo, hi = series["greedy_mean"][0], series["greedy_mean"][1]
    if hi > 0:
        ax.annotate(f"cliff: {lo:.1f} -> {hi:.1f} T  ({lo / hi:.1f}x drop)",
                    xy=(0.05, float(np.sqrt(lo * hi))), xytext=(0.24, lo * 0.45),
                    fontsize=11, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax2 = ax.twinx()
    ax2.plot(h, series["relative_safe"], "^:", color="#7f8c8d", lw=1.5,
             label="All-relative mask safe rate (%)")
    ax2.set_ylabel("All-relative mask safe rate (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower left",
              fontsize=9)
    ax.set_ylim(-0.3, max(series["greedy_mean"]) * 3.0)
    fig.tight_layout()
    caption = (f"mean +/- std over {series['n']} circuits per point" if series["live"]
               else "report transcription (run hprob_sweep.py --data-dir reports/figure_data)")
    fig.text(0.5, -0.02, f"data: {caption}", ha="center", fontsize=8, color="#666666")
    return _save(fig, out_dir, "fig2_hadamard_cliff", formats)


# ---------------------------------------------------------------------------
# figure 3 - absolute T-count per strategy
# ---------------------------------------------------------------------------


def figure_strategies(out_dir: Path, formats: Sequence[str]) -> List[Path]:
    pools = _load_json("strategy_pools.json")
    if not pools:
        raise SystemExit(
            "figure 3 needs absolute T-counts: run\n"
            "    .venv/bin/python collect_figure_data.py --pools\n"
            "to write reports/figure_data/strategy_pools.json"
        )
    names = list(pools["shallow"]["strategies"].keys())
    palette = {
        "all-exact": COLOR["exact"],
        "raw-threshold (illegal)": COLOR["illegal"],
        "threshold+repair": COLOR["repair"],
        "depth-cap": COLOR["cap"],
        "greedy": COLOR["greedy"],
    }
    plotted = [n for n in names if n != "all-exact"]     # shown as the baseline line

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    x = np.arange(len(pools))
    width = 0.19
    for offset, name in enumerate(plotted):
        means = [float(np.mean([r["t"] for r in pools[p]["strategies"][name]]))
                 for p in pools]
        positions = x + (offset - (len(plotted) - 1) / 2) * width
        bars = ax.bar(positions, means, width, label=name, color=palette[name],
                      hatch=("//" if "illegal" in name else ""), edgecolor="white")
        for rect, pool in zip(bars, pools):
            mean_t = rect.get_height()
            base = float(np.mean([r["t"] for r in pools[pool]["strategies"]["all-exact"]]))
            ax.annotate(f"{mean_t:.0f}\n(-{base - mean_t:.0f})",
                        (rect.get_x() + rect.get_width() / 2, mean_t),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    for xi, pool in zip(x, pools):
        base = float(np.mean([r["t"] for r in pools[pool]["strategies"]["all-exact"]]))
        ax.hlines(base, xi - 0.45, xi + 0.45, colors=COLOR["baseline"],
                  linestyles="--", lw=1.7, zorder=3)
        ax.annotate(f"all-exact baseline {base:.0f} T", (xi + 0.46, base),
                    va="center", fontsize=9, color=COLOR["baseline"])
    ax.set_xticks(x)
    ax.set_xticklabels([POOL_TITLES.get(p, p) for p in pools])
    ax.set_ylabel("Absolute T-count per circuit (mean)")
    ax.set_title("Shallow vs deep circuits: absolute T-count of every strategy", pad=12)
    _style(ax)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9)
    ax.set_ylim(0, max(
        float(np.mean([r["t"] for r in pools[p]["strategies"]["all-exact"]])) for p in pools
    ) * 1.30)
    fig.tight_layout()
    fig.text(0.5, -0.06,
             "labels: absolute T-count and (reduction vs all-exact); "
             "hatched = illegal mask; dashed line = all-exact baseline",
             ha="center", fontsize=8, color="#666666")
    return _save(fig, out_dir, "fig3_strategy_comparison", formats)


# ---------------------------------------------------------------------------
# consistency guard
# ---------------------------------------------------------------------------


def check_transcription() -> int:
    failures: List[str] = []
    report = (ROOT / "reports" / "strategy_b_experiments.md").read_text(encoding="utf-8")
    raw_bench = _load_json("verifier_bench.json")
    bench = _normalise_bench(raw_bench) if raw_bench else None
    if bench:
        for row in bench:
            recomputed = row["simulation_mean_s"] / row["symbolic_mean_s"]
            if abs(row["ratio_mean"] / recomputed - 1.0) > 0.02:
                failures.append(
                    f"{row['label']}: stored ratio {row['ratio_mean']:.0f}x vs "
                    f"{recomputed:.0f}x recomputed from the mean times"
                )
            ratio_text = f"{row['ratio_mean']:,.0f}"
            if ratio_text not in report:
                failures.append(
                    f"{row['label']}: speed-up {ratio_text} missing from the report"
                )
            if f"{row['simulation_mean_s']:.1f}" not in report:
                failures.append(
                    f"{row['label']}: mean simulation time "
                    f"{row['simulation_mean_s']:.1f}s missing from the report"
                )
    if failures:
        for item in failures:
            print(f"  [FAIL] {item}")
        return 1
    print(f"  [ok  ] figures and report agree "
          f"({'re-measured JSON' if bench else 'embedded fallback'})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the strategy-B figures (English).")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "figures"))
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--check", action="store_true",
                        help="only verify that the data matches the report")
    args = parser.parse_args(argv)
    if args.check:
        return check_transcription()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = Path(args.out_dir)
    written: List[Path] = []
    written += figure_speedup(out_dir, formats)
    written += figure_cliff(out_dir, formats)
    written += figure_strategies(out_dir, formats)
    status = check_transcription()
    print()
    for path in written:
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
