#!/usr/bin/env python3
"""Oracle vs greedy scatter: how close is the verifier-greedy decoder to optimal?

For every random small mixed circuit the *exhaustive* oracle is computed (every
one of the ``2**blocks`` downgrade masks is enumerated and checked with the
reduced verifier) and compared with what the deployed greedy decoder achieves on
the same circuit.  If the greedy were exactly optimal every point would sit on
``y = x``.

The script reports the honest metrics rather than only the picture:

* how many circuits are *exactly* optimal (the real headline),
* the mean / median / worst shortfall in T gates,
* R^2 of the fitted regression **and** R^2 against the ``y = x`` line,
* the shapes of the worst offenders.

Usage
-----
    .venv/bin/python oracle_vs_greedy.py --instances 300
    .venv/bin/python oracle_vs_greedy.py --instances 500 --out-dir reports/figures
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

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


MS = _load("mixed_stress_oracle", "mixed_stress.py")


# ---------------------------------------------------------------------------
# one circuit
# ---------------------------------------------------------------------------


def evaluate_circuit(instance, probs: Sequence[float]) -> Dict[str, Any]:
    verifier = instance.verifier
    blocks = int(instance.verifier.k)      # MixedInstance exposes counts, not 'blocks'
    tolerance = verifier.tolerance

    # exhaustive oracle: every mask, one shared partial-product sweep
    deviations = verifier.deviation_table()
    best_exact = instance.verifier.t_count(0)
    oracle_mask = 0
    for mask in range(1 << blocks):
        if deviations[mask] <= tolerance:
            value = instance.verifier.t_count(mask)
            if value < best_exact:
                best_exact = value
                oracle_mask = mask

    greedy_mask = verifier.greedy(probs)
    greedy_exact = instance.verifier.t_count(greedy_mask)

    base = instance.verifier.t_count(0)
    return {
        "blocks": int(blocks),
        "controls": list(int(c) for c in instance.counts),
        "all_exact": int(base),
        "oracle_saved": int(base - best_exact),
        "greedy_saved": int(base - greedy_exact),
        "oracle_mask": int(oracle_mask),
        "greedy_mask": int(greedy_mask),
    }


def make_instance(num_controls: Sequence[int], seed: int, h_prob: float):
    return MS.MixedInstance(num_controls, seed, h_prob)


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------


def plot_scatter(rows: Sequence[Dict[str, Any]], out_dir: Path, formats: Sequence[str]) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    oracle = np.array([r["oracle_saved"] for r in rows], dtype=float)
    greedy = np.array([r["greedy_saved"] for r in rows], dtype=float)
    exact = greedy == oracle
    gap = oracle - greedy
    total_oracle = float(oracle.sum())
    total_greedy = float(greedy.sum())
    capture = total_greedy / total_oracle if total_oracle > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    limit = max(oracle.max(), greedy.max()) + 3
    ax.plot([0, limit], [0, limit], "--", color="#8a8a8a", lw=1.5,
            label="y = x  (greedy exactly optimal)", zorder=1)
    ax.scatter(oracle[exact], greedy[exact], s=40, color="#2e7d32", alpha=0.80,
               edgecolors="white", linewidths=0.6,
               label=f"greedy optimal: {int(exact.sum())}/{len(rows)} "
                     f"({exact.mean():.1%})", zorder=3)
    if (~exact).any():
        ax.scatter(oracle[~exact], greedy[~exact], s=72, color="#c0392b",
                   marker="D", edgecolors="black", linewidths=0.6,
                   label=f"greedy suboptimal: {int((~exact).sum())} "
                         f"(worst short {int(gap.max())} T)", zorder=4)
        worst = np.argsort(gap)[::-1][:3]
        for idx in worst:
            if gap[idx] <= 0:
                continue
            ax.annotate(f"{int(gap[idx])} T short", (oracle[idx], greedy[idx]),
                        textcoords="offset points", xytext=(9, -10), fontsize=8,
                        color="#c0392b")
    ax.set_xlabel("Oracle saving (exhaustive $2^{N}$ search), T gates")
    ax.set_ylabel("Greedy saving (verifier-in-the-loop), T gates")
    ax.set_title("Oracle vs verifier-greedy T-gate saving\n"
                 f"{len(rows)} random mixed circuits, 5-12 blocks", pad=12)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(-0.8, limit)
    ax.set_ylim(-0.8, limit)
    ax.set_aspect("equal", adjustable="box")

    # statistics
    if len(rows) > 1 and oracle.std() > 0:
        slope, intercept = np.polyfit(oracle, greedy, 1)
        pred = slope * oracle + intercept
        ss_tot = float(((greedy - greedy.mean()) ** 2).sum())
        r2_fit = 1.0 - float(((greedy - pred) ** 2).sum()) / max(ss_tot, 1e-12)
        r2_line = 1.0 - float(((greedy - oracle) ** 2).sum()) / max(ss_tot, 1e-12)
    else:
        slope = intercept = r2_fit = r2_line = float("nan")
    text = (f"aggregate capture : {capture:.2%} of oracle T-saving\n"
            f"exactly optimal   : {exact.mean():.1%}  ({int(exact.sum())}/{len(rows)})\n"
            f"mean shortfall    : {gap.mean():.3f} T  (max {int(gap.max())})\n"
            f"fit               : y = {slope:.4f}x {intercept:+.4f}\n"
            f"R^2 (fit)         : {r2_fit:.4f}\n"
            f"R^2 (vs y = x)    : {r2_line:.4f}")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.6, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#cccccc"))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"fig4_oracle_vs_greedy.{fmt}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Oracle vs greedy scatter.")
    parser.add_argument("--instances", type=int, default=300)
    parser.add_argument("--min-blocks", type=int, default=5)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--control-values", default="2,3,4,5,6",
                        help="kept small so the exhaustive oracle stays cheap")
    parser.add_argument("--h-prob", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "figures"))
    parser.add_argument("--data", default=str(ROOT / "reports" / "figure_data" /
                                              "oracle_vs_greedy.json"))
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, _ = MS.load_policy(Path(args.checkpoint), device)
    control_values = [int(v) for v in args.control_values.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)
    seeds = np.random.SeedSequence(args.seed).generate_state(args.instances, dtype=np.uint32)

    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(args.instances):
        blocks = int(rng.integers(args.min_blocks, args.max_blocks + 1))
        counts = [int(rng.choice(control_values)) for _ in range(blocks)]
        instance = make_instance(counts, int(seeds[index]), args.h_prob)
        batch = MS.Batch.from_data_list(instance.blocks_data).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).tolist()
        row = evaluate_circuit(instance, probs)
        row["seed"] = int(seeds[index])
        rows.append(row)
        if (index + 1) % max(1, args.instances // 10) == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{index + 1}/{args.instances}] {elapsed:5.1f}s", flush=True)

    oracle = np.array([r["oracle_saved"] for r in rows], dtype=float)
    greedy = np.array([r["greedy_saved"] for r in rows], dtype=float)
    gap = oracle - greedy
    exact = gap == 0
    if oracle.std() > 0:
        slope, intercept = np.polyfit(oracle, greedy, 1)
        pred = slope * oracle + intercept
        ss_tot = float(((greedy - greedy.mean()) ** 2).sum())
        r2_fit = 1 - float(((greedy - pred) ** 2).sum()) / max(ss_tot, 1e-12)
        r2_line = 1 - float(((greedy - oracle) ** 2).sum()) / max(ss_tot, 1e-12)
    else:
        slope = intercept = r2_fit = r2_line = float("nan")

    print()
    print("=" * 74)
    print(f"  Oracle vs greedy - {len(rows)} random mixed circuits "
          f"({args.min_blocks}..{args.max_blocks} blocks)")
    print("=" * 74)
    print(f"  exactly optimal (y == x) : {int(exact.sum())}/{len(rows)} = {exact.mean():.1%}")
    print(f"  mean shortfall            : {gap.mean():.4f} T   "
          f"(median {np.median(gap):.1f}, max {gap.max():.0f})")
    print(f"  regression                : y = {slope:.4f}x {intercept:+.4f}   "
          f"R^2 = {r2_fit:.5f}")
    print(f"  R^2 against y = x         : {r2_line:.5f}")
    capture = greedy.sum() / oracle.sum() if oracle.sum() > 0 else float("nan")
    print(f"  aggregate capture        : {capture:.4%} of oracle T-saving")
    worst = sorted(rows, key=lambda r: (r["oracle_saved"] - r["greedy_saved"]), reverse=True)[:5]
    if any(r["oracle_saved"] > r["greedy_saved"] for r in worst):
        print("  worst cases:")
        for row in worst:
            if row["oracle_saved"] <= row["greedy_saved"]:
                continue
            print(f"    blocks={row['blocks']:>2} controls={row['controls']} "
                  f"oracle={row['oracle_saved']} greedy={row['greedy_saved']} "
                  f"(short {row['oracle_saved'] - row['greedy_saved']} T)")
    print("=" * 74)

    data_path = Path(args.data)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"  wrote {data_path}")
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for path in plot_scatter(rows, Path(args.out_dir), formats):
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
