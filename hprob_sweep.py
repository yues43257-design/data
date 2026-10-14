#!/usr/bin/env python3
"""Sweep the mixed large-circuit stress test over the inter-block Hadamard density.

``mixed_stress.run_stress`` draws the circuit *shapes* (block count, the per-block
control counts, the instance seeds) from ``numpy.default_rng(seed)`` and uses
``h_prob`` only when it lays down the inter-block Clifford layers.  Holding the
seed and the number of circuits fixed while varying ``h_prob`` therefore gives a
clean controlled ablation: every row of the table is measured on the *same*
circuits, only the Hadamard density inside the layers changes.

The Markdown table goes to stdout, progress goes to stderr, so

    .venv/bin/python hprob_sweep.py > sweep.md

produces a clean report file.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_H_PROBS = (0.0, 0.1, 0.3, 0.5, 0.8)


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


MS = _load("mixed_stress_sweep", "mixed_stress.py")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def _capture_rate(summary: Dict[str, Any]) -> float:
    headroom = summary["mean_exact_t"] - summary["mean_relative_t"]
    if abs(headroom) < 1e-9:
        return float("nan")
    return summary["greedy_saving"] / headroom


def format_markdown(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("### 混合大电路压力测试 —— h_prob 扫描")
    lines.append("")
    lines.append(
        f"> **受控对照**：seed = `{config['seed']}`，每档 {config['num_tests']} 条电路，"
        f"区块数 {config['min_blocks']}–{config['max_blocks']}，各区块控制数独立取自 "
        f"`{config['controls']}`；仅层内 Hadamard 密度 `h_prob` 变化，"
        f"电路形状与控制数序列完全一致。"
    )
    lines.append(f">")
    lines.append(f"> 权重：`{Path(config['checkpoint']).name}` ｜ 设备：`{config['device']}`")
    lines.append("")
    lines.append("| h_prob | All-Exact T-count | Greedy Saved T | Greedy Capture Rate | One-shot Saved T |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['h_prob']:.2f} "
            f"| {row['exact']:.1f} "
            f"| {row['greedy_saved']:.1f} "
            f"| {row['greedy_capture']:.1%} "
            f"| {row['one_shot_saved']:.1f} |"
        )
    lines.append("")
    lines.append("*列说明：`All-Exact T-count` = 不做任何降级时每条电路的平均 T 门数；"
                 "`Greedy Saved T` = 验证器贪心相对全精确省下的 T 门数；"
                 "`Greedy Capture Rate` = 贪心节省 / 全相对（合法上界）节省；"
                 "`One-shot Saved T` = RL 一次性掩码经**修复**（逐个丢弃最不自信的降级，直到安全）后的实际节省；"
                 "纯回退（旧行为）的数值见下方诊断表。*")
    lines.append("")
    lines.append("<details><summary>安全性诊断（点击展开）</summary>")
    lines.append("")
    lines.append("| h_prob | 全相对安全率 | 贪心安全率 | 原始阈值安全率 | 修复比例 | 一次性(纯回退) 省 T | 用时(s) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['h_prob']:.2f} "
            f"| {row['relative_safe']:.0%} "
            f"| {row['greedy_safe']:.0%} "
            f"| {row['one_shot_safe']:.0%} "
            f"| {row['one_shot_repair_rate']:.0%} "
            f"| {row['one_shot_fallback_saved']:.1f} "
            f"| {row['seconds']:.0f} |"
        )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _parse_h_probs(text: str) -> List[float]:
    values = [float(part) for part in text.replace(" ", "").split(",") if part]
    if not values:
        raise ValueError("empty h_prob list")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the mixed stress test over the inter-block Hadamard density."
    )
    parser.add_argument("--h-probs", default=",".join(f"{v:g}" for v in DEFAULT_H_PROBS),
                        help="comma-separated h_prob values to sweep")
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--min-blocks", type=int, default=30)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="also write the Markdown table here")
    parser.add_argument("--data-dir", default=str(ROOT / "reports" / "figure_data"),
                        help="where the per-circuit JSON used by plot_results.py is written")
    parser.add_argument("--verbose", action="store_true",
                        help="print the full per-circuit battle report to stderr")
    args = parser.parse_args(argv)

    h_probs = _parse_h_probs(args.h_probs)
    device = torch.device(args.device)
    model, checkpoint = MS.load_policy(Path(args.checkpoint), device)
    print(
        f"[sweep] checkpoint={Path(args.checkpoint).name} "
        f"(epoch={checkpoint.get('epoch')}) device={args.device}",
        file=sys.stderr,
    )
    print(
        f"[sweep] h_prob={h_probs} | tests={args.num_tests} | "
        f"blocks={args.min_blocks}..{args.max_blocks} | seed={args.seed}",
        file=sys.stderr,
    )

    results: List[Dict[str, Any]] = []
    per_circuit: Dict[str, List[Dict[str, Any]]] = {}
    started = time.perf_counter()
    for index, h_prob in enumerate(h_probs, start=1):
        if args.verbose:
            log = lambda message: print(message, file=sys.stderr, flush=True)
        else:
            log = lambda message: None
        step = time.perf_counter()
        outcome = MS.run_stress(
            model,
            device,
            num_tests=args.num_tests,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed,
            h_prob=h_prob,
            threshold=args.threshold,
            log=log,
        )
        summary = outcome["summary"]
        per_circuit[str(h_prob)] = outcome["rows"]
        elapsed = time.perf_counter() - step
        results.append(
            {
                "h_prob": h_prob,
                "exact": summary["mean_exact_t"],
                "greedy_saved": summary["greedy_saving"],
                "greedy_capture": _capture_rate(summary),
                "one_shot_saved": summary["one_shot_saving"],
                "one_shot_fallback_saved": summary["one_shot_fallback_saving"],
                "one_shot_repair_rate": summary["one_shot_repair_rate"],
                "relative_safe": summary["relative_safe_rate"],
                "greedy_safe": summary["greedy_safe_rate"],
                "one_shot_safe": summary["one_shot_safe_rate"],
                "seconds": elapsed,
            }
        )
        print(
            f"[sweep] {index}/{len(h_probs)}  h_prob={h_prob:g}  "
            f"greedy saves {summary['greedy_saving']:.1f} T "
            f"({_capture_rate(summary):.1%} of legal max)  [{elapsed:.0f}s]",
            file=sys.stderr,
        )

    config = {
        "seed": args.seed,
        "num_tests": args.num_tests,
        "min_blocks": args.min_blocks,
        "max_blocks": args.max_blocks,
        "controls": list(getattr(MS, "CONTROL_CHOICES", (2, 3, 4, 5, 6, 7, 8))),
        "checkpoint": args.checkpoint,
        "device": args.device,
    }
    table = format_markdown(results, config)
    print(table)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dump_path = data_dir / "hprob_sweep.json"
    with open(dump_path, "w", encoding="utf-8") as handle:
        json.dump({"config": config, "summary": results, "per_circuit": per_circuit},
                  handle, indent=2)
    print(f"[sweep] per-circuit data written to {dump_path}", file=sys.stderr)
    print(f"\n<!-- 总用时 {time.perf_counter() - started:.0f}s -->")
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"[sweep] written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
