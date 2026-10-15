#!/usr/bin/env python3
"""Ablation on greedy orderings: is the learned score worth anything?

The greedy decoder is

    for block in ORDER:
        if safe(accepted | {block}):  accepted |= {block}

so the *ordering* fully determines the result.  This script pits the frozen
policy ordering against three zero-cost static rules --

    policy          descending learned probability            (costs a forward pass)
    reverse_index   idx = K-1 .. 0                             (free: no downstream H)
    saving_desc     per-block T saving desc, ties from the end (free)
    random          shuffled                                   (free, averaged)

on two safety oracles:

  * the EXACT reduced verifier (non-monotone, width <= ~12) -- ordering matters,
  * the support/diagonal TRACKER (monotone, any width)      -- ordering is provably
    irrelevant when the query budget is unlimited, so the interesting question is
    the *budgeted* regime.

Query-budgeted greedy spends at most ``B`` oracle calls per circuit (the first B
blocks of the ordering) and asks: with a tight verification budget, does the
learned ordering actually buy more T saving than free heuristics?

Usage
-----
    .venv/bin/python ablation_ordering.py
    .venv/bin/python ablation_ordering.py --a1-circuits 500 --deep-circuits 20
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


ULT = _load("abl_ult", "ultimate_scalability_test.py")
MS = ULT.MS
GEN = ULT.GEN
ST = _load("abl_tracker", "support_tracker.py")
WIDE = _load("abl_wide", "wide_scale_stress.py")

ORDER_NAMES = ("policy", "reverse_index", "saving_desc", "random")


def make_orderings(k: int, probs: Sequence[float], deltas: Sequence[int],
                   rng: random.Random, random_trials: int):
    """Return ({name: order}, [shuffled orders]) for one circuit."""
    fixed = {
        "policy": sorted(range(k), key=lambda i: -probs[i]),
        "reverse_index": list(range(k - 1, -1, -1)),
        "saving_desc": sorted(range(k), key=lambda i: (-deltas[i], -i)),
    }
    randoms = [rng.sample(range(k), k) for _ in range(random_trials)]
    return fixed, randoms


def deltas_for(counts: Sequence[int]) -> List[int]:
    out = []
    for n in counts:
        t_exact, t_relative = GEN.block_t_counts(int(n))
        out.append(int(t_exact - t_relative))
    return out


def greedy_exact(verifier, order: Sequence[int], budget: Optional[int] = None):
    accepted = 0
    calls = 0
    for block in order:
        if budget is not None and calls >= budget:
            break
        calls += 1
        candidate = accepted | (1 << block)
        if verifier.is_safe(candidate):
            accepted = candidate
    return accepted, calls


def greedy_tracker(schedule, order: Sequence[int], budget: Optional[int] = None):
    accepted = 0
    calls = 0
    for block in order:
        if budget is not None and calls >= budget:
            break
        calls += 1
        candidate = accepted | (1 << block)
        if ST.track_safe(schedule, candidate):
            accepted = candidate
    return accepted, calls


def _summarise(per_ordering: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for name, values in per_ordering.items():
        arr = np.array(values, dtype=float)
        out[name] = {"mean_saved": float(arr.mean()), "total_saved": float(arr.sum())}
    return out


def run_exact_short(model, device, data_path: Path, limit: int, budgets: Sequence[int],
                    random_trials: int, seed: int):
    rows = json.loads(data_path.read_text(encoding="utf-8"))[:limit]
    rng = random.Random(seed)
    unbudgeted: Dict[str, List[float]] = {n: [] for n in ORDER_NAMES}
    budgeted: Dict[int, Dict[str, List[float]]] = {b: {n: [] for n in ORDER_NAMES}
                                                   for b in budgets}
    for row in rows:
        instance = MS.MixedInstance(row["controls"], row["seed"], 0.7)
        verifier = instance.verifier
        batch = MS.Batch.from_data_list(instance.blocks_data).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).tolist()
        counts = [int(c) for c in instance.counts]
        deltas = deltas_for(counts)
        k = len(counts)
        base = verifier.t_count(0)
        fixed, randoms = make_orderings(k, probs, deltas, rng, random_trials)
        for name, order in fixed.items():
            mask, _ = greedy_exact(verifier, order)
            unbudgeted[name].append(base - verifier.t_count(mask))
        unbudgeted["random"].append(
            float(np.mean([base - verifier.t_count(greedy_exact(verifier, o)[0])
                           for o in randoms])))
        for budget in budgets:
            for name, order in fixed.items():
                mask, _ = greedy_exact(verifier, order, budget)
                budgeted[budget][name].append(base - verifier.t_count(mask))
            budgeted[budget]["random"].append(
                float(np.mean([base - verifier.t_count(greedy_exact(verifier, o, budget)[0])
                               for o in randoms])))
    return {"circuits": len(rows), "oracle": "exact reduced verifier",
            "unbudgeted": _summarise(unbudgeted),
            "budgeted": {str(b): _summarise(v) for b, v in budgeted.items()}}


def _deep_probs(model, device, blocks_data):
    batch = MS.Batch.from_data_list(blocks_data).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(batch)).tolist()


def run_deep(model, device, circuits: int, seed: int, budgets: Sequence[int],
             random_trials: int, oracle: str, control_choices: Sequence[int],
             h_prob: float, width: Optional[int] = None):
    rng = random.Random(seed)
    unbudgeted: Dict[str, List[float]] = {n: [] for n in ORDER_NAMES}
    budgeted: Dict[int, Dict[str, List[float]]] = {b: {n: [] for n in ORDER_NAMES}
                                                   for b in budgets}
    blocks_seen: List[int] = []
    for _ in range(circuits):
        blocks = rng.randint(30, 50)
        circuit_seed = rng.randrange(1 << 30)
        if oracle == "tracker":
            instance = ULT.WideInstance(width=int(width), blocks=blocks,
                                        seed=circuit_seed, h_prob=h_prob)
            schedule = WIDE.build_schedule(instance)
            counts = list(instance.control_counts)
            base = int(instance.t_all_exact)
            probs = _deep_probs(model, device, instance.policy_inputs())
            decode = lambda order, budget=None: greedy_tracker(schedule, order, budget)
            cost = instance.t_count
        else:
            counts = [int(rng.choice(list(control_choices))) for _ in range(blocks)]
            instance = MS.MixedInstance(counts, circuit_seed, h_prob)
            verifier = instance.verifier
            base = verifier.t_count(0)
            probs = _deep_probs(model, device, instance.blocks_data)
            decode = lambda order, budget=None: greedy_exact(verifier, order, budget)
            cost = verifier.t_count
        blocks_seen.append(blocks)
        deltas = deltas_for(counts)
        k = len(counts)
        fixed, randoms = make_orderings(k, probs, deltas, rng, random_trials)
        for name, order in fixed.items():
            mask, _ = decode(order)
            unbudgeted[name].append(base - cost(mask))
        unbudgeted["random"].append(
            float(np.mean([base - cost(decode(o)[0]) for o in randoms])))
        for budget in budgets:
            for name, order in fixed.items():
                mask, _ = decode(order, budget)
                budgeted[budget][name].append(base - cost(mask))
            budgeted[budget]["random"].append(
                float(np.mean([base - cost(decode(o, budget)[0]) for o in randoms])))
    return {"circuits": circuits, "oracle": oracle, "h_prob": h_prob, "width": width,
            "mean_blocks": float(np.mean(blocks_seen)),
            "unbudgeted": _summarise(unbudgeted),
            "budgeted": {str(b): _summarise(v) for b, v in budgeted.items()}}


def _print_block(title: str, result: Dict[str, Any]) -> None:
    print("=" * 86)
    print(f"  {title}")
    print("=" * 86)
    extra = ""
    if result.get("width"):
        extra += f"  h_prob={result['h_prob']:g}  width={result['width']}"
    if "mean_blocks" in result:
        extra += f"  mean_blocks={result['mean_blocks']:.1f}"
    print(f"  circuits={result['circuits']}  oracle={result['oracle']}{extra}")

    def line(tag: str, table: Dict[str, Dict[str, float]]) -> None:
        policy = table["policy"]["mean_saved"]
        best_static = max(table[n]["mean_saved"]
                          for n in ("reverse_index", "saving_desc", "random"))
        cells = " | ".join(f"{n}={table[n]['mean_saved']:7.2f}" for n in ORDER_NAMES)
        print(f"  {tag:>11} | {cells}  | policy-beststatic {policy - best_static:+7.2f} T")

    line("no budget", result["unbudgeted"])
    for budget in sorted(result["budgeted"], key=int):
        line(f"B={budget}", result["budgeted"][budget])
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Greedy-ordering ablation.")
    parser.add_argument("--a1-data", default=str(ROOT / "reports" / "figure_data" /
                                                "oracle_vs_greedy.json"))
    parser.add_argument("--a1-circuits", type=int, default=500)
    parser.add_argument("--deep-circuits", type=int, default=20)
    parser.add_argument("--wide-circuits", type=int, default=20)
    parser.add_argument("--widths", default="24,50")
    parser.add_argument("--budgets", default="1,3,5,10")
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "ablation_ordering.json"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.a1_circuits = 30
        args.deep_circuits = 3
        args.wide_circuits = 3
        args.widths = "24"

    device = torch.device(args.device)
    model, _ = MS.load_policy(Path(args.checkpoint), device)
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    results: Dict[str, Any] = {"config": {k: v for k, v in vars(args).items()}}

    a1 = run_exact_short(model, device, Path(args.a1_data), args.a1_circuits,
                         budgets, args.random_trials, args.seed)
    results["a1_exact_short"] = a1
    _print_block("A1  exact verifier, mixed circuits (5-12 blocks)", a1)

    a2 = run_deep(model, device, args.deep_circuits, args.seed + 1, budgets,
                  args.random_trials, oracle="exact",
                  control_choices=(2, 3, 4), h_prob=0.5)
    results["a2_exact_deep"] = a2
    _print_block("A2  exact verifier, DEEP circuits (30-50 blocks, C^2X..C^4X)", a2)

    results["b_tracker_wide"] = {}
    for width in widths:
        for h_prob in (0.1, 0.5):
            res = run_deep(model, device, args.wide_circuits, args.seed + 2, budgets,
                           args.random_trials, oracle="tracker",
                           control_choices=(2, 3, 4, 5, 6, 7, 8), h_prob=h_prob, width=width)
            results["b_tracker_wide"][f"w{width}_h{h_prob:g}"] = res
            _print_block(f"B  tracker, wide {width}q deep circuits (30-50 blocks), h={h_prob:g}",
                         res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
