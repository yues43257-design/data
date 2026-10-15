#!/usr/bin/env python3
"""Beyond-verifier-scale stress test: relative-phase downgrades at 24-100 qubits.

Why this exists
---------------
The exact reduced verifier decides safety by comparing Z-basis measurement
distributions, i.e. it needs a ``2**N x 2**N`` object at width ``N``.  Past
roughly 12-19 data qubits that is out of memory, so *no* exact oracle exists at
24 / 32 / 50 / 100 qubits.  The only sound criterion that survives is the
polynomial-time **support/diagonality tracker** (``support_tracker.py``,
``O(gates)`` time, ``O(qubits)`` space, proven sound at small width by its
self-test).

This script measures, at widths the verifier cannot touch, what four decoding
strategies achieve under that tracker:

    RL one-shot (raw)            threshold the frozen policy's own probabilities
    RL one-shot + tracker repair drop least-confident downgrades until track-safe
    RL depth-cap + tracker       depth-conditioned top-k, then tracker repair
    tracker greedy               score-ordered greedy with the tracker in the loop

Everything is measured on genuinely wide circuits: every block keeps a native
control count in {2..8} (the shapes the policy was trained on) but is *placed* on
a random window of the wide register, and the inter-block Clifford layers act on
all data qubits.

On the "remap local wires" question
-----------------------------------
It is **not** needed.  ``generate_variant_dataset._block_graph(n)`` is the cached
canonical ZX diagram of a C^nX block whose qubit indices are local to the block
(``0..n+1``); the node features normalise by the block's own width, and
``context_features`` carries no global-width term.  A block therefore scores
identically wherever it sits in the wide register -- ``policy_inputs()`` builds
the very same tensors the policy was trained on.

Honest scope
------------
Safety here means "the accumulated correction provably stays diagonal"
(sufficient, sound, conservative).  It is *not* the exact measurement-statistic
criterion, which is uncomputable at these widths; a mask this script rejects may
still be exactly safe.  All reported savings are therefore a **lower bound**.

Usage
-----
    .venv/bin/python wide_scale_stress.py --quick
    .venv/bin/python wide_scale_stress.py --widths 24,32,50,100 \
        --h-probs 0.0,0.1,0.5 --circuits 20
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


ULT = _load("wide_ult", "ultimate_scalability_test.py")
MS = ULT.MS
ST = _load("wide_tracker", "support_tracker.py")


def build_schedule(instance) -> Dict[str, Any]:
    """Block-level schedule consumed by ``support_tracker.track_safe``."""
    layers: List[List[Tuple[str, List[int]]]] = []
    for layer in instance.layers:
        if layer is None:
            layers.append([])
            continue
        layers.append(
            [
                (ins.operation.name, [layer.find_bit(q).index for q in ins.qubits])
                for ins in layer.data
            ]
        )
    blocks = [
        {"controls": list(c), "ancillas": list(a), "target": int(t)}
        for c, a, t in zip(instance.controls, instance.ancillas, instance.targets)
    ]
    return {"total": instance.total, "layers": layers, "blocks": blocks}


def tracker_repair(schedule: Dict[str, Any], probs: Sequence[float], mask: int):
    """Drop least-confident downgrades until the tracker certifies the mask."""
    if ST.track_safe(schedule, mask):
        return int(mask), 0
    drops = 0
    order = sorted(
        (i for i in range(len(schedule["blocks"])) if (mask >> i) & 1),
        key=lambda i: probs[i],
    )
    for block in order:
        mask &= ~(1 << block)
        drops += 1
        if ST.track_safe(schedule, mask):
            return int(mask), drops
    return 0, drops


def evaluate(instance, schedule: Dict[str, Any], probs: Sequence[float]) -> Dict[str, Any]:
    blocks = int(instance.blocks)
    base = int(instance.t_all_exact)

    def saved(mask: int) -> int:
        return int(base - instance.t_count(mask))

    def count(mask: int) -> int:
        return int(bin(mask).count("1"))

    started = time.perf_counter()
    raw = ST.mask_from_probs(probs, 0.5)
    raw_safe = ST.track_safe(schedule, raw)
    t_raw = time.perf_counter() - started

    started = time.perf_counter()
    repaired, drops = tracker_repair(schedule, probs, raw)
    t_repair = time.perf_counter() - started

    cap = MS.top_mask(probs, MS.depth_budget(blocks))
    started = time.perf_counter()
    cap_mask, cap_drops = tracker_repair(schedule, probs, cap)
    t_cap = time.perf_counter() - started

    started = time.perf_counter()
    greedy = ST.tracker_greedy(schedule, probs)
    t_greedy = time.perf_counter() - started

    all_rel = (1 << blocks) - 1
    all_rel_safe = ST.track_safe(schedule, all_rel)

    return {
        "blocks": blocks,
        "control_hist": {int(n): int(instance.control_counts.count(n))
                         for n in sorted(set(instance.control_counts))},
        "t_all_exact": base,
        "t_all_relative": int(instance.t_all_relative),
        "oneshot_downgrades": count(raw),
        "oneshot_safe": bool(raw_safe),
        "oneshot_saved": saved(raw),
        "oneshot_seconds": t_raw,
        "repair_downgrades": count(repaired),
        "repair_saved": saved(repaired),
        "repair_drops": int(drops),
        "repair_seconds": t_repair,
        "depthcap_blocks": int(MS.depth_budget(blocks)),
        "depthcap_downgrades": count(cap_mask),
        "depthcap_saved": saved(cap_mask),
        "depthcap_seconds": t_cap,
        "greedy_downgrades": count(greedy),
        "greedy_saved": saved(greedy),
        "greedy_seconds": t_greedy,
        "all_relative_safe": bool(all_rel_safe),
        "all_relative_saved": int(base - instance.t_all_relative),
    }


def verifier_bytes(width: int) -> int:
    """Memory the exact reduced verifier would need at this width (complex128)."""
    return (1 << (2 * int(width))) * 16


def _fmt_bytes(n: int) -> str:
    for unit, scale in (("PB", 2 ** 50), ("TB", 2 ** 40), ("GB", 2 ** 30), ("MB", 2 ** 20)):
        if n >= scale:
            value = n / scale
            return f"{value:.1f} {unit}" if value < 1.0e4 else f"{value:.2e} {unit}"
    return f"{n} B"


def run(args) -> int:
    device = torch.device(args.device)
    model, _ = MS.load_policy(Path(args.checkpoint), device)
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]

    print("=" * 92)
    print("  Beyond-verifier-scale stress test  (tracker-only safety, no 2^N object)")
    print("=" * 92)
    print(f"  widths        : {widths} data qubits")
    print(f"  h_prob        : {h_probs}")
    print(f"  circuits/cell : {args.circuits}   blocks in [{args.min_blocks}, {args.max_blocks}]")
    print(f"  policy        : {Path(args.checkpoint).name}  (frozen, per-block, width-agnostic)")
    for w in widths:
        print(f"  verifier @ {w:>3}q : 2^{w} dense operator = {_fmt_bytes(verifier_bytes(w))}"
              f"  ->  OOM (2^N)")
    print("-" * 92)

    all_rows: List[Dict[str, Any]] = []
    cells: List[Dict[str, Any]] = []
    rng = random.Random(args.seed)

    for width in widths:
        for h_prob in h_probs:
            rows: List[Dict[str, Any]] = []
            for index in range(args.circuits):
                seed = rng.randrange(1 << 30)
                blocks = rng.randint(args.min_blocks, args.max_blocks)
                instance = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                            h_prob=h_prob)
                schedule = build_schedule(instance)
                batch = MS.Batch.from_data_list(instance.policy_inputs()).to(device)
                with torch.no_grad():
                    probs = torch.sigmoid(model(batch)).tolist()
                row = evaluate(instance, schedule, probs)
                row.update({"width": width, "h_prob": h_prob, "seed": seed})
                rows.append(row)
                all_rows.append(row)

            def mean(key: str) -> float:
                return float(np.mean([r[key] for r in rows]))

            cell = {
                "width": width,
                "h_prob": h_prob,
                "circuits": len(rows),
                "verifier": "OOM (2^N)",
                "verifier_bytes": verifier_bytes(width),
                "mean_blocks": mean("blocks"),
                "oneshot_safe_rate": float(np.mean([r["oneshot_safe"] for r in rows])),
                "oneshot_saved": mean("oneshot_saved"),
                "oneshot_seconds": mean("oneshot_seconds"),
                "repair_saved": mean("repair_saved"),
                "repair_drops": mean("repair_drops"),
                "repair_seconds": mean("repair_seconds"),
                "depthcap_saved": mean("depthcap_saved"),
                "depthcap_seconds": mean("depthcap_seconds"),
                "greedy_saved": mean("greedy_saved"),
                "greedy_seconds": mean("greedy_seconds"),
                "all_relative_safe_rate": float(np.mean([r["all_relative_safe"] for r in rows])),
                "mean_all_exact": mean("t_all_exact"),
            }
            cells.append(cell)
            print(f"  [{width:>3}q h={h_prob:g}] blocks~{cell['mean_blocks']:.0f} "
                  f"| one-shot safe {cell['oneshot_safe_rate']:5.0%} "
                  f"save {cell['oneshot_saved']:6.1f} T | repair save "
                  f"{cell['repair_saved']:6.1f} T ({cell['repair_seconds'] * 1e3:.1f} ms) "
                  f"| greedy save {cell['greedy_saved']:6.1f} T "
                  f"({cell['greedy_seconds'] * 1e3:.1f} ms)", flush=True)

    print("-" * 92)
    print("| width | h_prob | one-shot safe | one-shot T | +repair T | depth-cap T "
          "| greedy T | greedy ms |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        print(f"| {cell['width']} | {cell['h_prob']:g} | {cell['oneshot_safe_rate']:.0%} "
              f"| {cell['oneshot_saved']:.1f} | {cell['repair_saved']:.1f} "
              f"| {cell['depthcap_saved']:.1f} | {cell['greedy_saved']:.1f} "
              f"| {cell['greedy_seconds'] * 1e3:.2f} |")
    print("-" * 92)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "widths": widths,
            "h_probs": h_probs,
            "circuits": args.circuits,
            "min_blocks": args.min_blocks,
            "max_blocks": args.max_blocks,
            "seed": args.seed,
            "checkpoint": Path(args.checkpoint).name,
            "safety_criterion": "support_tracker (diagonal correction, sound)",
        },
        "cells": cells,
        "rows": all_rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {len(all_rows)} circuits)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Beyond-verifier-scale stress test.")
    parser.add_argument("--widths", default="24,32,50,100")
    parser.add_argument("--h-probs", default="0.0,0.1,0.5")
    parser.add_argument("--circuits", type=int, default=20)
    parser.add_argument("--min-blocks", type=int, default=20)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "wide_scale_stress.json"))
    parser.add_argument("--quick", action="store_true",
                        help="1 circuit/cell, 2 widths (smoke test)")
    parser.add_argument("--self-test", action="store_true",
                        help="validate tracker soundness in the wide layout")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.quick:
        args.widths = "24,50"
        args.circuits = 1
        args.h_probs = "0.0,0.5"
    return run(args)


def _self_test() -> int:
    """Soundness of ``track_safe`` in the *wide* layout, against exact 2^N tensors.

    The stock ``support_tracker`` self-test only covers the homogeneous layout;
    this one rebuilds the check on ``WideInstance`` circuits (blocks on random
    windows, layers on all data qubits, ancilla path exercised).
    """
    from qiskit.quantum_info import Operator

    cases = [
        (5, 2, (2, 3), 5, 1, 0.7),
        (5, 2, (2, 3), 5, 2, 0.1),
        (6, 2, (2, 3, 4), 4, 3, 1.0),
        (5, 2, (2, 3), 6, 7, 0.3),
    ]
    total_masks = accepted_total = unsound = 0
    for width, pool, controls, blocks, seed, h_prob in cases:
        instance = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                    h_prob=h_prob, ancilla_pool=pool,
                                    control_choices=controls)
        schedule = build_schedule(instance)
        exact = Operator(instance.circuit(0)).data
        exact_dag = exact.conj().T
        bad = accepted = 0
        for mask in range(1 << blocks):
            correction = Operator(instance.circuit(mask)).data @ exact_dag
            off = float(np.abs(correction - np.diag(np.diag(correction))).max())
            tracked = ST.track_safe(schedule, mask)
            accepted += int(tracked)
            total_masks += 1
            if tracked and off > 1e-9:
                bad += 1
        accepted_total += accepted
        unsound += bad
        print(f"  [{'ok  ' if bad == 0 else 'FAIL'}] w={width} pool={pool} ctrl={controls} "
              f"b={blocks} h={h_prob} seed={seed}: {1 << blocks} masks, "
              f"accepted={accepted}, unsound={bad}")
    print()
    if unsound:
        print(f"wide self-test FAILED: {unsound}/{total_masks} unsound acceptances")
        return 1
    print(f"wide self-test passed: tracker sound on {total_masks} masks "
          f"(accepted {accepted_total}); exact 2^N tensors, wide layout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
