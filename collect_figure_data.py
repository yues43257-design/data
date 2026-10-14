#!/usr/bin/env python3
"""Re-measure the numbers behind the figures and dump them as JSON.

Two blocks of data are produced under ``reports/figure_data/``:

* ``verifier_bench.json`` - high-precision wall-clock timings for the two
  verifier benchmarks quoted in the report (section 5.1).  Re-measured here with
  ``time.perf_counter`` so the times *and* the speed-up ratios come from the same
  run and are therefore self-consistent (the published pair was a rounded log
  line, which is what made the two documents disagree).
* ``strategy_pools.json`` - absolute T-counts (not just savings) for every
  strategy on the held-out shallow/deep pools of section 4.2.

The H-density sweep (section 5.3, with per-circuit records) is produced by
``hprob_sweep.py --data-dir reports/figure_data``.

Usage
-----
    .venv/bin/python collect_figure_data.py --benchmarks        # ~10 min
    .venv/bin/python collect_figure_data.py --pools             # ~2 min
    .venv/bin/python collect_figure_data.py --benchmarks --pools
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


GEN = _load("variant_generator_collect", "generate_variant_dataset.py")
SYM = _load("variant_symbolic_collect", "symbolic_verifier.py")
MS = _load("mixed_stress_collect", "mixed_stress.py")
RL = _load("rl_trainer_collect", "train_variant_policy_rl.py")


# ---------------------------------------------------------------------------
# 5.1 - verifier benchmarks, high-precision timings
# ---------------------------------------------------------------------------


def _simulated_oracle_seconds(num_controls: int, blocks: int, seed: int, h_prob: float) -> float:
    """Full-simulation oracle enumeration (the slow baseline)."""
    reference = GEN.build_instance(num_controls, blocks, 0, seed, h_prob)[0]
    reference_dist = GEN.measurement_distributions(reference, num_controls)
    started = time.perf_counter()
    for mask in range(1 << blocks):
        circuit = GEN.build_instance(num_controls, blocks, mask, seed, h_prob)[0]
        dist = GEN.measurement_distributions(circuit, num_controls)
        max(GEN.total_variation(reference_dist[b], dist[b]) for b in reference_dist)
    return time.perf_counter() - started


def _summarise(key: str, label: str, detail: str,
               sim_runs: Sequence[float], sym_runs: Sequence[float]) -> Dict[str, Any]:
    """Mean/std over repeats for both sides, and for the speed-up ratio itself."""
    sim = np.asarray(sim_runs, dtype=float)
    sym = np.asarray(sym_runs, dtype=float)
    ratios = sim / np.maximum(sym, 1e-12)
    ddof = 1 if len(sim) > 1 else 0

    def _std(values: np.ndarray) -> float:
        return float(values.std(ddof=ddof)) if len(values) > 1 else 0.0

    return {
        "key": key,
        "label": label,
        "detail": detail,
        "repeats": int(len(sim)),
        "simulation_mean_s": float(sim.mean()),
        "simulation_std_s": _std(sim),
        "simulation_runs_s": [float(x) for x in sim],
        "symbolic_mean_s": float(sym.mean()),
        "symbolic_std_s": _std(sym),
        "symbolic_runs_s": [float(x) for x in sym],
        "ratio_mean": float(ratios.mean()),
        "ratio_std": _std(ratios),
        "ratio_runs": [float(x) for x in ratios],
    }


def measure_benchmarks(repeats: int = 3) -> List[Dict[str, Any]]:
    """Each benchmark is run ``repeats`` times; both sides are timed per run."""
    rows: List[Dict[str, Any]] = []
    repeats = max(1, int(repeats))

    # (a) oracle enumeration on a homogeneous C^6X x4 instance
    num_controls, blocks, seed, h_prob = 6, 4, 7, 0.7
    print(f"  [bench] oracle enumeration (C^6X x4) x{repeats} - "
          f"each run simulates {1 << blocks} masks, this is the slow one ...", flush=True)
    sim_runs: List[float] = []
    sym_runs: List[float] = []
    for run in range(repeats):
        sim_runs.append(_simulated_oracle_seconds(num_controls, blocks, seed, h_prob))
        started = time.perf_counter()
        verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks, seed, h_prob)
        verifier.deviation_table()
        sym_runs.append(time.perf_counter() - started)
        print(f"    run {run + 1}/{repeats}: simulation={sim_runs[-1]:.3f}s "
              f"symbolic={sym_runs[-1]:.5f}s "
              f"({sim_runs[-1] / max(sym_runs[-1], 1e-12):,.0f}x)", flush=True)
    rows.append(_summarise("oracle", "C^6X x4 oracle enumeration",
                           f"2^{blocks} = {1 << blocks} masks", sim_runs, sym_runs))
    print(f"    -> simulation {rows[-1]['simulation_mean_s']:.3f} +/- "
          f"{rows[-1]['simulation_std_s']:.3f}s | symbolic "
          f"{rows[-1]['symbolic_mean_s']:.5f} +/- {rows[-1]['symbolic_std_s']:.5f}s | "
          f"{rows[-1]['ratio_mean']:,.0f} +/- {rows[-1]['ratio_std']:,.0f}x", flush=True)

    # (b) greedy decode on a 30-block instance
    num_controls, blocks, seed = 4, 30, 7
    probs = [0.9 - 0.01 * i for i in range(blocks)]
    reference = GEN.build_instance(num_controls, blocks, 0, seed, h_prob)[0]
    reference_dist = GEN.measurement_distributions(reference, num_controls)

    def simulated_safe(mask: int) -> bool:
        circuit = GEN.build_instance(num_controls, blocks, mask, seed, h_prob)[0]
        dist = GEN.measurement_distributions(circuit, num_controls)
        return max(
            GEN.total_variation(reference_dist[b], dist[b]) for b in reference_dist
        ) <= 1e-9

    def simulated_greedy() -> int:
        accepted = 0
        for block in sorted(range(blocks), key=lambda i: -probs[i]):
            candidate = accepted | (1 << block)
            if simulated_safe(candidate):
                accepted = candidate
        return accepted

    print(f"  [bench] 30-block greedy decode x{repeats} - simulating ...", flush=True)
    sim_runs = []
    sym_runs = []
    identical = True
    for run in range(repeats):
        started = time.perf_counter()
        simulated_mask = simulated_greedy()
        sim_runs.append(time.perf_counter() - started)
        verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks, seed, h_prob)
        started = time.perf_counter()
        symbolic_mask = verifier.greedy(probs)
        sym_runs.append(time.perf_counter() - started)
        identical = identical and (symbolic_mask == simulated_mask)
        print(f"    run {run + 1}/{repeats}: simulation={sim_runs[-1]:.3f}s "
              f"symbolic={sym_runs[-1]:.5f}s "
              f"({sim_runs[-1] / max(sym_runs[-1], 1e-12):,.0f}x)", flush=True)
    rows.append(_summarise("greedy", "30-block greedy decode",
                           f"{blocks} candidates, identical mask: {identical}",
                           sim_runs, sym_runs))
    print(f"    -> simulation {rows[-1]['simulation_mean_s']:.3f} +/- "
          f"{rows[-1]['simulation_std_s']:.3f}s | symbolic "
          f"{rows[-1]['symbolic_mean_s']:.5f} +/- {rows[-1]['symbolic_std_s']:.5f}s | "
          f"{rows[-1]['ratio_mean']:,.0f} +/- {rows[-1]['ratio_std']:,.0f}x", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 4.2 - strategy pools, absolute T-counts
# ---------------------------------------------------------------------------


def _pools(device) -> Dict[str, List[Any]]:
    config = RL.GEN.VariantConfig(h_prob=0.7)
    control_values = [2, 3, 4, 5, 6, 7, 7, 8, 8]

    def build(kind: str, count: int, salt: int) -> List[Any]:
        rng = np.random.default_rng(20261012 + salt)
        seeds = np.random.SeedSequence(20261012 + salt).generate_state(count, dtype=np.uint32)
        if kind == "shallow":
            return [
                RL.Instance(int(rng.choice(control_values)), int(rng.integers(2, 8)),
                            int(seeds[i]), 0.7, config)
                for i in range(count)
            ]
        return [
            RL.Instance(2, int(rng.integers(30, 51)), int(seeds[i]), 0.7, config)
            for i in range(count)
        ]

    return {"shallow": build("shallow", 16, 123), "deep": build("deep", 12, 999)}


def measure_pools(checkpoint: Path, device) -> Dict[str, Any]:
    model, _ = MS.load_policy(checkpoint, device)
    pools = _pools(device)
    out: Dict[str, Any] = {}
    for name, pool in pools.items():
        strategies: Dict[str, List[Dict[str, float]]] = {
            "all-exact": [], "raw-threshold (illegal)": [],
            "threshold+repair": [], "depth-cap": [], "greedy": [],
        }
        for instance in pool:
            batch, _ = RL.instance_batch([instance], device)
            probs = torch.sigmoid(model(batch)).tolist()
            exact_t = instance.t_all_exact
            raw_mask = instance.symbolic.mask_from_probs(probs, 0.5)
            repaired_mask, _ = MS.repair_mask(instance.symbolic, probs, raw_mask)
            cap_mask = MS.top_mask(probs, MS.depth_budget(instance.blocks))
            cap_repaired, _ = MS.repair_mask(instance.symbolic, probs, cap_mask)
            greedy_mask = instance.symbolic.greedy(probs)
            values = {
                "all-exact": exact_t,
                "raw-threshold (illegal)": instance.t_policy(raw_mask),
                "threshold+repair": instance.t_policy(repaired_mask),
                "depth-cap": instance.t_policy(cap_repaired),
                "greedy": instance.t_policy(greedy_mask),
            }
            for key, value in values.items():
                strategies[key].append({"t": float(value), "saved": float(exact_t - value)})
        out[name] = {
            "n": len(pool),
            "mean_blocks": float(np.mean([i.blocks for i in pool])),
            "strategies": strategies,
        }
        print(f"  [pools] {name}: n={len(pool)} mean blocks={out[name]['mean_blocks']:.1f} "
              f"all-exact={np.mean([s['t'] for s in strategies['all-exact']]):.1f} T", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect the figure data as JSON.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "figure_data"))
    parser.add_argument("--benchmarks", action="store_true", help="re-measure section 5.1")
    parser.add_argument("--repeats", type=int, default=3,
                        help="repetitions per benchmark (mean/std are reported; 3 by default)")
    parser.add_argument("--pools", action="store_true", help="re-measure section 4.2 (~2 min)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    args = parser.parse_args(argv)
    if not (args.benchmarks or args.pools):
        parser.error("choose at least one of --benchmarks / --pools")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.benchmarks:
        rows = measure_benchmarks(args.repeats)
        path = out_dir / "verifier_bench.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"  wrote {path}", flush=True)
    if args.pools:
        pools = measure_pools(Path(args.checkpoint), device)
        path = out_dir / "strategy_pools.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(pools, handle, indent=2)
        print(f"  wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
