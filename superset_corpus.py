#!/usr/bin/env python3
"""Reinforcement 4, acceptance run: NEGATIVE CONTROL + EXPANDED CORPUS.

One pass, two questions.

EXPANDED CORPUS
    For every mask the tracker ACCEPTS, the true dependency set of the residual

        R = U_cand . U_ideal^dagger

    must be covered by the set the tracker propagated:

        true_support(R)  subset-of  tracked_support(mask)

    If a qubit the residual genuinely depends on were missing from the tracked
    set, a later Hadamard on it would go unnoticed and the tracker could accept
    an unsafe mask.  ``violations == 0`` is what upgrades "sound by lemma plus
    implementation assumption" into an end-to-end statement about THIS code.
    Coverage is reported per width N (data qubits + ancillas), including
    ancilla-rich layouts (ancilla_pool in {2, 4, 6}), up to N = 12.

NEGATIVE CONTROL
    A property whose test cannot fail is not evidence.  The tracker's support
    comes from four propagation rules; this run deletes them one at a time and
    re-asks the same question on the SAME accepted masks:

        drop the ancilla merge          (downgraded / conjugated block)
        drop the CX push rule           (layer CX whose target is supported)
        drop the block conjugation rule (whole-block permutation)
        drop the mask injection rule    (the downgraded block itself)

    The control is discriminative iff at least one deletion produces violations.
    Every deletion is reported separately, including the ones that stay silent:
    a rule the corpus never exercises is an honest gap in the evidence, not a
    pass, and is called out as such.

Usage
-----
    .venv/bin/python superset_corpus.py --phase small
    .venv/bin/python superset_corpus.py --phase n10 --jobs 4
    .venv/bin/python superset_corpus.py --all --jobs 4
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "figure_data" / "superset_corpus.jsonl"


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


ULT = _load("sc_ult", "ultimate_scalability_test.py")
WS = _load("sc_ws", "wide_scale_stress.py")
ST = _load("sc_st", "support_tracker.py")
SETUP = _load("sc_setup", "experiment_setup.py")
FR = _load("sc_fr", "fast_residual.py")


# ---------------------------------------------------------------------------
# propagation rules
# ---------------------------------------------------------------------------

def tracked_support(schedule: Dict[str, Any], mask: int, *,
                    cx_rule: bool = True, block_rule: bool = True,
                    mask_rule: bool = True, ancillas: bool = True) -> Set[int]:
    """The tracker's propagated support, with individual rules switchable off.

    The defaults reproduce ``support_tracker.SupportTracker`` exactly; the
    switches exist only for the negative control.
    """
    support: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        if cx_rule:
            for name, qubits in schedule["layers"][index]:
                if name == "cx" and int(qubits[1]) in support:
                    support.add(int(qubits[0]))
        if block_rule and block["target"] in support:
            support.update(int(q) for q in block["controls"])
            if ancillas:
                support.update(int(q) for q in block["ancillas"])
        if mask_rule and ((mask >> index) & 1):
            support.add(int(block["target"]))
            support.update(int(q) for q in block["controls"])
            if ancillas:
                support.update(int(q) for q in block["ancillas"])
    return support


VARIANTS: Dict[str, Dict[str, bool]] = {
    "full (shipped rules)": {},
    "drop ancilla-merge": {"ancillas": False},
    "drop cx-push": {"cx_rule": False},
    "drop block-conjugation": {"block_rule": False},
    "drop mask-injection": {"mask_rule": False},
}


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------

def check_instance(job: Tuple[int, int, int, float, int]) -> Dict[str, Any]:
    width, pool, blocks, h_prob, seed = job
    t0 = time.perf_counter()
    inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                            h_prob=h_prob, ancilla_pool=pool)
    schedule = WS.build_schedule(inst)
    ideal = inst.circuit(0)
    record: Dict[str, Any] = {
        "width": width, "ancilla_pool": pool, "N": inst.total, "blocks": blocks,
        "h_prob": h_prob, "seed": seed,
        "masks": 1 << blocks, "accepted": 0, "checked": 0,
        "diag_min_abs": None, "off_diagonal_max": 0.0, "undefined": 0,
        "variant_violations": {name: 0 for name in VARIANTS},
        "examples": [],
    }
    for mask in range(1 << blocks):
        if not ST.track_safe(schedule, mask):
            continue                       # rejected masks: support is moot
        record["accepted"] += 1
        cand = inst.circuit(mask)
        R = FR.residual(FR.build_unitary(cand), ideal)
        off = FR.off_diagonal(R)
        record["off_diagonal_max"] = max(record["off_diagonal_max"], off)
        diag_abs = float(np.abs(np.diag(R)).min())
        record["diag_min_abs"] = diag_abs if record["diag_min_abs"] is None \
            else min(record["diag_min_abs"], diag_abs)
        truth = FR.true_support(R, inst.total)
        if truth is None:
            record["undefined"] += 1
            del R
            continue
        record["checked"] += 1
        for name, kwargs in VARIANTS.items():
            tracked = tracked_support(schedule, mask, **kwargs)
            if not truth.issubset(tracked):
                record["variant_violations"][name] += 1
                if len(record["examples"]) < 3:
                    record["examples"].append({
                        "variant": name, "mask": mask, "N": inst.total,
                        "width": width, "ancilla_pool": pool,
                        "missing": sorted(truth - tracked),
                        "true": sorted(truth), "tracked": sorted(tracked),
                    })
        del R
    record["seconds"] = round(time.perf_counter() - t0, 2)
    return record


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def grid(phase: str) -> List[Tuple[int, int, int, float, int]]:
    """(width, ancilla_pool, blocks, h_prob, seed) jobs, N = width + pool."""
    if phase == "small":       # N <= 9, wide sweep of every axis
        # width >= 2: WideInstance picks a window of n <= width-2 data qubits
        layouts = [(w, 2) for w in range(2, 8)] + \
                  [(w, 4) for w in range(2, 6)] + \
                  [(w, 6) for w in range(2, 4)]
        blocks_list, hs, seeds = (1, 2, 3, 4), (0.0, 0.3, 0.7, 1.0), 3
    elif phase == "n10":
        layouts = [(8, 2), (6, 4), (4, 6)]
        blocks_list, hs, seeds = (1, 2, 3, 4), (0.3, 0.7, 1.0), 2
    elif phase == "n11":
        layouts = [(9, 2), (7, 4), (5, 6)]
        blocks_list, hs, seeds = (2, 3, 4), (0.7, 1.0), 1
    elif phase == "n12":
        layouts = [(10, 2), (8, 4), (6, 6)]
        blocks_list, hs, seeds = (2, 3), (0.7, 1.0), 1
    elif phase == "smoke":
        layouts = [(4, 2), (3, 4)]
        blocks_list, hs, seeds = (2, 3), (0.7,), 1
    else:                      # pragma: no cover - defensive
        raise ValueError(phase)
    jobs: List[Tuple[int, int, int, float, int]] = []
    for width, pool in layouts:
        for blocks in blocks_list:
            for h in hs:
                for si in range(seeds):
                    seed = SETUP.seed_for(si) * 11 + width * 7 + blocks * 3 + pool \
                        + int(h * 100) * 13
                    jobs.append((width, pool, blocks, h, seed))
    return jobs


PHASE_JOBS = {"smoke": 4, "small": 5, "n10": 4, "n11": 3, "n12": 2}


def check_tracker_equivalence(phases: Sequence[str]) -> int:
    """The rule set above must reproduce the SHIPPED ``DiagonalTracker`` exactly.

    ``tracked_support`` is a re-implementation (it needs the rules switchable
    for the negative control), so "0 violations" is only a statement about the
    shipped tracker if the two agree.  This replays ``ST.DiagonalTracker`` by
    hand on every mask of the grid and compares both the support set (on
    accepted masks) and the safety verdict.
    """
    from collections import Counter

    names: Counter = Counter()
    checked = accepted = 0
    bad_safety = bad_support = 0
    examples: List[Any] = []
    for phase in phases:
        for width, pool, blocks, h_prob, seed in grid(phase):
            inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                    h_prob=h_prob, ancilla_pool=pool)
            schedule = WS.build_schedule(inst)
            for layer in schedule["layers"]:
                for name, _ in layer:
                    names[name] += 1
            for mask in range(1 << blocks):
                tracker = ST.DiagonalTracker(schedule["total"])
                for index, block in enumerate(schedule["blocks"]):
                    tracker.apply_layer(schedule["layers"][index])
                    tracker.apply_block(block)
                    if (mask >> index) & 1:
                        tracker.inject_correction(block)
                checked += 1
                if tracker.is_safe() != ST.track_safe(schedule, mask):
                    bad_safety += 1
                if tracker.is_safe():
                    accepted += 1
                    if tracked_support(schedule, mask) != tracker.support:
                        bad_support += 1
                        if len(examples) < 3:
                            examples.append((width, blocks, h_prob, mask,
                                             sorted(tracker.support),
                                             sorted(tracked_support(schedule, mask))))
    print("=" * 96)
    print("  does the local rule set reproduce the SHIPPED DiagonalTracker?")
    print(f"    layer gate names seen : {dict(names)}")
    print(f"    masks replayed        : {checked} (accepted {accepted})")
    print(f"    safety verdict differs: {bad_safety}")
    print(f"    support set differs   : {bad_support}")
    for example in examples:
        print(f"      {example}")
    ok = bad_safety == 0 and bad_support == 0
    print(f"    -> {'IDENTICAL on every accepted mask' if ok else 'DIVERGES'}")
    print("=" * 96)
    return 0 if ok else 1


def run_phase(phase: str, jobs: int, sink_path: Path, truncate: bool) -> Dict[str, Any]:
    work = grid(phase)
    print(f"  phase {phase:<6} jobs={len(work):<4} workers={jobs}")
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    ctx = mp.get_context("fork")
    with sink_path.open("w" if truncate else "a", encoding="utf-8") as sink:
        with ctx.Pool(jobs) as pool:
            for record in pool.imap_unordered(check_instance, work, chunksize=1):
                rows.append(record)
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                done = len(rows)
                if done % 20 == 0 or done == len(work):
                    print(f"    {done}/{len(work)} instances  "
                          f"({time.perf_counter() - started:.0f}s)", flush=True)
    elapsed = time.perf_counter() - started
    return {"phase": phase, "jobs": len(work), "rows": rows, "seconds": elapsed}


def summarise(result: Dict[str, Any]) -> None:
    rows = result["rows"]
    total_checked = sum(r["checked"] for r in rows)
    print("=" * 96)
    print(f"  phase {result['phase']}: {len(rows)} instances, "
          f"{total_checked} accepted masks checked, {result['seconds']:.0f}s")
    print(f"  {'N':>3} {'inst':>5} {'masks':>6} {'acc':>5} {'checked':>8} "
          f"{'offdiag_max':>12} {'min|diag|':>10}")
    by_n: Dict[int, Dict[str, float]] = {}
    for r in rows:
        cell = by_n.setdefault(r["N"], {"inst": 0, "masks": 0, "acc": 0,
                                        "checked": 0, "off": 0.0, "diag": 1.0})
        cell["inst"] += 1
        cell["masks"] += r["masks"]
        cell["acc"] += r["accepted"]
        cell["checked"] += r["checked"]
        cell["off"] = max(cell["off"], r["off_diagonal_max"])
        if r["diag_min_abs"] is not None:
            cell["diag"] = min(cell["diag"], r["diag_min_abs"])
    for n in sorted(by_n):
        c = by_n[n]
        print(f"  {n:>3} {int(c['inst']):>5} {int(c['masks']):>6} {int(c['acc']):>5} "
              f"{int(c['checked']):>8} {c['off']:>12.2e} {c['diag']:>10.6f}")
    print("-" * 96)
    print("  propagation-rule deletions (negative control) on the SAME masks:")
    for name in VARIANTS:
        viol = sum(r["variant_violations"][name] for r in rows)
        tag = "discriminative" if viol else ("reference" if name.startswith("full")
                                             else "NO EFFECT (rule unexercised here)")
        print(f"    {name:<24} violations={viol:<6} {tag}")
    undefined = sum(r["undefined"] for r in rows)
    if undefined:
        print(f"    NOTE: {undefined} accepted masks had an undefined residual support")
    examples = [ex for r in rows for ex in r["examples"]
                if not ex["variant"].startswith("full")]
    for ex in examples[:4]:
        print(f"    e.g. {ex['variant']}: N={ex.get('N')} missing={ex['missing']}")
    print("=" * 96)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="small",
                        choices=["smoke", "small", "n10", "n11", "n12"])
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="discard records from earlier phases")
    parser.add_argument("--check-tracker", action="store_true",
                        help="cross-check the rule set against the shipped tracker")
    args = parser.parse_args(argv)
    phases = ["smoke", "small", "n10", "n11", "n12"] if args.all else [args.phase]
    if args.check_tracker:
        return check_tracker_equivalence(phases)
    all_rows: List[Dict[str, Any]] = []
    for phase in phases:
        workers = args.jobs or PHASE_JOBS[phase]
        result = run_phase(phase, workers, OUT, truncate=args.fresh)
        args.fresh = False                # only the first phase may truncate
        all_rows.extend(result["rows"])
        summarise(result)
    # cumulative aggregate over every record written so far (all phases)
    cumulative = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    summary = {
        "instances": len(cumulative),
        "accepted_masks_checked": sum(r["checked"] for r in cumulative),
        "accepted_masks_total": sum(r["accepted"] for r in cumulative),
        "variant_violations": {name: sum(r["variant_violations"][name]
                                          for r in cumulative) for name in VARIANTS},
        "undefined": sum(r["undefined"] for r in cumulative),
        "max_off_diagonal_on_accepted": max(r["off_diagonal_max"] for r in cumulative),
        "min_diag_abs_on_accepted": min(r["diag_min_abs"] for r in cumulative
                                        if r["diag_min_abs"] is not None),
        "rows": cumulative,
    }
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("=" * 96)
    print("  CUMULATIVE (all phases so far)")
    print(f"    instances = {summary['instances']}, "
          f"accepted masks checked = {summary['accepted_masks_checked']}")
    print(f"    max off-diagonal on an accepted mask = "
          f"{summary['max_off_diagonal_on_accepted']:.2e}")
    print(f"    min |diag| on an accepted mask       = "
          f"{summary['min_diag_abs_on_accepted']:.6f}")
    for name, viol in summary["variant_violations"].items():
        tag = "OK" if (name.startswith("full") and viol == 0) else (
            "discriminative" if viol else "unexercised")
        print(f"    {name:<24} violations={viol:<6} {tag}")
    print("=" * 96)
    print(f"  per-instance records: {OUT}")
    print(f"  aggregate summary   : {OUT.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
