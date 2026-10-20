#!/usr/bin/env python3
"""Hybrid pre-filter + exact refinement, with proposal modes (v2).

    accept  -> certificate (no exact check)
    reject  -> exact check, if this width has a backend:
                 N <= 11          dense Operator        (``dense``)
                 12 <= N <= 18    QCEC, alternating     (``qcec``, needs a venv)
                 otherwise        unresolved

Proposal modes (``--proposal``):

    enum      every mask (2**blocks)             -- upper bound on recall
    one_shot  only the all-relative mask         -- what a policy proposes
    greedy    sequential accept, tracker only    -- no exact calls at all

The three cost regimes are reported **separately** (they differ by 4 orders of
magnitude):

    homogeneous + reduced verifier   ~ms total
    WideInstance + dense             ~10 ms .. 10 s per call
    large N + QCEC                   0.1-0.9 s per query, 1-15 s per full check

Usage
-----
    .venv/bin/python hybrid_check.py
    .venv/bin/python hybrid_check.py --proposal one_shot
    .venv/bin/python hybrid_check.py --proposal enum --qcec-python /tmp/qcec-venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC      # noqa: E402
import support_tracker as ST                # noqa: E402
import ultimate_scalability_test as ULT     # noqa: E402
import wide_scale_stress as WS              # noqa: E402

HYBRID_VERSION = "hybrid-v2"
DENSE_LIMIT = 11


# ---------------------------------------------------------------------------
# exact backends
# ---------------------------------------------------------------------------

def exact_dense(ideal: Any, candidate: Any) -> bool:
    return bool(RPC.relative_phase_ok(ideal, candidate))


def exact_qcec(ideal: Any, candidate: Any, python: str,
               timeout: float = 30) -> Optional[bool]:
    """All-n Z_q commutation via QCEC; None if the checker is unavailable."""
    if not Path(python).exists():
        return None
    import qcec_validate as QV

    pairs = QV._diag_pairs(0, ideal, candidate)
    with tempfile.TemporaryDirectory() as tmp:
        results = QV._run_config(python, pairs, "alternating", timeout,
                                 timeout * 2, Path(tmp))
    if any(r["verdict"] == "not_equivalent" for r in results):
        return False
    if all(r["verdict"] == "equivalent" for r in results):
        return True
    return None


def check_mask(schedule: Dict[str, Any], ideal: Any,
               build_candidate: Callable[[int], Any], mask: int,
               n_qubits: int, *, threshold: int = 18, backend: str = "auto",
               qcec_python: str = "",
               exact_fn: Optional[Callable[[int], bool]] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    certificate = ST.certify(schedule, mask)
    if certificate["accepted"]:
        return {"mask": mask, "verdict": "accepted", "route": "tracker",
                "seconds": time.perf_counter() - started,
                "certificate": certificate}
    exact: Optional[bool] = None
    route = "none"
    if exact_fn is not None:                       # symbolic verifier backend
        exact, route = bool(exact_fn(mask)), "symbolic"
    elif n_qubits <= DENSE_LIMIT and backend in ("auto", "dense"):
        exact, route = exact_dense(ideal, build_candidate(mask)), "dense"
    elif n_qubits <= threshold and backend in ("auto", "qcec") and qcec_python:
        exact = exact_qcec(ideal, build_candidate(mask), qcec_python)
        route = "qcec" if exact is not None else "none"
    verdict = ("legal" if exact else "illegal") if exact is not None else "unresolved"
    return {"mask": mask, "verdict": verdict, "route": route,
            "seconds": time.perf_counter() - started, "certificate": certificate}


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------

def proposals(blocks: int, mode: str, schedule: Dict[str, Any]) -> List[int]:
    if mode == "enum":
        return list(range(1 << blocks))
    if mode == "one_shot":
        return [(1 << blocks) - 1]
    if mode == "greedy":
        accepted = 0
        for index in range(blocks):
            candidate = accepted | (1 << index)
            if ST.track_safe(schedule, candidate):
                accepted = candidate
        return [accepted]
    raise ValueError(mode)


def certificate_bytes(cert: Dict[str, Any]) -> int:
    return len(json.dumps(cert))


def run_population(kind: str, mode: str, backend: str,
                   qcec_python: str, threshold: int) -> Dict[str, Any]:
    import experiment_setup as SETUP
    import symbolic_verifier as SYM

    rows: List[Dict[str, Any]] = []
    agg = {"proposals": 0, "legal": 0, "accepted": 0, "illegal_accepted": 0,
           "tracker_accept": 0,
           "tracker_s": 0.0, "exact_s": 0.0, "exact_calls": 0,
           "cert_bytes": 0, "best_saving": 0, "tracker_only_saving": 0}
    configs = ([(2, 0.1), (3, 0.3), (4, 0.7)] if kind == "homogeneous" else
               [(4, 0.5), (6, 0.7)])
    for first, h_prob in configs:
        if kind == "homogeneous":
            verifier = SYM.ReducedSymbolicVerifier(first, 4, SETUP.seed_for(0), h_prob)
            schedule = ST.homogeneous_schedule(first, 4, SETUP.seed_for(0), h_prob)
            blocks = 4
            ideal = build = None
            truth = verifier.is_safe
            n_qubits = first + 3
        else:
            inst = ULT.WideInstance(width=first, blocks=4, seed=20261020,
                                    h_prob=h_prob, ancilla_pool=2)
            schedule = WS.build_schedule(inst)
            blocks, ideal, n_qubits = inst.blocks, inst.circuit(0), inst.total
            build = inst.circuit
            truth = lambda mask, i=inst: bool(RPC.relative_phase_ok(i.circuit(0), i.circuit(mask)))
        masks = proposals(blocks, mode, schedule)
        row = {"population": kind, "config": f"{first}/h={h_prob}", "N": n_qubits,
               "mode": mode, "proposals": len(masks),
               "tracker_s": 0.0, "exact_s": 0.0, "exact_calls": 0,
               "legal": 0, "accepted": 0, "illegal_accepted": 0,
               "tracker_accept": 0, "cert_bytes": 0, "tracker_only_saving": 0,
               "best_saving": 0}
        for mask in masks:
            legal = bool(truth(mask))
            row["legal"] += int(legal)
            result = check_mask(
                schedule, ideal, build if build else (lambda m: None), mask,
                n_qubits, threshold=threshold, backend=backend,
                qcec_python=qcec_python,
                exact_fn=(lambda m, v=verifier: bool(v.is_safe(m)))
                if kind == "homogeneous" else None)
            accepted = result["verdict"] in ("accepted", "legal")
            row["accepted"] += int(accepted)
            row["illegal_accepted"] += int(accepted and not legal)
            if accepted:      # only count T for masks we actually certify
                row["best_saving"] = max(row["best_saving"], 3 * bin(mask).count("1"))
            row["cert_bytes"] += certificate_bytes(result["certificate"])
            if result["route"] == "tracker":
                row["tracker_s"] += result["seconds"]
                row["tracker_accept"] += 1
                row["tracker_only_saving"] = max(
                    row["tracker_only_saving"], 3 * bin(mask).count("1"))
            elif result["route"] in ("dense", "qcec", "symbolic"):
                row["exact_s"] += result["seconds"]
                row["exact_calls"] += 1
        rows.append(row)
        for key in ("proposals", "legal", "accepted", "illegal_accepted",
                    "tracker_accept", "exact_calls"):
            agg[key] += row[key]
        agg["tracker_s"] += row["tracker_s"]
        agg["exact_s"] += row["exact_s"]
        agg["cert_bytes"] += row["cert_bytes"]
        agg["best_saving"] += row["best_saving"]
        agg["tracker_only_saving"] += row["tracker_only_saving"]
    return {"rows": rows, "totals": agg}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", default="all",
                        choices=["enum", "one_shot", "greedy", "all"])
    parser.add_argument("--backend", default="auto", choices=["auto", "dense", "qcec"])
    parser.add_argument("--qcec-python", default="/tmp/qcec-venv/bin/python")
    parser.add_argument("--threshold", type=int, default=11,
                        help="exact fallback up to this width. Default 11 is "
                             "interactive; 12-18 is an explicit slow path.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "hybrid_check.json"))
    args = parser.parse_args(argv)
    modes = ["enum", "one_shot", "greedy"] if args.proposal == "all" else [args.proposal]
    started = time.perf_counter()
    print("=" * 104)
    print(f"  HYBRID PRE-FILTER  version={HYBRID_VERSION}  backend={args.backend}  "
          f"threshold={args.threshold}")
    print("=" * 104)
    out: Dict[str, Any] = {
        "version": HYBRID_VERSION, "backend": args.backend,
        "threshold": args.threshold,
        "note": "recall denominators differ by proposal mode; compare "
                "best_T / enum_best_T instead",
        "results": {}}
    for kind in ("homogeneous", "wide"):
        enum_data = run_population(kind, "enum", args.backend, args.qcec_python,
                                   args.threshold)
        enum_best = enum_data["totals"]["best_saving"]
        for mode in modes:
            data = (enum_data if mode == "enum" else
                    run_population(kind, mode, args.backend, args.qcec_python,
                                   args.threshold))
            for row in data["rows"]:
                row["enum_best_T"] = enum_best
                row["best_over_enum"] = (round(row["best_saving"] / enum_best, 3)
                                         if enum_best else None)
            out["results"][f"{kind}/{mode}"] = data
            t = data["totals"]
            ratio = (t["best_saving"] / enum_best) if enum_best else float("nan")
            print(f"  {kind:<12}{mode:<10} proposals={t['proposals']:>3} "
                  f"legal_in_prop={t['legal']:>3} best_T={t['best_saving']:>3} "
                  f"enum_best={enum_best:>3} ratio={ratio:.2f} "
                  f"trackerOnly_T={t['tracker_only_saving']:>3} "
                  f"illegal={t['illegal_accepted']} "
                  f"tracker={t['tracker_s'] * 1e3:8.2f} ms "
                  f"exact={t['exact_s'] * 1e3:9.1f} ms "
                  f"calls={t['exact_calls']:>3} cert={t['cert_bytes']:>6} B", flush=True)
    print(f"  ({time.perf_counter() - started:.1f}s)")
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
