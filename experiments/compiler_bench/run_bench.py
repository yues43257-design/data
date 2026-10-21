#!/usr/bin/env python3
"""Scoring protocol for the compiler-bug benchmark.

Each verifier answers exactly one question:

    dense      -> L (legality)
    QCEC       -> L (legality, optional subset)
    cert_v1    -> C (certificate, unbound semantics)
    cert_v2    -> C (certificate, bound semantics)

``n/a`` (None) is an **abstention**, not a false positive.  Accuracy is computed
over decided answers only; false positives (said "pass" when truth is fail) and
false negatives are reported separately.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from qiskit import qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import support_tracker as ST                # noqa: E402
import experiments.compiler_bench.cases as C  # noqa: E402
import experiments.multipass_attestation.verify_chain as VC  # noqa: E402
import qcec_validate as QV                  # noqa: E402

BENCH_VERSION = "compiler-bench-v2"
QCEEC = "/tmp/qcec-venv/bin/python"


def cert_v1(entry: Dict[str, Any]) -> Optional[bool]:
    """Old, unbound semantics: compare the core fields only."""
    if entry.get("kind") == "pass2":
        return None
    fresh = ST.certify(entry["schedule"], entry["mask"])
    core = ("mask", "accepted", "final_support", "h_checks", "violations")
    same = all(entry["certificate"].get(k) == fresh.get(k) for k in core)
    return bool(same and entry["certificate"].get("accepted", False))


def cert_v2(entry: Dict[str, Any]) -> Optional[bool]:
    """Bound semantics (hash + witness)."""
    if entry.get("kind") == "pass2":
        return bool(VC.verify_chain(entry)["ok"])
    try:
        valid = bool(ST.verify_certificate(
            entry["schedule"], entry["mask"], entry["certificate"],
            circuit=entry["circuit"], block_ranges=entry["ranges"]))
    except Exception:
        return False
    return bool(valid and entry["certificate"].get("accepted", False))


def dense(entry: Dict[str, Any]) -> Optional[bool]:
    return entry.get("truth_L") if "truth_L" in entry else None


def qcec(entry: Dict[str, Any]) -> Optional[bool]:
    circuit, ideal = entry["circuit"], entry["ideal"]
    if circuit.num_clbits or ideal.num_qubits > 18:
        return None
    try:
        pairs = QV._diag_pairs(0, ideal, circuit)
    except Exception:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        results = QV._run_config(QCEEC, pairs, "alternating", 30, 60, Path(tmp))
    if any(r["verdict"] == "not_equivalent" for r in results):
        return False
    if all(r["verdict"] == "equivalent" for r in results):
        return True
    return None


def _score(verdict: Optional[bool], truth: Optional[bool]) -> str:
    if verdict is None or truth is None:
        return "abstain"
    return "correct" if verdict == truth else "wrong"


def run(with_qcec: bool = False, qcec_limit: int = 0) -> Dict[str, Any]:
    cases = C.all_cases()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        if "error" in case:
            rows.append({"case": case["name"], "error": case["error"]})
            continue
        entry = case["entry"]
        verdicts = {
            "dense": case["truth_L"],
            "qcec": (qcec(entry) if with_qcec and qcec_limit > 0
                     and index < qcec_limit and entry.get("ideal") is not None
                     else None),
            "cert_v1": cert_v1(entry),
            "cert_v2": cert_v2(entry),
        }
        row = {"case": case["name"], "attack": case["attack"],
               "family": case["family"], "layout": case["layout"],
               "ground_truth": case["ground_truth"],
               "truth_L": case["truth_L"], "truth_S": case["truth_S"],
               "truth_C": case["truth_C"], **verdicts}
        row["score_dense"] = _score(verdicts["dense"], case["truth_L"])
        row["score_qcec"] = _score(verdicts["qcec"], case["truth_L"])
        row["score_cert_v1"] = _score(verdicts["cert_v1"], case["truth_C"])
        row["score_cert_v2"] = _score(verdicts["cert_v2"], case["truth_C"])
        rows.append(row)

    def metrics(prefix: str) -> Dict[str, Any]:
        scores = [r[f"score_{prefix}"] for r in rows if f"score_{prefix}" in r]
        decided = [s for s in scores if s != "abstain"]
        return {
            "cases": len(scores),
            "decided": len(decided),
            "abstain": scores.count("abstain"),
            "correct": decided.count("correct"),
            "wrong": decided.count("wrong"),
            "accuracy": round(decided.count("correct") / len(decided), 3)
            if decided else None,
        }

    return {"version": BENCH_VERSION, "cases": len(rows),
            "metrics": {k: metrics(k) for k in
                        ("dense", "qcec", "cert_v1", "cert_v2")},
            "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-qcec", action="store_true")
    parser.add_argument("--qcec-limit", type=int, default=24)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = run(args.with_qcec, args.qcec_limit)
    print("=" * 118)
    print(f"  compiler-bug benchmark  {BENCH_VERSION}  cases={report['cases']}")
    print("=" * 118)
    print(f"  {'case':<38}{'L':>6}{'S':>6}{'C':>6}{'dense':>7}{'qcec':>7}"
          f"{'v1':>6}{'v2':>6}")
    for r in report["rows"]:
        if "error" in r:
            print(f"  {r['case']:<38}  {r['error']}")
            continue
        print(f"  {r['case']:<38}{str(r['truth_L']):>6}{str(r['truth_S']):>6}"
              f"{str(r['truth_C']):>6}{str(r['dense']):>7}{str(r['qcec']):>7}"
              f"{str(r['cert_v1']):>6}{str(r['cert_v2']):>6}")
    print("-" * 118)
    for name, m in report["metrics"].items():
        print(f"  {name:<8} decided={m['decided']:<4} abstain={m['abstain']:<4} "
              f"correct={m['correct']:<4} wrong={m['wrong']:<3} "
              f"accuracy={m['accuracy']}")
    print(f"  ({time.perf_counter() - started:.1f}s)")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
