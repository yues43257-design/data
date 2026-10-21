#!/usr/bin/env python3
"""Scoring protocol v3: L (soundness/completeness), S (semantics), C (properties).

Each verifier answers exactly one question; ``n/a`` is an abstention, never a
false positive.  Cross-verifier comparisons use the **common decided subset**,
and every rate carries a Wilson 95% interval.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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

BENCH_VERSION = "compiler-bench-v3"
QCEEC = "/tmp/qcec-venv/bin/python"


def wilson(successes: int, total: int, z: float = 1.96):
    if not total:
        return (None, None)
    centre = (successes + z * z / 2) / (total + z * z)
    half = z * math.sqrt(centre * (1 - centre) / (total + z * z))
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def cert_verdict(entry: Dict[str, Any]) -> Optional[bool]:
    if entry.get("kind") == "pass2":
        return bool(VC.verify_chain(entry)["ok"])
    try:
        valid = bool(ST.verify_certificate(
            entry["schedule"], entry["mask"], entry["certificate"],
            circuit=entry["circuit"], block_ranges=entry["ranges"]))
    except Exception:
        return False
    return bool(valid and entry["certificate"].get("accepted", False))


def qcec_legal(entry: Dict[str, Any]) -> Optional[bool]:
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


def qcec_exact(entry: Dict[str, Any]) -> Optional[bool]:
    circuit, ideal = entry["circuit"], entry["ideal"]
    if circuit.num_clbits or ideal.num_qubits > 18:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.qasm", Path(tmp) / "b.qasm"
        try:
            a.write_text(qasm2.dumps(ideal))
            b.write_text(qasm2.dumps(circuit))
        except Exception:
            return None
        try:
            proc = subprocess.run([QCEEC, "/tmp/qcec_one.py", str(a), str(b), "30"],
                                  capture_output=True, text=True, timeout=60)
            out = (proc.stdout or "").strip().splitlines()
            verdict = out[-1].split()[0] if out else "none"
        except Exception:
            return None
    if verdict == "equivalent":
        return True
    if verdict == "not_equivalent":
        return False
    return None


def run(with_qcec: bool = False, qcec_limit: int = 24) -> Dict[str, Any]:
    cases = C.all_cases()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases):
        if "error" in case:
            rows.append({"case": case["name"], "error": case["error"]})
            continue
        entry = case["entry"]
        use_qcec = (with_qcec and (qcec_limit <= 0 or index < qcec_limit)
                    and entry.get("ideal") is not None)
        t0 = time.perf_counter()
        q_legal = qcec_legal(entry) if use_qcec else None
        q_legal_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        q_exact = qcec_exact(entry) if use_qcec else None
        q_exact_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cert = cert_verdict(entry)
        cert_s = time.perf_counter() - t0
        cert_bytes = (len(json.dumps(entry["certificate"]))
                      if "certificate" in entry else None)
        truth_l = case["truth_L"] if case["truth_L"] is not None else q_legal
        truth_s = case["truth_S"] if case["truth_S"] is not None else q_exact
        prop = C.cert_property(case["attack"], truth_l)
        rows.append({
            "case": case["name"], "family": case["family"], "layout": case["layout"],
            "seed": case["seed"], "attack": case["attack"],
            "cert_property": prop, "qubits": case["qubits"],
            "ground_truth": case["ground_truth"],
            "truth_L": truth_l, "truth_S": truth_s,
            "dense_legal": case["truth_L"], "dense_exact": case["truth_S"],
            "qcec_legal": q_legal, "qcec_exact": q_exact,
            "cert_ok": cert,
            "qcec_legal_s": round(q_legal_s, 3),
            "qcec_exact_s": round(q_exact_s, 3),
            "cert_s": round(cert_s, 6),
            "cert_bytes": cert_bytes,
        })

    def l_metrics(key: str) -> Dict[str, Any]:
        decided = [r for r in rows if r.get(key) is not None and r.get("truth_L") is not None]
        illegal = [r for r in decided if r["truth_L"] is False]
        legal = [r for r in decided if r["truth_L"] is True]
        sound = sum(1 for r in illegal if r[key] is False)
        complete = sum(1 for r in legal if r[key] is True)
        return {
            "decided": len(decided), "coverage": round(len(decided) / max(1, len(rows)), 3),
            "soundness": round(sound / len(illegal), 3) if illegal else None,
            "soundness_ci": wilson(sound, len(illegal)),
            "completeness": round(complete / len(legal), 3) if legal else None,
            "completeness_ci": wilson(complete, len(legal)),
            "illegal": len(illegal), "legal": len(legal),
        }

    s_pairs = [r for r in rows if r.get("qcec_exact") is not None
               and r.get("dense_exact") is not None]
    s_correct = sum(1 for r in s_pairs if r["qcec_exact"] == r["dense_exact"])

    def cert_metrics(prop: str, expect_accept: bool) -> Dict[str, Any]:
        cells = [r for r in rows if r.get("cert_property") == prop]
        good = sum(1 for r in cells if r["cert_ok"] is expect_accept)
        return {"cases": len(cells), "correct": good,
                "rate": round(good / len(cells), 3) if cells else None,
                "ci": wilson(good, len(cells))}

    return {
        "version": BENCH_VERSION, "cases": len(rows),
        "L": {"dense": l_metrics("dense_legal"), "qcec": l_metrics("qcec_legal")},
        "S": {"common_subset": len(s_pairs), "correct": s_correct,
              "accuracy": round(s_correct / len(s_pairs), 3) if s_pairs else None,
              "ci": wilson(s_correct, len(s_pairs))},
        "C": {
            "positive_accept": cert_metrics("positive", True),
            "illegal_rewrite_reject": cert_metrics("illegal_rewrite", False),
            "undecided_legal_cases": sum(
                1 for r in rows if r.get("cert_property") == "undecided_legal"),
            "binding_reject": cert_metrics("binding", False),
            "tamper_reject": cert_metrics("tamper", False),
            "chain_reject": cert_metrics("chain", False),
        },
        "cost": {
            "qcec_legal_s_mean": round(sum(r.get("qcec_legal_s", 0) for r in rows)
                                       / max(1, sum(1 for r in rows
                                                    if r.get("qcec_legal_s"))), 3),
            "qcec_exact_s_mean": round(sum(r.get("qcec_exact_s", 0) for r in rows)
                                       / max(1, sum(1 for r in rows
                                                    if r.get("qcec_exact_s"))), 3),
            "cert_ms_mean": round(1e3 * sum(r.get("cert_s", 0) for r in rows)
                                  / max(1, len(rows)), 3),
            "cert_bytes_mean": round(sum(r.get("cert_bytes") or 0 for r in rows)
                                     / max(1, sum(1 for r in rows if r.get("cert_bytes"))), 1),
        },
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-qcec", action="store_true")
    parser.add_argument("--qcec-limit", type=int, default=0,
                        help="0 = all cases")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    report = run(args.with_qcec, args.qcec_limit)
    print("=" * 104)
    print(f"  compiler-bug benchmark  {BENCH_VERSION}  cases={report['cases']}")
    print("=" * 104)
    for name, m in (("dense", report["L"]["dense"]), ("qcec", report["L"]["qcec"])):
        print(f"  L/{name:<6} decided={m['decided']:<4} coverage={m['coverage']} "
              f"soundness={m['soundness']} {m['soundness_ci']} "
              f"completeness={m['completeness']} {m['completeness_ci']}")
    s = report["S"]
    print(f"  S        common={s['common_subset']} correct={s['correct']} "
          f"accuracy={s['accuracy']} {s['ci']}")
    for name, m in report["C"].items():
        if not isinstance(m, dict):
            continue
        print(f"  C/{name:<16} cases={m['cases']:<4} correct={m['correct']:<4} "
              f"rate={m['rate']} {m['ci']}")
    print(f"  C/undecided_legal_cases = {report['C']['undecided_legal_cases']}")
    print(f"  cost: {report['cost']}")
    print(f"  ({time.perf_counter() - started:.1f}s)")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
