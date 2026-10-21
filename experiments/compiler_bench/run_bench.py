#!/usr/bin/env python3
"""Run the compiler correctness/safety benchmark.

Verifiers:
  dense   -- exact ``relative_phase_ok`` on the full operator (N <= 10)
  qcec    -- exact Z_q-commutation via MQT QCEC (N <= 18, alternating config)
  cert    -- support_tracker bound certificate (hash + witness)

Legality cases are scored against ``expected_legal``; certificate cases against
``expected_cert``; two-pass cases only against the chain verifier.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from qiskit import qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC    # noqa: E402
import support_tracker as ST              # noqa: E402
import experiments.compiler_bench.cases as C          # noqa: E402
import experiments.multipass_attestation.verify_chain as VC  # noqa: E402
import qcec_validate as QV                # noqa: E402

BENCH_VERSION = "compiler-bench-v1"
QCEEC = "/tmp/qcec-venv/bin/python"


def _dense(case: Dict[str, Any]) -> Optional[bool]:
    circuit = case["circuit"]
    if circuit.num_clbits or case["ideal"].num_qubits > 10:
        return None
    if any(ins.operation.name in ("measure", "reset", "if_else")
           for ins in circuit.data):
        return None
    try:
        return bool(RPC.relative_phase_ok(case["ideal"], circuit))
    except Exception:
        return None


def _qcec(case: Dict[str, Any]) -> Optional[bool]:
    circuit = case["circuit"]
    if circuit.num_clbits or case["ideal"].num_qubits > 18:
        return None
    try:
        pairs = QV._diag_pairs(0, case["ideal"], circuit)
    except Exception:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        results = QV._run_config(QCEEC, pairs, "alternating", 30, 60, Path(tmp))
    if any(r["verdict"] == "not_equivalent" for r in results):
        return False
    if all(r["verdict"] == "equivalent" for r in results):
        return True
    return None


def _cert(case: Dict[str, Any]) -> bool:
    if case["kind"] == "pass2":
        return bool(VC.verify_chain(case["entry"])["ok"])
    try:
        valid = bool(ST.verify_certificate(
            case["schedule"], case["mask"], case["certificate"],
            circuit=case["circuit"], block_ranges=case["ranges"]))
        return bool(valid and case["certificate"].get("accepted", False))
    except Exception:
        return False


def run() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in C.all_cases():
        row: Dict[str, Any] = {"case": case["name"],
                               "expected_legal": case["expected_legal"],
                               "expected_cert": case["expected_cert"],
                               "ground_truth": case["ground_truth"],
                               "kind": case["kind"]}
        started = time.perf_counter()
        dense = _dense(case) if case["kind"] == "rp" else None
        row["dense_legal"] = dense
        row["dense_s"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        qcec = _qcec(case) if case["kind"] == "rp" else None
        row["qcec_legal"] = qcec
        row["qcec_s"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        cert = _cert(case)
        row["cert_ok"] = cert
        row["cert_s"] = round(time.perf_counter() - started, 6)
        row["cert_bytes"] = len(json.dumps(case.get("certificate", {}))) \
            if case["kind"] == "rp" else None
        if case["expected_legal"] is not None and dense is not None:
            row["dense_correct"] = dense == case["expected_legal"]
        if case["expected_legal"] is not None and qcec is not None:
            row["qcec_correct"] = qcec == case["expected_legal"]
        row["cert_correct"] = cert == case["expected_cert"]
        rows.append(row)

    def rate(key: str) -> Dict[str, Any]:
        values = [r[key] for r in rows if key in r]
        return {"applicable": len(values),
                "correct": sum(1 for v in values if v),
                "rate": round(sum(1 for v in values if v) / len(values), 3)
                if values else None}

    metrics = {
        "dense": rate("dense_correct"),
        "qcec": rate("qcec_correct"),
        "certificate": rate("cert_correct"),
        "cert_false_positive": sum(
            1 for r in rows
            if r["expected_cert"] is True and r["cert_ok"] is False),
        "dense_false_positive": sum(
            1 for r in rows
            if r.get("expected_legal") is True and r.get("dense_legal") is False),
        "qcec_false_positive": sum(
            1 for r in rows
            if r.get("expected_legal") is True and r.get("qcec_legal") is False),
    }
    return {"version": BENCH_VERSION, "cases": len(rows),
            "metrics": metrics, "rows": rows}


def main() -> int:
    report = run()
    print("=" * 118)
    print(f"  compiler correctness/safety benchmark  {BENCH_VERSION}  "
          f"cases={report['cases']}")
    print("=" * 118)
    print(f"  {'case':<24}{'expected':>14}{'dense':>7}{'qcec':>7}{'cert':>7}"
          f"{'dense_s':>9}{'qcec_s':>8}{'cert_ms':>9}")
    for r in report["rows"]:
        expected = (f"L={r['expected_legal']}/C={r['expected_cert']}")
        print(f"  {r['case']:<24}{expected:>14}{str(r['dense_legal']):>7}"
              f"{str(r['qcec_legal']):>7}{str(r['cert_ok']):>7}"
              f"{r['dense_s']:>9.2f}{r['qcec_s']:>8.2f}"
              f"{r['cert_s'] * 1e3:>9.2f}")
    print("-" * 118)
    print(f"  metrics: {report['metrics']}")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
