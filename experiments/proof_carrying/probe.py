#!/usr/bin/env python3
"""Cost probe: proof-log check vs dense vs QCEC.

Usage
-----
    .venv/bin/python experiments/proof_carrying/probe.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.proof_carrying import checker, phase_poly, rewrite  # noqa: E402

PROBE_VERSION = "proof-carrying-v1"
QCEEC = "/tmp/qcec-venv/bin/python"


def _dense_equal(a, b, tol: float = 1e-9):
    ua = np.asarray(Operator(a).data, dtype=complex)
    ub = np.asarray(Operator(b).data, dtype=complex)
    m = ua @ ub.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diagonal = np.diag(m)
    return bool(off < tol and float(np.abs(diagonal - diagonal[0]).max()) < tol)


def _qcec(a, b, timeout: int = 60):
    if not Path(QCEEC).exists():
        return None, None
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = Path(tmp) / "a.qasm", Path(tmp) / "b.qasm"
        pa.write_text(qasm2.dumps(a))
        pb.write_text(qasm2.dumps(b))
        started = time.perf_counter()
        try:
            proc = subprocess.run([QCEEC, "/tmp/qcec_one.py", str(pa), str(pb),
                                   str(timeout)],
                                  capture_output=True, text=True, timeout=timeout + 20)
            out = (proc.stdout or "").strip().splitlines()
            verdict = out[-1].split()[0] if out else "NO-OUTPUT"
        except subprocess.TimeoutExpired:
            verdict = "HARD-TIMEOUT"
        return verdict, round(time.perf_counter() - started, 3)


def run() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n, pairs in ((5, 2), (6, 3), (8, 4), (18, 6)):
        circuit = rewrite.build_instance(n, pairs)
        started = time.perf_counter()
        optimized, proof = rewrite.rewrite(circuit)
        rewrite_ms = (time.perf_counter() - started) * 1e3

        started = time.perf_counter()
        result = checker.check(circuit, optimized, proof)
        check_ms = (time.perf_counter() - started) * 1e3
        proof_bytes = len(json.dumps(proof))

        dense_ms = dense_ok = None
        if n <= 11:
            started = time.perf_counter()
            dense_ok = _dense_equal(circuit, optimized)
            dense_ms = (time.perf_counter() - started) * 1e3
        qcec_verdict, qcec_s = _qcec(circuit, optimized)

        tampered = json.loads(json.dumps(proof))
        if tampered["steps"]:
            tampered["steps"][0]["mask"] ^= 1
        tamper = checker.check(circuit, optimized, tampered)
        rows.append({
            "N": n, "pairs": pairs,
            "t_before": result["t_before"], "t_after": result["t_after"],
            "t_saved": result["t_before"] - result["t_after"],
            "steps": result["steps"], "proof_bytes": proof_bytes,
            "rewrite_ms": round(rewrite_ms, 3),
            "check_ms": round(check_ms, 3),
            "check_ok": result["ok"],
            "dense_ok": dense_ok, "dense_ms": (round(dense_ms, 2)
                                               if dense_ms is not None else None),
            "qcec_verdict": qcec_verdict, "qcec_s": qcec_s,
            "tampered_detected": not tamper["ok"],
        })
    return rows


def main() -> int:
    rows = run()
    print("=" * 104)
    print(f"  toy proof-carrying pass  {PROBE_VERSION}")
    print("=" * 104)
    print(f"  {'N':>3}{'T_before':>9}{'T_after':>8}{'steps':>7}{'proofB':>8}"
          f"{'rewrite_ms':>11}{'check_ms':>10}{'dense_ms':>9}{'qcec_s':>8}  ok")
    for r in rows:
        print(f"  {r['N']:>3}{r['t_before']:>9}{r['t_after']:>8}{r['steps']:>7}"
              f"{r['proof_bytes']:>8}{r['rewrite_ms']:>11.2f}"
              f"{r['check_ms']:>10.2f}{str(r['dense_ms']):>9}"
              f"{str(r['qcec_s']):>8}  {r['check_ok']}")
    verdict = ("GO" if all(r["check_ok"] and r["tampered_detected"] for r in rows)
               else "NO-GO")
    print("-" * 104)
    print(f"  all proofs check, tampering detected: {verdict}")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({"version": PROBE_VERSION, "rows": rows,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
