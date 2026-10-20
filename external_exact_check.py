#!/usr/bin/env python3
"""Exact-checker comparison on the 24-100q workload (§5.53).

Question: can an off-the-shelf **exact** equivalence checker decide the
"up to diagonal" criterion at the sizes where the tracker is used?

The criterion is decided exactly by commuting with every Z_q:

    C = U_cand . U_ideal^dagger   is diagonal   <=>   C Z_q = Z_q C for all q

(Commuting with X_q would test that the phase is *independent* of bit q, i.e.
that C is scalar -- that was a wrong test and is recorded in §5.53.)

Each Z_q commutation is an ordinary circuit-equivalence query, so QCEC can
answer it.  A full check therefore needs ``n`` QCEC runs; this script measures
one run per size (the cost per qubit) plus the wall clock to the first failure.

Tool availability (measured, 2026-10-19)
  * ``mqt.qcec`` 3.9.0  -- on PyPI, used below.
  * Manjushri           -- not on PyPI; not obtainable in this environment.
  * Pandora (quantum)   -- the PyPI name is an unrelated computer-vision
                           package; not obtainable.
  * phase-polynomial    -- see ``pp_exact_checker.py``: 0/12 applicability.

Usage
-----
    .venv/bin/python external_exact_check.py \
        --qcec-python /tmp/qcec-venv/bin/python      # an interpreter with mqt.qcec
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ultimate_scalability_test as ULT      # noqa: E402
import support_tracker as ST                 # noqa: E402
import wide_scale_stress as WS               # noqa: E402

CHECKER = "external-exact-v1"
CASES = ((4, 6, 6), (6, 2, 6), (12, 4, 6), (16, 2, 6), (24, 6, 10))

_ONE = """
import sys, time, mqt.qcec as qcec
t0 = time.perf_counter()
try:
    r = qcec.verify(sys.argv[1], sys.argv[2], timeout=int(sys.argv[3]))
    print(str(r.equivalence).split('.')[-1], round(time.perf_counter() - t0, 3))
except Exception as exc:
    print("EXC:" + type(exc).__name__, round(time.perf_counter() - t0, 3))
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcec-python", default="/tmp/qcec-venv/bin/python")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--hard-timeout", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "external_exact_check.json"))
    args = parser.parse_args(argv)
    python = Path(args.qcec_python)
    if not python.exists():
        print(f"  QCEC interpreter not found: {python}")
        print("  install with: python3 -m venv /tmp/qcec-venv && "
              "/tmp/qcec-venv/bin/pip install mqt.qcec qiskit")
        return 2

    rows: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for width, pool, blocks in CASES:
            inst = ULT.WideInstance(width=width, blocks=blocks, seed=3,
                                    h_prob=0.0, ancilla_pool=pool)
            ideal, cand = inst.circuit(0), inst.circuit((1 << blocks) - 1)
            n = ideal.num_qubits
            correction = QuantumCircuit(n)
            # compose() appends, so the declared C = U_cand . U_ideal^dagger
            # requires the inverse first (see qcec_validate.py).
            correction.compose(ideal.inverse(), inplace=True)
            correction.compose(cand, inplace=True)
            left, right = QuantumCircuit(n), QuantumCircuit(n)
            left.z(0)
            left.compose(correction, inplace=True)      # C . Z_0
            right.compose(correction, inplace=True)
            right.z(0)                                   # Z_0 . C
            a = tmpdir / f"zA_{n}.qasm"
            b = tmpdir / f"zB_{n}.qasm"
            a.write_text(qasm2.dumps(left))
            b.write_text(qasm2.dumps(right))

            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    [str(python), "-c", _ONE, str(a), str(b), str(args.timeout)],
                    capture_output=True, text=True, timeout=args.hard_timeout)
                out = (proc.stdout or "").strip().splitlines()
                verdict, seconds = (out[-1].split()[0], float(out[-1].split()[1])) \
                    if out else ("NO-OUTPUT", time.perf_counter() - started)
            except subprocess.TimeoutExpired:
                verdict, seconds = "HARD-TIMEOUT", time.perf_counter() - started

            schedule = WS.build_schedule(inst)
            mask = (1 << blocks) - 1
            t0 = time.perf_counter()
            safe = bool(ST.track_safe(schedule, mask))
            tracker_s = time.perf_counter() - t0
            rows.append({"N": n, "gates": len(left.data), "qcec_verdict": verdict,
                         "qcec_seconds": round(seconds, 3), "tracker_safe": safe,
                         "tracker_seconds": round(tracker_s, 6)})
            print(f"  N={n:>3} gates={len(left.data):>5}  QCEC={verdict:<14}"
                  f"{seconds:8.2f}s   tracker={tracker_s * 1e3:7.3f} ms", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps({
        "checker": CHECKER,
        "criterion": "C diagonal <=> [C, Z_q] = 0 for all q (one QCEC call per qubit)",
        "tools": {"mqt.qcec": "3.9.0", "manjushri": "not on PyPI",
                  "pandora": "PyPI name is an unrelated CV package",
                  "phase_polynomial": "see pp_exact_checker.py (0/12 applicable)"},
        "timeout_s": args.timeout, "hard_timeout_s": args.hard_timeout,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
