#!/usr/bin/env python3
"""One command, one version, one exit code (task 3.7).

Runs the fast acceptance suite:

    1. test_tracker_soundness.py   6 regression tests (tail, HTH, CNOT direction,
                                   known-unsound corpus, unmodelled gates, certificate)
    2. tail_regression_sota.py     9 SOTA circuits, tail H / flips / unsound
    3. tail_fuzz.py                200 random circuits + 40 ancilla-layout circuits
    4. qasmbench_census.py         5 real QASMBench circuits (domain boundary)

``--with-qcec`` additionally runs ``qcec_boundary.py`` (needs the QCEC venv).

Usage
-----
    .venv/bin/python frozen_ci.py
    .venv/bin/python frozen_ci.py --with-qcec --qcec-python /tmp/qcec-venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
CI_VERSION = "frozen-ci-v1"


def run(step: str, cmd: List[str], timeout_s: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=timeout_s)
        ok, tail = proc.returncode == 0, (proc.stdout or "").strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        ok, tail = False, ["HARD-TIMEOUT"]
    return {"step": step, "ok": ok, "seconds": round(time.perf_counter() - started, 1),
            "tail": tail}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-qcec", action="store_true")
    parser.add_argument("--qcec-python", default="/tmp/qcec-venv/bin/python")
    args = parser.parse_args(argv)
    py = str(Path(sys.executable))
    steps = [
        ("regressions", [py, "test_tracker_soundness.py"], 600),
        ("sota_tail", [py, "tail_regression_sota.py"], 600),
        ("tail_fuzz", [py, "tail_fuzz.py", "--circuits", "200"], 1800),
        ("qasmbench", [py, "qasmbench_census.py"], 600),
    ]
    if args.with_qcec:
        steps.append(("qcec_boundary", [py, "qcec_boundary.py",
                                        "--qcec-python", args.qcec_python], 3600))

    print("=" * 96)
    print(f"  FROZEN CI  version={CI_VERSION}  steps={len(steps)}")
    print("=" * 96)
    results = []
    for step, cmd, timeout_s in steps:
        result = run(step, cmd, timeout_s)
        results.append(result)
        print(f"  [{'PASS' if result['ok'] else 'FAIL'}] {step:<16}"
              f"{result['seconds']:>7.1f}s   {result['tail'][-1] if result['tail'] else ''}")
    ok = all(r["ok"] for r in results)
    print("=" * 96)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}  ({sum(r['seconds'] for r in results):.0f}s)")
    out = ROOT / "reports" / "figure_data" / "frozen_ci.json"
    out.write_text(json.dumps({"version": CI_VERSION, "ok": ok,
                               "steps": results}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
