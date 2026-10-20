#!/usr/bin/env python3
"""Scan a directory of QASM files with ``toffoli_recover`` (item 2/3).

For every circuit it reports

    has_ccx (before) | blocks recovered | in tracker domain
    windows tried    | recovery seconds | timed out

and aggregates: how many circuits have a ``ccx``, how many can be recovered,
how many enter the tracker domain, and the recovery cost distribution.

Usage
-----
    .venv/bin/python toffoli_scan.py --qasm-dir benchmarks/qasmbench
    .venv/bin/python toffoli_scan.py --qasm-dir <dir> --timeout 5 --max-window 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import qasm2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import toffoli_recover as TR          # noqa: E402

SCAN_VERSION = "toffoli-scan-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasm-dir", default=str(ROOT / "benchmarks" / "qasmbench"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-window", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "toffoli_recover_scan.json"))
    args = parser.parse_args(argv)
    paths = sorted(Path(args.qasm_dir).glob("*.qasm"))
    rows: List[Dict[str, Any]] = []
    print("=" * 110)
    print(f"  toffoli scan  version={SCAN_VERSION}  dir={args.qasm_dir}  "
          f"files={len(paths)}  timeout={args.timeout}s")
    print("=" * 110)
    print(f"  {'circuit':<16}{'gates':>6}{'ccx0':>5}{'rec':>4}{'out':>5}"
          f"{'domain':>8}{'windows':>9}{'sec':>8}  reason")
    for path in paths:
        try:
            circuit = qasm2.loads(path.read_text())
        except Exception as exc:
            row = {"circuit": path.stem, "error": f"{type(exc).__name__}: {exc}",
                   "has_ccx_before": None, "in_domain": False,
                   "windows_tried": 0, "seconds": 0.0, "timed_out": False,
                   "recovered": 0, "remaining_gates": None,
                   "remaining_histogram": {}, "equivalence_max_error": None,
                   "reason": "parse-error"}
            rows.append(row)
            print(f"  {row['circuit']:<16}{'-':>6}{'-':>5}{0:>4}{'-':>5}"
                  f"{'False':>8}{0:>9}{0.0:>8.3f}  parse-error")
            continue
        before = Counter(ins.operation.name for ins in circuit.data)
        out, report = TR.recover(path.read_text(), timeout_s=args.timeout,
                                 max_window=args.max_window)
        row = {"circuit": path.stem, "has_ccx_before": before.get("ccx", 0),
               "in_domain": report["in_tracker_domain"],
               "windows_tried": report["windows_tried"],
               "seconds": report["seconds"], "timed_out": report["timed_out"],
               "recovered": report["recovered_blocks"],
               "remaining_gates": report["remaining_gates"],
               "remaining_histogram": dict(Counter(i.operation.name for i in out.data)),
               "equivalence_max_error": report.get("equivalence_max_error"),
               "reason": report["unmatched_reason"] or "ok"}
        rows.append(row)
        print(f"  {row['circuit']:<16}{len(circuit.data):>6}{row['has_ccx_before']:>5}"
              f"{row['recovered']:>4}{row['remaining_gates']:>5}"
              f"{str(row['in_domain']):>8}{row['windows_tried']:>9}"
              f"{row['seconds']:>8.3f}  {row['reason']}")
    times = [r["seconds"] for r in rows]
    summary = {
        "circuits": len(rows),
        "with_ccx_before": sum(1 for r in rows
                               if (r["has_ccx_before"] or 0) > 0),
        "recovered_any": sum(1 for r in rows if r["recovered"] > 0),
        "in_domain": sum(1 for r in rows if r["in_domain"]),
        "timed_out": sum(1 for r in rows if r["timed_out"]),
        "windows_total": sum(r["windows_tried"] for r in rows),
        "seconds_total": round(sum(times), 3),
        "seconds_median": round(statistics.median(times), 4) if times else None,
        "seconds_max": round(max(times), 4) if times else None,
    }
    print("-" * 110)
    print(f"  summary: {summary}")
    out_path = Path(args.out)
    out_path.write_text(json.dumps({"version": SCAN_VERSION,
                                    "timeout_s": args.timeout,
                                    "max_window": args.max_window,
                                    "summary": summary, "rows": rows},
                                   indent=2), encoding="utf-8")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
