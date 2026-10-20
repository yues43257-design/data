#!/usr/bin/env python3
"""Task 2: QCEC boundary at N = 18 / 24 / 30 / 56 / 106.

Reports, for both a legal mask and a guaranteed-illegal one:

  * ``single_query_s``  -- median time of one ``[C, Z_q]`` equivalence query;
  * ``full_check_s``    -- legal: all ``n`` queries; illegal: time to the first
    ``not_equivalent`` (the filter's actual cost, since it can stop there);
  * ``n_queries`` / ``first_failing_q``.

The configuration is ``alternating`` only: §5.53 showed ``zx_only`` can give a
false ``probably_not_equivalent`` and ``construct`` (DD) returns
``no_information`` at N >= 24.

Usage
-----
    .venv/bin/python qcec_boundary.py --qcec-python /tmp/qcec-venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import qcec_validate as QV          # noqa: E402
import ultimate_scalability_test as ULT  # noqa: E402

BOUNDARY_VERSION = "qcec-boundary-v1"
SIZES = ((16, 2), (18, 6), (24, 6), (50, 6), (100, 6))   # -> N = 18..106


def run_case(inst, kind: str, python: str, timeout: float, hard: float,
             tmpdir: Path) -> Dict[str, Any]:
    ideal = inst.circuit(0)
    candidate = inst.circuit((1 << inst.blocks) - 1)
    if kind == "illegal":
        candidate = candidate.copy()
        candidate.h(inst.targets[-1])
    pairs = QV._diag_pairs(0, ideal, candidate)
    n = inst.total
    # both kinds scan every qubit: a legal mask needs all n, and an illegal one
    # needs to find *some* failing q (which may be far from q=0).  At 0.08 s per
    # query even N=106 costs ~9 s, so there is no reason to cap.
    used = pairs
    started = time.perf_counter()
    results = QV._run_config(python, used, "alternating", timeout, hard, tmpdir)
    total = time.perf_counter() - started
    per_query = [r["seconds"] for r in results]
    first_fail = next((i for i, r in enumerate(results)
                       if r["verdict"] == "not_equivalent"), None)
    if kind == "legal":
        full = sum(per_query) if all(r["verdict"] == "equivalent" for r in results) else None
    else:
        full = sum(per_query[: first_fail + 1]) if first_fail is not None else None
    return {
        "kind": kind, "N": n, "queries": len(used),
        "single_query_s": round(statistics.median(per_query), 4),
        "full_check_s": round(full, 3) if full is not None else None,
        "verdict": ("all_equivalent" if kind == "legal" and full is not None
                    else ("not_equivalent" if first_fail is not None else "undecided")),
        "first_failing_q": first_fail,
        "wall_clock_s": round(total, 2),
        "verdicts": [r["verdict"] for r in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcec-python", default="/tmp/qcec-venv/bin/python")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--hard-timeout", type=float, default=90)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "qcec_boundary.json"))
    args = parser.parse_args(argv)
    if not Path(args.qcec_python).exists():
        print(f"QCEC interpreter not found: {args.qcec_python}")
        return 2
    rows: List[Dict[str, Any]] = []
    print("=" * 100)
    print(f"  QCEC boundary  version={BOUNDARY_VERSION}  config=alternating")
    print(f"  {'N':>4}{'kind':>9}{'q':>5}{'single(q)':>12}{'full':>10}"
          f"{'firstFail':>11}  verdict")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for width, pool in SIZES:
            for kind in ("legal", "illegal"):
                inst = ULT.WideInstance(width=width, blocks=6, seed=3,
                                        h_prob=0.0, ancilla_pool=pool)
                row = run_case(inst, kind, args.qcec_python, args.timeout,
                               args.hard_timeout, tmpdir)
                rows.append(row)
                print(f"  {row['N']:>4}{kind:>9}{row['queries']:>5}"
                      f"{row['single_query_s']:>12.3f}"
                      f"{(row['full_check_s'] if row['full_check_s'] is not None else -1):>10.2f}"
                      f"{str(row['first_failing_q']):>11}  {row['verdict']}", flush=True)
    out = Path(args.out)
    out.write_text(json.dumps({"boundary_version": BOUNDARY_VERSION,
                               "config": "alternating", "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
