#!/usr/bin/env python3
"""Is the wide-circuit acceptance drop conservativeness, or real illegality?

``wide_h_window.py`` measured tracker *acceptance* at N=66 where no exact oracle
exists, and I wrongly read it as "conservativeness".  The reviewer's point: in a
*structured* H layout the H layers can restore diagonality, so the rejected sites
may well be legal, and the acceptance drop would then be a property of v1 rather
than of the rewrite.

This script shrinks the SAME structured family to ``N <= 11`` so the exact
verifier is available, and measures

    truth    = per-site legality by the exact verifier
    v1       = per-site acceptance by support_tracker
    conserv. = 1 - v1 / truth

Pre-registered (agreed before running): if, in the declared domain, sound-v2
recovers >= 40% of the "legal but v1-rejected" sites with UNSOUND = 0, build v2;
otherwise close it permanently.  Either way we then go to writing.

Usage
-----
    .venv/bin/python structured_h_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
import realcircuit_window as RCW          # noqa: E402
import support_tracker as ST              # noqa: E402
import wide_h_window as WH                # noqa: E402

EXACT_LIMIT = 11


def measure(label: str, canon, every: int) -> Optional[Dict[str, Any]]:
    circuit = (WH.interleave_h_layers(canon, every, canon.num_qubits)
               if every else canon)
    if circuit.num_qubits > EXACT_LIMIT:
        print(f"  [skip] {label}: {circuit.num_qubits}q exceeds exact limit")
        return None
    schedule = RB._tracker_schedule(circuit)
    ccx_positions = [i for i, ins in enumerate(circuit.data)
                     if ins.operation.name == "ccx"]
    truth = track = 0
    for site, position in enumerate(ccx_positions):
        mask = 1 << site
        track += int(bool(ST.track_safe(schedule, mask)))
        ok = RB.relative_phase_ok(circuit,
                                  RCW.replace_one_ccx(circuit, position),
                                  EXACT_LIMIT)
        truth += int(bool(ok))
    sites = len(ccx_positions)
    conserv = (1 - track / truth) if truth else float("nan")
    print(f"  {label:<34}{circuit.num_qubits:>5}{sites:>7}{truth:>8}{track:>7}"
          f"{conserv:>10.2f}")
    return {"variant": label, "N_total": circuit.num_qubits, "sites": sites,
            "truth_legal": truth, "v1_accept": track, "conservativeness": conserv}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="structured-H truth vs v1.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "structured_h_check.json"))
    args = parser.parse_args(argv)

    print("=" * 92)
    print("  structured-H family at exact-oracle size: truth vs tracker v1")
    print("  (CDKM adder, H layer every N ccx, full-width H layers)")
    print("=" * 92)
    print(f"  {'variant':<34}{'N_tot':>5}{'sites':>7}{'truth':>8}{'v1':>7}"
          f"{'conserv':>10}")
    rows: List[Dict[str, Any]] = []
    for n in (3, 4):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        for label, every in ((f"adder{n}-bit H-free", 0),
                             (f"adder{n}-bit H every 2 ccx", 2),
                             (f"adder{n}-bit H every 1 ccx", 1)):
            row = measure(label, canon, every)
            if row is not None:
                rows.append(row)
    print("=" * 92)
    print("  truth = exact per-site legality; v1 = support_tracker acceptance")
    print("  conserv = 1 - v1/truth  (0 => tracker is exact in this domain)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"script": "structured_h_check.py", "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
