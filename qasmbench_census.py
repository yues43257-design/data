#!/usr/bin/env python3
"""Item 3: first pass over real QASMBench circuits.

These are **downloaded from pnnl/QASMBench** (branch ``master``) into
``benchmarks/qasmbench/``.  This is a census, not a full benchmark study: it
reports, per circuit, whether the circuit is inside the tracker's declared
domain (block-level Toffoli sites, no measurement/classical control), what the
all-relative and single-site masks are certified as, and -- where the width
allows -- the exact verdict.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from qiskit import qasm2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW          # noqa: E402
import realworld_benchmark as RB          # noqa: E402
import relative_phase_criterion as RPC    # noqa: E402
import support_tracker as ST              # noqa: E402

DIR = ROOT / "benchmarks" / "qasmbench"
EXACT_MAX = 11


def main() -> int:
    rows: List[Dict[str, Any]] = []
    print("=" * 104)
    print("  QASMBench census (source: pnnl/QASMBench, master)")
    print("=" * 104)
    print(f"  {'circuit':<14}{'q':>3}{'gates':>7}{'ccx':>5}{'H':>5}"
          f"{'blocks':>7}{'in-domain':>10}{'all-rel':>9}{'exact':>8}")
    for path in sorted(DIR.glob("*.qasm")):
        circuit = qasm2.loads(path.read_text())
        names = Counter(ins.operation.name for ins in circuit.data)
        schedule = RB._tracker_schedule(circuit)
        blocks = len(schedule["blocks"])
        has_measure = any(n in names for n in ("measure", "reset", "if_else"))
        in_domain = blocks > 0 and not has_measure
        verdict = exact = "-"
        if in_domain:
            mask = (1 << blocks) - 1
            try:
                cert = ST.certify(schedule, mask)
                verdict = "accept" if cert["accepted"] else "reject"
            except ST.UnsupportedGateError:
                in_domain, verdict = False, "unsupported"
            if circuit.num_qubits <= EXACT_MAX:
                positions = [i for i, ins in enumerate(circuit.data)
                             if ins.operation.name == "ccx"]
                candidate = circuit
                for position in sorted(positions, reverse=True):
                    candidate = RCW.replace_one_ccx(candidate, position)
                exact = "legal" if RPC.relative_phase_ok(circuit, candidate) else "illegal"
        rows.append({"circuit": path.stem, "qubits": circuit.num_qubits,
                     "gates": len(circuit.data),
                     "ccx": names.get("ccx", 0), "h": names.get("h", 0),
                     "blocks": blocks, "in_domain": in_domain,
                     "tracker_all_relative": verdict, "exact_all_relative": exact,
                     "gate_histogram": dict(names)})
        print(f"  {path.stem:<14}{circuit.num_qubits:>3}{len(circuit.data):>7}"
              f"{names.get('ccx', 0):>5}{names.get('h', 0):>5}{blocks:>7}"
              f"{str(in_domain):>10}{verdict:>9}{exact:>8}")
    out = ROOT / "reports" / "figure_data" / "qasmbench_census.json"
    out.write_text(json.dumps({"source": "pnnl/QASMBench@master", "rows": rows},
                              indent=2), encoding="utf-8")
    print("=" * 104)
    print(f"  in-domain: {sum(r['in_domain'] for r in rows)}/{len(rows)}   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
