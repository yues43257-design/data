#!/usr/bin/env python3
"""Task 2: run the tail-layer regression over the 9 SOTA circuits of §6.2.

For every ccx site (single-site mask) report
  * the number of Hadamards in the TAIL layer (gates after the last block);
  * old (tail-dropped) vs new (tail-applied) ``track_safe``;
  * UNSOUND counts against the dense criterion, where the width allows it.

The point is not to re-run §6, it is to prove the tail fix changes nothing on
the circuits the paper actually reports -- and to name the ones where the exact
oracle is unavailable (so "no change" there is a structural statement, not a
measured one).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import relative_phase_criterion as RPC  # noqa: E402
import support_tracker as ST          # noqa: E402

EXACT_MAX_QUBITS = 11


def old_track_safe(schedule: Dict[str, Any], mask: int) -> bool:
    tracker = ST.DiagonalTracker(schedule["total"])
    for index, block in enumerate(schedule["blocks"]):
        tracker.apply_layer(schedule["layers"][index])
        tracker.apply_block(block)
        if (mask >> index) & 1:
            tracker.inject_correction(block)
    return tracker.is_safe()


def cases() -> List[tuple[str, Any]]:
    return [
        ("C3X ladder", RB.circuit_mcx(3)),
        ("C4X ladder", RB.circuit_mcx(4)),
        ("C5X ladder", RB.circuit_mcx(5)),
        ("CDKM adder 3-bit", RB.to_canonical(CDKMRippleCarryAdder(3))),
        ("CDKM adder 4-bit", RB.to_canonical(CDKMRippleCarryAdder(4))),
        ("CDKM adder 8-bit", RB.to_canonical(CDKMRippleCarryAdder(8))),
        ("CDKM adder 16-bit", RB.to_canonical(CDKMRippleCarryAdder(16))),
        ("CDKM adder 32-bit", RB.to_canonical(CDKMRippleCarryAdder(32))),
        ("Grover 3-qubit", RB.circuit_grover(3)),
    ]


def main() -> int:
    rows: List[Dict[str, Any]] = []
    print("=" * 104)
    print("  tail-layer regression on the 9 SOTA circuits")
    print("=" * 104)
    print(f"  {'circuit':<20}{'q':>4}{'sites':>6}{'tailH':>7}{'flips':>7}"
          f"{'oldUNS':>8}{'newUNS':>8}  exact?")
    for name, circuit in cases():
        schedule = RB._tracker_schedule(circuit)
        positions = [i for i, ins in enumerate(circuit.data)
                     if ins.operation.name == "ccx"]
        n_blocks = len(schedule["blocks"])
        tail_h = (sum(1 for g, _ in schedule["layers"][-1] if g == "h")
                  if len(schedule["layers"]) > n_blocks else 0)
        flips = old_uns = new_uns = 0
        exact = circuit.num_qubits <= EXACT_MAX_QUBITS
        for site, position in enumerate(positions):
            mask = 1 << site
            old, new = old_track_safe(schedule, mask), ST.track_safe(schedule, mask)
            flips += int(old != new)
            if exact:
                legal = bool(RPC.relative_phase_ok(
                    circuit, RCW.replace_one_ccx(circuit, position)))
                old_uns += int(old and not legal)
                new_uns += int(new and not legal)
        row = {"circuit": name, "qubits": circuit.num_qubits, "sites": len(positions),
               "tail_h": tail_h, "flips": flips,
               "blocks_in_schedule": n_blocks,
               "tracker_covers": n_blocks > 0,
               "old_unsound": old_uns if exact else None,
               "new_unsound": new_uns if exact else None,
               "exact_checked": exact}
        rows.append(row)
        uns = f"{old_uns:>8}{new_uns:>8}" if exact else f"{'n/a':>8}{'n/a':>8}"
        coverage = "" if n_blocks else "   [OUTSIDE tracker: 0 ccx blocks]"
        print(f"  {name:<20}{circuit.num_qubits:>4}{len(positions):>6}{tail_h:>7}"
              f"{flips:>7}{uns}  {'yes' if exact else 'no (>11q)'}{coverage}")
    out = ROOT / "reports" / "figure_data" / "tail_regression_sota.json"
    out.write_text(json.dumps({"exact_max_qubits": EXACT_MAX_QUBITS, "rows": rows},
                              indent=2), encoding="utf-8")
    print("=" * 104)
    print(f"  total sites={sum(r['sites'] for r in rows)}  "
          f"tailH={sum(r['tail_h'] for r in rows)}  "
          f"flips={sum(r['flips'] for r in rows)}  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
