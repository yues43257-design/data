#!/usr/bin/env python3
"""Step 3: recompute the §6 legality column under the DECLARED definition.

The declared definition (frozen in §5.23) is the clean-ancilla post-selection
criterion ``C = U_cand . U_ideal^dagger diagonal``, implemented by
``relative_phase_criterion`` and already wired into
``realworld_benchmark.relative_phase_ok``.

This script prints, for every circuit in the §6 table, the OLD verdict (stored in
``reports/figure_data/realworld_sota.json``, produced by the wrong
right-multiplied convention) next to the NEW verdict, so the change is explicit
rather than silently overwritten.  Circuits above the exact limit stay ``n/a``
(the tracker takes over there).

Usage
-----
    .venv/bin/python recompute_legality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
from qiskit import qasm2                   # noqa: E402
from qiskit.circuit.library import CDKMRippleCarryAdder  # noqa: E402

EXACT_LIMIT = 11


def main() -> int:
    stored = {row["name"]: row for row in json.load(
        open(ROOT / "reports" / "figure_data" / "realworld_sota.json"))}

    circuits = [
        ("mcx(3)", "ladder", 3), ("mcx(4)", "ladder", 4), ("mcx(5)", "ladder", 5),
        ("adder(3)", "toffoli", 3), ("adder(4)", "toffoli", 4),
        ("grover(3)", "toffoli", 3),
    ]
    print("=" * 96)
    print("  §6 legality recomputed under the declared clean-ancilla definition")
    print("=" * 96)
    print(f"  {'circuit':<12}{'q':>4}{'OLD (right-mult)':>19}{'NEW (declared)':>17}"
          f"   change")
    flips = []
    for label, mode, size in circuits:
        if mode == "ladder":
            qc = None
            reference = RB.circuit_mcx(size)
            ours = RB.LM.relative_phase_mcx_circuit(int(size))
        else:
            qc = CDKMRippleCarryAdder(size) if label.startswith("adder") else \
                RB.circuit_grover(size)
            canonical = RB.to_canonical(qc)
            reference = canonical
            ours = RB.substitute_rccx(canonical)
        old = stored.get(label, {}).get("relative_phase_ok")
        new = RB.relative_phase_ok(reference, ours, EXACT_LIMIT)
        if old != new:
            flips.append((label, old, new))
        print(f"  {label:<12}{reference.num_qubits:>4}{str(old):>19}{str(new):>17}"
              f"   {'<-- FLIPPED' if old != new else ''}")
    print("=" * 96)
    print(f"  flips: {len(flips)}  ->  {flips}")
    print("  note: adder(8/16/32) sit above the exact limit; the tracker certifies them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
