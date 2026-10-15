#!/usr/bin/env python3
"""Decide: is the tracker unsound, or is the oracle convention wrong?

``structured_h_check`` found 34 sites where the tracker ACCEPTS but
``relative_phase_ok`` REJECTS -- which contradicts the soundness lemma, so one of
the two is wrong.  This settles it with a convention-free check:

    build U_ideal and U_cand as full operators, form C = U_cand . U_ideal^dagger,
    measure max|C - diag(C)|.

    C diagonal      -> tracker right, ``relative_phase_ok`` uses another
                       criterion                                      -> case (b)
    C NOT diagonal  -> the tracker accepted an illegal rewrite         -> case (a)

It also prints what ``relative_phase_metrics`` returns, so the convention gap is
visible rather than guessed.

Usage
-----
    .venv/bin/python oracle_discrepancy_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qiskit import qasm2
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
import realcircuit_window as RCW          # noqa: E402
import support_tracker as ST              # noqa: E402
import wide_h_window as WH                # noqa: E402

LM = RB.LM


def main() -> int:
    print("=" * 100)
    print("  tracker-accepts vs exact-rejects: which one is right?")
    print("=" * 100)
    print(f"  {'site':>4}{'N':>4}{'tracker':>9}{'rel_phase_ok':>13}"
          f"{'max|offdiag(C)|':>17}{'rel_phase_metrics':>26}")

    found = 0
    for n in (3, 4):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        circuit = WH.interleave_h_layers(canon, 1, canon.num_qubits)  # H every ccx
        schedule = RB._tracker_schedule(circuit)
        ccx_positions = [i for i, ins in enumerate(circuit.data)
                         if ins.operation.name == "ccx"]
        u_ideal = Operator(circuit).data
        for site, position in enumerate(ccx_positions):
            mask = 1 << site
            accepted = bool(ST.track_safe(schedule, mask))
            candidate = RCW.replace_one_ccx(circuit, position)
            ok = RB.relative_phase_ok(circuit, candidate, 11)
            if not (accepted and not ok):
                continue
            found += 1
            c = Operator(candidate).data @ u_ideal.conj().T
            off = float(np.abs(c - np.diag(np.diag(c))).max())
            ref = LM.zx.Circuit.from_qasm(qasm2.dumps(circuit))
            trial = LM.zx.Circuit.from_qasm(qasm2.dumps(candidate))
            metrics = LM.relative_phase_metrics(ref, trial, ancillas=[])
            print(f"  {site:>4}{circuit.num_qubits:>4}{str(accepted):>9}"
                  f"{str(ok):>13}{off:>17.3e}   {metrics}")
            if found >= 8:
                break
        if found >= 8:
            break

    print("=" * 100)
    print("  case (b) if max|offdiag(C)| ~ 0  -> tracker right, oracle convention differs")
    print("  case (a) if max|offdiag(C)| >> 0 -> tracker UNSOUND, lemma must be revisited")
    print(f"  inspected {found} such sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
