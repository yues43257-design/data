#!/usr/bin/env python3
"""Reinforcement ④: end-to-end soundness as a *testable superset property*.

The lemma says "support avoids H => the residual stays diagonal => legal".  That
argument has two halves:

  (i)  the mathematics (diagonal conjugation by a basis permutation stays
       diagonal; only H-class gates break it) -- proven by hand;
  (ii) the IMPLEMENTATION: the set the tracker propagates must be a **superset**
       of the qubits the residual actually depends on.  If the bound missed a
       qubit, a downstream H on that qubit would go unnoticed -> unsound.

This script tests (ii) exhaustively on small circuits: for every mask it compares
the tracker's final support with the TRUE dependency set extracted from
``R = U_cand . U_ideal^dagger`` (a qubit is depended on iff R fails to commute
with X_q, or its diagonal depends on that bit).

    PASS  <=>  true_support ⊆ tracked_support   for every sampled mask

Together with (i) this upgrades "sound by lemma + assumption" into an
end-to-end statement: *this implementation* is sound wherever it returns True.

Usage
-----
    .venv/bin/python superset_check.py
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


ULT = _load("ss_ult", "ultimate_scalability_test.py")
ST = _load("ss_st", "support_tracker.py")
WS = _load("ss_ws", "wide_scale_stress.py")
SETUP = _load("ss_setup", "experiment_setup.py")


def tracked_support(schedule: Dict[str, Any], mask: int) -> Set[int]:
    support: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "cx" and int(qubits[1]) in support:
                support.add(int(qubits[0]))
        if block["target"] in support:
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
        if (mask >> index) & 1:
            support.add(int(block["target"]))
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
    return support


def true_support(u_ideal: np.ndarray, u_cand: np.ndarray, n: int) -> Set[int]:
    r = u_cand @ u_ideal.conj().T
    diag = np.diag(r)
    diag = diag / diag[0]
    out: Set[int] = set()
    for q in range(n):
        # Qiskit is LITTLE-endian: qubit q carries weight 2**q, so it is the
        # LAST factor of the Kronecker product (the reflected index was a bug).
        x = np.kron(np.kron(np.eye(2 ** (n - q - 1)), np.array([[0, 1], [1, 0]])),
                    np.eye(2 ** q))
        if not np.allclose(r @ x, x @ r, atol=1e-9):
            out.add(q)
            continue
        idx = np.arange(2 ** n) ^ (1 << q)
        if np.any(np.abs(diag - diag[idx]) > 1e-9):
            out.add(q)
    return out


def main() -> int:
    total = violations = skipped = 0
    examples: List[Dict[str, Any]] = []
    for width in (3, 4, 5):
        for blocks in (2, 3, 4):
            for h_prob in (0.5, 1.0):
                for seed_index in range(2):
                    seed = SETUP.seed_for(seed_index) * 13 + width * 5 + blocks
                    inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                            h_prob=h_prob, ancilla_pool=2)
                    if inst.total > 11:
                        skipped += 1
                        continue
                    schedule = WS.build_schedule(inst)
                    base = np.asarray(Operator(inst.circuit(0)).data, dtype=complex)
                    for mask in range(1 << blocks):
                        if not ST.track_safe(schedule, mask):
                            continue          # rejected: support set is irrelevant
                        cand = np.asarray(Operator(inst.circuit(mask)).data,
                                          dtype=complex)
                        ts = true_support(base, cand, inst.total)
                        ks = tracked_support(schedule, mask)
                        total += 1
                        if not ts.issubset(ks):
                            violations += 1
                            if len(examples) < 3:
                                examples.append({"width": width, "blocks": blocks,
                                                 "h": h_prob, "mask": mask,
                                                 "true": sorted(ts),
                                                 "tracked": sorted(ks)})
    print("=" * 92)
    print("  reinforcement 4: does the tracker's support always COVER the truth?")
    print("=" * 92)
    print(f"  masks checked          : {total}")
    print(f"  instances skipped (>11q): {skipped}")
    print(f"  true_not_subset_of_tracked : {violations}")
    for ex in examples:
        print(f"    counterexample: {ex}")
    print("=" * 92)
    print(f"  SUPERSET PROPERTY: {'HOLDS on every checked mask' if violations == 0 else 'VIOLATED'}")
    out = ROOT / "reports" / "figure_data" / "superset_check.json"
    out.write_text(json.dumps({"masks": total, "violations": violations,
                               "examples": examples}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
