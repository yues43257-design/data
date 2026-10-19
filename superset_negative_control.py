#!/usr/bin/env python3
"""Reinforcement 4, final acceptance: negative control + expanded corpus.

NEGATIVE CONTROL (the missing piece):
    deliberately delete a propagation rule -- stop merging a block's ancillas --
    and assert that the superset test then FAILS.  A positive result whose test
    cannot fail is not evidence.

EXPANDED CORPUS:
    all N <= 12, several control-count mixes, ancilla-rich pools (2 and 4),
    h in {0.3, 0.7, 1.0}; assert ``violations == 0`` for the correct tracker.

Property under test (same as superset_check.py):
    for every mask the tracker ACCEPTS:
        true_support(U_cand . U_ideal^dagger)  subset of  tracked_support

Usage
-----
    .venv/bin/python superset_negative_control.py
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


ULT = _load("nc_ult", "ultimate_scalability_test.py")
WS = _load("nc_ws", "wide_scale_stress.py")
ST = _load("nc_st", "support_tracker.py")
SETUP = _load("nc_setup", "experiment_setup.py")


def tracked_support(schedule: Dict[str, Any], mask: int,
                    merge_ancillas: bool = True) -> Set[int]:
    support: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "cx" and int(qubits[1]) in support:
                support.add(int(qubits[0]))
        if block["target"] in support:
            support.update(int(q) for q in block["controls"])
            if merge_ancillas:
                support.update(int(q) for q in block["ancillas"])
        if (mask >> index) & 1:
            support.add(int(block["target"]))
            support.update(int(q) for q in block["controls"])
            if merge_ancillas:
                support.update(int(q) for q in block["ancillas"])
    return support


def true_support(u_ideal: np.ndarray, u_cand: np.ndarray, n: int) -> Set[int]:
    r = u_cand @ u_ideal.conj().T
    diag = np.diag(r)
    diag = diag / diag[0]
    out: Set[int] = set()
    x1 = np.array([[0, 1], [1, 0]])
    for q in range(n):
        # Qiskit is LITTLE-endian: qubit q is the LAST Kronecker factor.
        xq = np.kron(np.kron(np.eye(2 ** (n - q - 1)), x1), np.eye(2 ** q))
        if not np.allclose(r @ xq, xq @ r, atol=1e-9):
            out.add(q)
            continue
        idx = np.arange(2 ** n) ^ (1 << q)
        if np.any(np.abs(diag - diag[idx]) > 1e-9):
            out.add(q)
    return out


def run_corpus(merge_ancillas: bool, widths, pools, hs, blocks_list, seeds):
    checked = violations = 0
    examples: List[Dict[str, Any]] = []
    for width in widths:
        for pool in pools:
            for blocks in blocks_list:
                for h in hs:
                    for si in range(seeds):
                        seed = SETUP.seed_for(si) * 11 + width * 7 + blocks * 3 + pool
                        inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                                h_prob=h, ancilla_pool=pool)
                        if inst.total > 12:
                            continue
                        schedule = WS.build_schedule(inst)
                        base = np.asarray(Operator(inst.circuit(0)).data, dtype=complex)
                        for mask in range(1 << blocks):
                            if not ST.track_safe(schedule, mask):
                                continue
                            cand = np.asarray(Operator(inst.circuit(mask)).data,
                                              dtype=complex)
                            ts = true_support(base, cand, inst.total)
                            ks = tracked_support(schedule, mask, merge_ancillas)
                            checked += 1
                            if not ts.issubset(ks):
                                violations += 1
                                if len(examples) < 3:
                                    examples.append({"N": inst.total, "blocks": blocks,
                                                     "h": h, "pool": pool,
                                                     "mask": mask,
                                                     "true": sorted(ts),
                                                     "tracked": sorted(ks)})
    return checked, violations, examples


def main() -> int:
    print("=" * 94)
    print("  reinforcement 4: negative control + expanded corpus")
    print("=" * 94)
    # NEGATIVE CONTROL: ancillas NOT merged -> the test MUST fail
    n_checked, n_viol, _ = run_corpus(False, (4, 5), (2, 4), (0.7,), (2, 3), 2)
    print(f"  NEGATIVE CONTROL (ancillas not merged): checked={n_checked} "
          f"violations={n_viol}  -> {'discriminative' if n_viol else 'NOT discriminative'}")
    # CORRECT rules on the expanded corpus
    checked, viol, examples = run_corpus(True, (3, 4, 5, 6, 7), (2, 4),
                                         (0.3, 0.7, 1.0), (2, 3, 4), 3)
    print(f"  EXPANDED CORPUS (correct rules): checked={checked} violations={viol}")
    for ex in examples:
        print(f"    counterexample: {ex}")
    print("=" * 94)
    ok = (n_viol > 0) and (viol == 0)
    print(f"  NEGATIVE CONTROL {'PASS' if n_viol else 'FAIL'}; "
          f"SUPERSET PROPERTY {'HOLDS' if viol == 0 else 'VIOLATED'}")
    print(f"  REINFORCEMENT 4: {'ACCEPTED' if ok else 'NOT ACCEPTED'} "
          f"(needs discriminative control AND zero violations)")
    out = ROOT / "reports" / "figure_data" / "superset_negative_control.json"
    out.write_text(json.dumps({"negative_control_violations": n_viol,
                               "negative_control_checked": n_checked,
                               "corpus_violations": viol,
                               "corpus_checked": checked,
                               "examples": examples}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
