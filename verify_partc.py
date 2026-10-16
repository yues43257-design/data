#!/usr/bin/env python3
"""Two-stage verification of the Part-C counterexamples.

STAGE 1 - domain membership.  The declared domain requires the ancillas to be
uncomputed back to |0> and reusable, i.e. U_ideal must map the clean-ancilla
subspace into itself.  If it does not, the counterexample is OUT OF DOMAIN and
the comparison is void (the clean-subspace criterion does not apply).

STAGE 2 - illegality.  Only if stage 1 passes: compute
``C = U_cand . U_ideal^dagger`` restricted to the clean subspace and measure
``max|C - diag(C)|``.  Non-diagonal => real bug; diagonal => harness artefact.

Also prints the tracker's final support set for the mask, so we can see whether
ancilla indices ever enter it.

Usage
-----
    .venv/bin/python verify_partc.py
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


RO = _load("vpc_ro", "remaining_options.py")
ST = _load("vpc_st", "support_tracker.py")
SETUP = _load("vpc_setup", "experiment_setup.py")
WS = _load("vpc_ws", "wide_scale_stress.py")


def tracker_support(schedule: Dict[str, Any], mask: int) -> Set[int]:
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


def main() -> int:
    data = json.loads((ROOT / "reports" / "figure_data" /
                       "remaining_options.json").read_text())
    examples: List[Dict[str, Any]] = data["part_c"]["examples"]
    print("=" * 96)
    print(f"  two-stage verification of Part-C counterexamples "
          f"({len(examples)} examples available; checking up to 4)")
    print("=" * 96)
    for ex in examples[:4]:
        width, blocks, h_prob, seed, mask = (ex["width"], ex["blocks"],
                                             ex["h"], ex["seed"], ex["mask"])
        inst = RO.build_wide_with_ancilla_h(width, blocks, seed, h_prob)
        total = inst.total
        ancillas = list(range(width, total))
        clean = [x for x in range(2 ** total)
                 if all(not (x >> a) & 1 for a in ancillas)]
        dirty = [x for x in range(2 ** total) if x not in set(clean)]
        u_ideal = np.asarray(Operator(inst.circuit(0)).data, dtype=complex)
        # STAGE 1: does the clean subspace stay clean?
        leak = float(np.abs(u_ideal[np.ix_(dirty, clean)]).max())
        in_domain = leak < 1e-9
        # STAGE 2: only meaningful if in-domain
        u_cand = np.asarray(Operator(inst.circuit(mask)).data, dtype=complex)
        c = u_cand[np.ix_(clean, clean)] @ u_ideal[np.ix_(clean, clean)].conj().T
        off = float(np.abs(c - np.diag(np.diag(c))).max())
        support = tracker_support(WS.build_schedule(inst), mask)
        anc_in_support = sorted(set(support) & set(ancillas))
        verdict = ("OUT OF DOMAIN -> comparison void"
                   if not in_domain else
                   ("REAL BUG (C non-diagonal)" if off > 1e-9
                    else "artefact (C diagonal)"))
        print(f"  N={total} blocks={blocks} h={h_prob} mask={mask}")
        print(f"     stage1 leak |U[dirty,clean]| = {leak:.3e}  "
              f"in_domain={in_domain}")
        print(f"     stage2 max|offdiag C| (clean subspace) = {off:.3e}")
        print(f"     tracker support = {sorted(support)}  "
              f"ancillas = {ancillas}  anc_in_support = {anc_in_support}")
        print(f"     -> {verdict}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
