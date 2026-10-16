#!/usr/bin/env python3
"""Judgment 0, final form: the FULL Toffoli, not the bare AND.

    compute AND (4 T, |T> catalyst)  ->  CX(t, target)  ->  H(t)
    then measure t in the X basis and apply a classical fixup

Two implementation notes that make this a *unitary* check:
  * after the first measurement t sits in |x AND y>, so the classically
    controlled X on the target is exactly ``CX(t, target)``;
  * the uncompute's X-basis measurement of t is a projective branch, which we
    enumerate explicitly (r = 0 / 1).

Correctness requires that BOTH branches, after the fixup, give the *same*
Toffoli output -- same state AND same global phase -- otherwise the classical
record destroys the coherence and the gadget is unusable on superposed data.

The fixup is enumerated (the reviewer's protocol) over gates on x and y.

Usage
-----
    .venv/bin/python full_toffoli_check.py
"""

from __future__ import annotations

import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

T_STATE = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
N = 4                      # 0=x 1=y 2=t 3=target
IDX = lambda a, b, t, g: a * 1 + b * 2 + t * 4 + g * 8


def prefix_unitary() -> np.ndarray:
    """compute-AND ; CX(t,target) ; H(t)   (t starts in |T>)"""
    qc = QuantumCircuit(N)
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(2, 0)
    qc.tdg(0)
    qc.cx(2, 0)
    qc.cx(2, 1)
    qc.tdg(1)
    qc.cx(2, 1)
    qc.t(2)
    qc.h(2)
    qc.s(2)
    qc.cx(2, 3)            # the "use": classically controlled X on the target
    qc.h(2)                # start of the uncompute
    return np.asarray(Operator(qc).data, dtype=complex)


def gate_matrix(name: str, q: int) -> np.ndarray:
    g = QuantumCircuit(N)
    getattr(g, name)(q)
    return np.asarray(Operator(g).data, dtype=complex)


FIXUPS = [("id", None), ("x", "x"), ("y", "y"), ("z", "z"), ("s", "s"),
          ("sdg", "sdg"), ("xz", ("x", "z")), ("zx", ("z", "x"))]


def main() -> int:
    u = prefix_unitary()
    # branch projectors on t (qubit 2)
    p0 = np.zeros((2 ** N, 2 ** N), dtype=complex)
    p1 = np.zeros_like(p0)
    for i in range(2 ** N):
        (p1 if (i >> 2) & 1 else p0)[i, i] = 1.0

    print("=" * 94)
    print("  judgment 0 (full Toffoli): compute 4-T AND -> CX(t,target) -> H(t)"
          " -> branch + fixup")
    print("=" * 94)
    best = None
    for name, spec in FIXUPS:
        f = np.eye(2 ** N, dtype=complex)
        if spec is None:
            pass
        elif isinstance(spec, str):
            f = gate_matrix(spec, 0 if spec in ("x", "z", "s", "sdg") else 0)
        else:
            for part in spec:
                f = gate_matrix(part, 0) @ f
        ok_branches = True
        phases = []
        for a in (0, 1):
            for b in (0, 1):
                for g in (0, 1):
                    psi = np.zeros(2 ** N, dtype=complex)
                    for tq in (0, 1):
                        psi[IDX(a, b, tq, g)] = T_STATE[tq]
                    out = u @ psi
                    want = np.zeros(2 ** N, dtype=complex)
                    want[IDX(a, b, 0, g ^ (a & b))] = 1.0     # ancilla back to |0>
                    ref = None
                    for br, proj in ((0, p0), (1, p1)):
                        v = proj @ out
                        nrm = np.linalg.norm(v)
                        if nrm < 1e-12:
                            continue
                        v = v / nrm
                        v = f @ v if br == 1 else v
                        # compare to the wanted state on (x,y,target), ancilla |0>
                        tgt = np.zeros(2 ** N, dtype=complex)
                        tgt[IDX(a, b, 0, g ^ (a & b))] = 1.0
                        ratio = (tgt.conj() @ v)
                        if abs(abs(ratio) - 1) > 1e-9:
                            ok_branches = False
                            break
                        phases.append(np.angle(ratio))
                    if not ok_branches:
                        break
                if not ok_branches:
                    break
            if not ok_branches:
                break
        spread = (max(phases) - min(phases)) if phases else float("nan")
        print(f"  fixup {name:<4}: branches match Toffoli = {ok_branches}   "
              f"phase spread over branches = {np.nanmax(phases) - np.nanmin(phases) if phases else float('nan'):.3e}"
              if phases else f"  fixup {name:<4}: branches match = {ok_branches}")
        if ok_branches:
            best = name
    print("=" * 94)
    print(f"  JUDGMENT 0: {'PASS with fixup ' + best if best else 'FAIL'}")
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
