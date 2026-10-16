#!/usr/bin/env python3
"""Judgment 0: does Gidney 2018's Figure-3 AND gadget work?

Figure-3 computation circuit (wires x, y, t; t starts in |T> = T|+>):

    CNOT(x,t) ; CNOT(y,t)
    CNOT(t,x) ; CNOT(t,y)
    Tdag(x)   ; Tdag(y) ; T(t)
    CNOT(t,x) ; CNOT(t,y)
    H(t)            -> then measure t in the Z basis

T-count = 4 (Tdag, Tdag, T plus the |T> catalyst).

Judgment 0 asserts, BEFORE any optimisation claim:
  1. P(t=1) is deterministic for every (x,y) in {0,1}^2 and equals x AND y;
  2. the (x,y) marginal is unchanged (the gadget leaves the data alone);
  3. for a SUPERPOSED (x,y) the (x,y) marginal is still unchanged.
Skipping (1) is exactly what invalidated the earlier probe.

Usage
-----
    .venv/bin/python gidney_and_check.py
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, partial_trace, Statevector

T_STATE = np.array([1.0, np.exp(1j * np.pi / 4)]) / np.sqrt(2)


def and_body() -> QuantumCircuit:
    qc = QuantumCircuit(3)          # q0 = x, q1 = y, q2 = t
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.cx(2, 0)
    qc.cx(2, 1)
    qc.tdg(0)
    qc.tdg(1)
    qc.t(2)
    qc.cx(2, 0)
    qc.cx(2, 1)
    qc.h(2)
    return qc


def main() -> int:
    body = and_body()
    u = Operator(body).data
    t_gates = sum(1 for i in body.data if i.operation.name in ("t", "tdg"))
    print("=" * 92)
    print("  judgment 0: Gidney Fig-3 AND gadget")
    print("=" * 92)
    print(f"  body T gates = {t_gates}  (+1 for the |T> catalyst = {t_gates + 1} per AND)")

    ok = True
    for x in (0, 1):
        for y in (0, 1):
            psi = np.zeros(8, dtype=complex)
            for tq in (0, 1):
                psi[x * 1 + y * 2 + tq * 4] = T_STATE[tq]
            out = u @ psi
            p1 = float(np.sum(np.abs(out[[i for i in range(8) if (i >> 2) & 1]]) ** 2))
            det = min(p1, 1 - p1) < 1e-9
            equals = abs(p1 - (x & y)) < 1e-9
            # (x,y) marginal unchanged?
            rho = np.outer(out, out.conj())
            data = np.zeros((4, 4), dtype=complex)
            for i in range(8):
                for j in range(8):
                    if (i & 3) == (j & 3):
                        data[i & 3, j & 3] += rho[i, j]
            expect = np.zeros((4, 4), dtype=complex)
            expect[x + 2 * y, x + 2 * y] = 1.0
            marg_ok = float(np.abs(data - expect).max()) < 1e-9
            ok &= det and equals and marg_ok
            print(f"  x={x} y={y}: P(t=1)={p1:.6f}  deterministic={det}  "
                  f"==AND:{equals}  (x,y) marginal unchanged={marg_ok}")

    # superposed data check
    rng = np.random.default_rng(0)
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    v /= np.linalg.norm(v)
    psi = np.zeros(8, dtype=complex)
    for i in range(4):
        for tq in (0, 1):
            psi[i + tq * 4] = v[i] * T_STATE[tq]
    out = u @ psi
    rho = np.outer(out, out.conj())
    data = np.zeros((4, 4), dtype=complex)
    for i in range(8):
        for j in range(8):
            if (i & 3) == (j & 3):
                data[i & 3, j & 3] += rho[i, j]
    sup_ok = float(np.abs(data - np.outer(v, v.conj())).max()) < 1e-9
    print(f"  superposed (x,y): data marginal unchanged = {sup_ok}")
    ok &= sup_ok
    print("=" * 92)
    print(f"  JUDGMENT 0: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
