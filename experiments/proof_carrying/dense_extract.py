"""Independent extraction of the phase polynomial from the dense unitary.

``phase_poly.extract`` walks the gate list.  This module instead builds the full
``2**n x 2**n`` unitary, reads the permutation and the diagonal phases directly,
and checks that the gate-level representation reproduces **every** phase on
**every** basis state (up to one global Z_8 offset).

That is the cross-check the proof checker needs: if the two agree on all 2**n
inputs, a shared bug in the gate walk is very unlikely to survive.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import phase_poly

DENSE_LIMIT = 8          # 2**8 x 2**8 is cheap; keeps tests fast


def dense_permutation_and_phases(circuit: QuantumCircuit
                                 ) -> Tuple[List[int], List[int]]:
    matrix = np.asarray(Operator(circuit).data, dtype=complex)
    dim = matrix.shape[0]
    n = dim.bit_length() - 1
    permutation: List[int] = []
    phases: List[int] = []
    for column in range(dim):
        row = int(np.argmax(np.abs(matrix[:, column])))
        permutation.append(row)
        angle = float(np.angle(matrix[row, column]))
        phases.append(int(round(angle / (math.pi / 4))) % 8)
    return permutation, phases


def _apply_affine(affine: List[List[int]], x: int) -> int:
    out = 0
    for qubit, (mask, const) in enumerate(affine):
        value = (bin(mask & x).count("1") & 1) ^ int(const)
        out |= value << qubit
    return out


def _poly_phase(terms: List[Tuple[int, int, int]], x: int) -> int:
    total = 0
    for mask, const, coeff in terms:
        parity = (bin(mask & x).count("1") & 1) ^ int(const)
        if parity:
            total += coeff
    return total % 8


def cross_validate(circuit: QuantumCircuit) -> Dict[str, Any]:
    n = circuit.num_qubits
    if n > DENSE_LIMIT:
        return {"ok": None, "reason": f"N={n} > {DENSE_LIMIT}"}
    permutation, phases = dense_permutation_and_phases(circuit)
    info = phase_poly.extract(circuit)
    affine = info["affine"]
    terms = phase_poly.canonical(info["terms"])

    affine_ok = all(_apply_affine(affine, x) == permutation[x]
                    for x in range(1 << n))
    offsets = [(_poly_phase(terms, x) - phases[x]) % 8 for x in range(1 << n)]
    phase_ok = len(set(offsets)) == 1
    return {
        "ok": bool(affine_ok and phase_ok),
        "affine_ok": affine_ok,
        "phase_ok": phase_ok,
        "global_offset": offsets[0] if offsets else None,
        "inputs_checked": 1 << n,
        "terms": len(terms),
    }
