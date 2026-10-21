"""Phase-polynomial extraction for H-free circuits (toy proof-carrying pass).

Gate set: {x, cx, t, tdg, s, sdg, z}.  A circuit over this set is
``U = P . D`` where ``P`` is the affine permutation induced by X/CX and ``D`` is
a diagonal phase polynomial.  We represent ``D`` as a list of
``(coefficient mod 8, parity mask)`` terms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qiskit import QuantumCircuit

_COEFF = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4}


def extract(circuit: QuantumCircuit) -> Dict[str, Any]:
    n = circuit.num_qubits
    mask = [1 << q for q in range(n)]
    const = [0] * n
    terms: List[Dict[str, Any]] = []
    for index, instruction in enumerate(circuit.data):
        name = instruction.operation.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if name == "x":
            const[qubits[0]] ^= 1
        elif name == "cx":
            c, t = qubits
            mask[t] ^= mask[c]
            const[t] ^= const[c]
        elif name in _COEFF:
            terms.append({"gate": index, "qubit": qubits[0],
                          "coeff": _COEFF[name] % 8, "mask": mask[qubits[0]]})
        else:
            raise ValueError(f"gate {name!r} outside the phase-polynomial set")
    return {"affine": [[mask[q], const[q]] for q in range(n)], "terms": terms}


def canonical(terms: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Merge equal masks (mod 8), drop zeros, return a sorted tuple list."""
    merged: Dict[int, int] = {}
    for term in terms:
        merged[term["mask"]] = (merged.get(term["mask"], 0) + term["coeff"]) % 8
    return sorted((mask, coeff) for mask, coeff in merged.items() if coeff)


def t_count(canonical_terms: List[Tuple[int, int]]) -> int:
    """Number of odd (T-like) terms: even coefficients are Clifford phases."""
    return sum(1 for _, coeff in canonical_terms if coeff % 2 == 1)
