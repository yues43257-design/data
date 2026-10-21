"""Local rewrite: cancel T / Tdagger pairs with the same parity mask.

Each cancellation is exact (the two diagonal terms sum to 0) and the affine
part is untouched, so ``C1 == C2`` as unitaries.  Every step is recorded in a
proof log: ``(rule, gate indices, parity mask)``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from qiskit import QuantumCircuit

from . import phase_poly

PROOF_VERSION = "toy-proof-log-v1"


def build_instance(n: int = 5, pairs: int = 2) -> QuantumCircuit:
    """H-free circuit with ``pairs`` guaranteed T/Tdagger cancellations."""
    if n < 3:
        raise ValueError("need at least 3 qubits")
    circuit = QuantumCircuit(n)
    target = n - 1
    controls = list(range(n - 1))
    for k in range(pairs):
        a = controls[k % len(controls)]
        b = controls[(k + 1) % len(controls)]
        if b == a:
            b = controls[(k + 2) % len(controls)]
        circuit.cx(a, target)
        circuit.cx(b, target)
        circuit.t(target)
        circuit.cx(b, target)
        circuit.cx(a, target)
        circuit.cx(a, target)
        circuit.cx(b, target)
        circuit.tdg(target)
        circuit.cx(b, target)
        circuit.cx(a, target)
    return circuit


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def rewrite(circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    info = phase_poly.extract(circuit)
    plus: Dict[int, List[int]] = {}
    minus: Dict[int, List[int]] = {}
    for term in info["terms"]:
        bucket = plus if term["coeff"] == 1 else (minus if term["coeff"] == 7 else None)
        if bucket is not None:
            bucket.setdefault(term["mask"], []).append(term["gate"])

    steps: List[Dict[str, Any]] = []
    removed = set()
    for mask, gates in plus.items():
        partners = minus.get(mask, [])
        for first, second in zip(gates, partners):
            steps.append({"rule": "cancel_T_Tdg", "gates": [first, second],
                          "mask": mask})
            removed.update((first, second))

    out = QuantumCircuit(circuit.num_qubits)
    for index, instruction in enumerate(circuit.data):
        if index in removed:
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        out.append(instruction.operation, qubits,
                   [circuit.find_bit(c).index for c in instruction.clbits])

    before = phase_poly.canonical(info["terms"])
    out_info = phase_poly.extract(out)
    after = phase_poly.canonical(out_info["terms"])
    t_before = sum(1 for term in info["terms"] if term["coeff"] % 2 == 1)
    t_after = sum(1 for term in out_info["terms"] if term["coeff"] % 2 == 1)
    proof = {
        "version": PROOF_VERSION,
        "input_hash": _hash(phase_poly.extract(circuit)["affine"]),
        "steps": steps,
        "input_terms": before,
        "output_terms": after,
        "input_affine": info["affine"],
        "output_affine": phase_poly.extract(out)["affine"],
        "t_before": t_before,
        "t_after": t_after,
    }
    return out, proof
