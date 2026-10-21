"""Local, proof-logging rewrites on H-free phase-polynomial circuits.

Rules (all exact, all logged):

  cancel_T_Tdg        T . Tdg with equal parity mask            -> delete both
  merge_T_T_to_S      T . T on one qubit, parity == that qubit  -> S
  merge_Tdg_Tdg_to_Sdg same with Tdg                             -> Sdg
  commute_diag_cx     diagonal gate on the control of a CX       -> swap

Every step is a local precondition plus a deterministic transformation of the
gate list; the checker re-validates the precondition and re-applies the
transformation **independently**.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import SGate, SdgGate

from . import phase_poly

PROOF_VERSION = "toy-proof-log-v2"
_DIAGONAL = frozenset({"t", "tdg", "s", "sdg", "z"})


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


def _ops(circuit: QuantumCircuit) -> List[Tuple[str, List[int]]]:
    return [(ins.operation.name,
             [circuit.find_bit(q).index for q in ins.qubits])
            for ins in circuit.data]


def _build(num_qubits: int, ops: Sequence[Tuple[str, List[int]]]) -> QuantumCircuit:
    from qiskit.circuit.library import CXGate, TGate, TdgGate, ZGate
    table = {"cx": CXGate(), "t": TGate(), "tdg": TdgGate(),
             "s": SGate(), "sdg": SdgGate(), "z": ZGate()}
    out = QuantumCircuit(num_qubits)
    for name, qubits in ops:
        if name == "cx":
            out.cx(qubits[0], qubits[1])
        elif name == "x":
            out.x(qubits[0])
        else:
            out.append(table[name], qubits[:1])
    return out


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _cancel_step(circuit: QuantumCircuit) -> Optional[Dict[str, Any]]:
    info = phase_poly.extract(circuit)
    plus: Dict[Tuple[int, int], List[int]] = {}
    minus: Dict[Tuple[int, int], List[int]] = {}
    for term in info["terms"]:
        key = (term["mask"], int(term.get("const", 0)))
        if term["coeff"] == 1:
            plus.setdefault(key, []).append(term["gate"])
        elif term["coeff"] == 7:
            minus.setdefault(key, []).append(term["gate"])
    for key, first in plus.items():
        if minus.get(key):
            return {"rule": "cancel_T_Tdg", "gates": [first[0], minus[key][0]]}
    return None


def _merge_step(circuit: QuantumCircuit) -> Optional[Dict[str, Any]]:
    info = phase_poly.extract(circuit)
    for coeff, rule, name in ((1, "merge_T_T_to_S", "t"),
                              (7, "merge_Tdg_Tdg_to_Sdg", "tdg")):
        terms = [t for t in info["terms"] if t["coeff"] == coeff]
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                if a["qubit"] != b["qubit"]:
                    continue
                q = a["qubit"]
                if a["mask"] != (1 << q) or b["mask"] != (1 << q):
                    continue
                if a.get("const", 0) or b.get("const", 0):
                    continue
                lo, hi = sorted((a["gate"], b["gate"]))
                between = _ops(circuit)[lo + 1:hi]
                if any(q in qubits for _, qubits in between):
                    continue
                return {"rule": rule, "gates": [lo, hi], "qubit": q,
                        "replacement": "s" if coeff == 1 else "sdg"}
    return None


def _commute_step(circuit: QuantumCircuit) -> Optional[Dict[str, Any]]:
    ops = _ops(circuit)
    for i in range(len(ops) - 1):
        first, second = ops[i], ops[i + 1]
        if first[0] in _DIAGONAL and second[0] == "cx" and second[1][0] == first[1][0]:
            return {"rule": "commute_diag_cx", "gates": [i, i + 1],
                    "qubit": first[1][0]}
    return None


def _apply(circuit: QuantumCircuit, step: Dict[str, Any]) -> QuantumCircuit:
    ops = list(_ops(circuit))
    rule = step["rule"]
    if rule == "cancel_T_Tdg":
        drop = set(step["gates"])
        ops = [op for index, op in enumerate(ops) if index not in drop]
    elif rule in ("merge_T_T_to_S", "merge_Tdg_Tdg_to_Sdg"):
        lo, hi = step["gates"]
        replacement = step["replacement"]
        ops[hi] = (replacement, ops[hi][1][:1])
        ops.pop(lo)
    elif rule == "commute_diag_cx":
        i = step["gates"][0]
        ops[i], ops[i + 1] = ops[i + 1], ops[i]
    else:
        raise ValueError(rule)
    return _build(circuit.num_qubits, ops)


def rewrite(circuit: QuantumCircuit, max_steps: int = 200
            ) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    current = circuit.copy()
    steps: List[Dict[str, Any]] = []
    for _ in range(max_steps):
        step = (_cancel_step(current) or _merge_step(current)
                or _commute_step(current))
        if step is None:
            break
        current = _apply(current, step)
        steps.append(step)

    info = phase_poly.extract(circuit)
    out_info = phase_poly.extract(current)
    proof = {
        "version": PROOF_VERSION,
        "input_hash": _hash(info["affine"]),
        "steps": steps,
        "input_terms": phase_poly.canonical(info["terms"]),
        "output_terms": phase_poly.canonical(out_info["terms"]),
        "input_affine": info["affine"],
        "output_affine": out_info["affine"],
        "t_before": sum(1 for t in info["terms"] if t["coeff"] % 2 == 1),
        "t_after": sum(1 for t in out_info["terms"] if t["coeff"] % 2 == 1),
    }
    return current, proof
