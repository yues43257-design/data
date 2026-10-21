"""Independent checker for the v2 proof log.

Written separately from ``rewrite.py`` on purpose: it re-derives each
precondition from the circuit and re-applies each transformation, then compares
the resulting gate list with the claimed output circuit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qiskit import QuantumCircuit

from . import phase_poly

_DIAGONAL = frozenset({"t", "tdg", "s", "sdg", "z"})


def _ops(circuit: QuantumCircuit) -> List[Tuple[str, List[int]]]:
    return [(ins.operation.name,
             [circuit.find_bit(q).index for q in ins.qubits])
            for ins in circuit.data]


def _build(num_qubits: int, ops) -> QuantumCircuit:
    out = QuantumCircuit(num_qubits)
    for name, qubits in ops:
        if name == "cx":
            out.cx(qubits[0], qubits[1])
        else:
            getattr(out, name)(qubits[0])
    return out


def check(circuit_in: QuantumCircuit, circuit_out: QuantumCircuit,
          proof: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    current = circuit_in.copy()
    for step in proof.get("steps", []):
        rule = step.get("rule")
        ops = _ops(current)
        if rule == "cancel_T_Tdg":
            i, j = step["gates"]
            info = phase_poly.extract(current)
            by_gate = {term["gate"]: term for term in info["terms"]}
            a, b = by_gate.get(i), by_gate.get(j)
            if a is None or b is None:
                reasons.append(f"{i}/{j}: not T-like gates")
                continue
            if {a["coeff"], b["coeff"]} != {1, 7}:
                reasons.append(f"{i}/{j}: not T/Tdg")
                continue
            if a["mask"] != b["mask"] or a.get("const", 0) != b.get("const", 0):
                reasons.append(f"{i}/{j}: different parity functions")
                continue
            current = _build(current.num_qubits,
                             [op for k, op in enumerate(ops) if k not in (i, j)])
        elif rule in ("merge_T_T_to_S", "merge_Tdg_Tdg_to_Sdg"):
            i, j = step["gates"]
            expected = 1 if rule == "merge_T_T_to_S" else 7
            info = phase_poly.extract(current)
            by_gate = {term["gate"]: term for term in info["terms"]}
            a, b = by_gate.get(i), by_gate.get(j)
            if a is None or b is None or a["coeff"] != expected or b["coeff"] != expected:
                reasons.append(f"{i}/{j}: coefficients do not match {rule}")
                continue
            q = a["qubit"]
            if b["qubit"] != q or a["mask"] != (1 << q) or b["mask"] != (1 << q):
                reasons.append(f"{i}/{j}: not a single-qubit parity on one qubit")
                continue
            if a.get("const", 0) or b.get("const", 0):
                reasons.append(f"{i}/{j}: parity is negated (const=1)")
                continue
            if any(q in qubits for _, qubits in ops[i + 1:j]):
                reasons.append(f"{i}/{j}: a gate touches qubit {q} in between")
                continue
            new_ops = list(ops)
            new_ops[j] = ("s" if expected == 1 else "sdg", [q])
            new_ops.pop(i)
            current = _build(current.num_qubits, new_ops)
        elif rule == "commute_diag_cx":
            i = step["gates"][0]
            if i + 1 >= len(ops):
                reasons.append(f"{i}: no following gate")
                continue
            (n1, q1), (n2, q2) = ops[i], ops[i + 1]
            if n1 not in _DIAGONAL or n2 != "cx" or q2[0] != q1[0]:
                reasons.append(f"{i}: not a diagonal-on-control commutation")
                continue
            new_ops = list(ops)
            new_ops[i], new_ops[i + 1] = new_ops[i + 1], new_ops[i]
            current = _build(current.num_qubits, new_ops)
        else:
            reasons.append(f"unknown rule {rule!r}")

    if _ops(current) != _ops(circuit_out):
        reasons.append("replayed circuit differs from C2")
    info_in = phase_poly.extract(circuit_in)
    info_out = phase_poly.extract(circuit_out)
    if phase_poly.canonical(info_in["terms"]) != phase_poly.canonical(info_out["terms"]):
        reasons.append("phase polynomials differ")
    if info_in["affine"] != info_out["affine"]:
        reasons.append("affine part changed")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "steps": len(proof.get("steps", [])),
        "t_before": sum(1 for t in info_in["terms"] if t["coeff"] % 2 == 1),
        "t_after": sum(1 for t in info_out["terms"] if t["coeff"] % 2 == 1),
    }
