"""Independent checker for the toy proof log.

It re-extracts the phase polynomial from C1, replays every ``cancel_T_Tdg``
step against the actual gate list, derives the expected output polynomial, and
compares it with the polynomial extracted from C2.  It never trusts the proof's
``input_terms``/``output_terms`` fields -- those are only checked for
consistency.
"""

from __future__ import annotations

from typing import Any, Dict, List

from qiskit import QuantumCircuit

from . import phase_poly


def _lists(terms) -> List[List[int]]:
    return [list(term) for term in terms]


def check(circuit_in: QuantumCircuit, circuit_out: QuantumCircuit,
          proof: Dict[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    info = phase_poly.extract(circuit_in)
    by_gate = {term["gate"]: term for term in info["terms"]}
    removed = set()
    for step in proof.get("steps", []):
        if step.get("rule") != "cancel_T_Tdg":
            reasons.append(f"unknown rule {step.get('rule')!r}")
            continue
        first, second = step["gates"]
        a, b = by_gate.get(first), by_gate.get(second)
        if a is None or b is None:
            reasons.append(f"gate indices {first}/{second} are not T-like gates")
            continue
        if {a["coeff"], b["coeff"]} != {1, 7}:
            reasons.append(f"pair {first}/{second} is not T/Tdg")
            continue
        if a["mask"] != b["mask"]:
            reasons.append(f"pair {first}/{second} has different parity masks")
            continue
        if step.get("mask") != a["mask"]:
            reasons.append(f"pair {first}/{second} mask does not match the log")
            continue
        removed.update((first, second))

    expected = phase_poly.canonical(
        [term for term in info["terms"] if term["gate"] not in removed])
    actual = phase_poly.canonical(phase_poly.extract(circuit_out)["terms"])
    if _lists(expected) != _lists(actual):
        reasons.append("output polynomial does not match the replayed rewrite")
    if info["affine"] != phase_poly.extract(circuit_out)["affine"]:
        reasons.append("affine (permutation) part changed")
    if _lists(proof.get("input_terms", [])) != _lists(
            phase_poly.canonical(info["terms"])):
        reasons.append("proof.input_terms disagrees with C1")
    out_info = phase_poly.extract(circuit_out)
    return {
        "ok": not reasons,
        "reasons": reasons,
        "steps": len(proof.get("steps", [])),
        "t_before": sum(1 for term in info["terms"] if term["coeff"] % 2 == 1),
        "t_after": sum(1 for term in out_info["terms"] if term["coeff"] % 2 == 1),
    }
