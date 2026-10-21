"""Property-based tests for the toy proof-carrying checker."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.proof_carrying import checker, rewrite  # noqa: E402

GATES = st.sampled_from(["x", "cx", "t", "tdg", "s", "sdg", "z"])


@st.composite
def random_circuit(draw):
    n = draw(st.integers(min_value=3, max_value=6))
    length = draw(st.integers(min_value=1, max_value=25))
    ops = draw(st.lists(st.tuples(GATES, st.integers(0, n - 1),
                                  st.integers(0, n - 1)),
                        min_size=length, max_size=length))
    qc = QuantumCircuit(n)
    for name, a, b in ops:
        if name == "cx":
            if a != b:
                qc.cx(a, b)
        elif name == "x":
            qc.x(a)
        else:
            getattr(qc, name)(a)
    return qc


@settings(max_examples=60, deadline=None)
@given(random_circuit())
def test_every_valid_proof_checks(circuit):
    optimized, proof = rewrite.rewrite(circuit)
    result = checker.check(circuit, optimized, proof)
    assert result["ok"] is True, result["reasons"]


@settings(max_examples=60, deadline=None)
@given(random_circuit())
def test_random_rule_corruption_is_rejected(circuit):
    optimized, proof = rewrite.rewrite(circuit)
    forged = copy.deepcopy(proof)
    forged["steps"].append({"rule": "bogus", "gates": [0, 0]})
    assert checker.check(circuit, optimized, forged)["ok"] is False


@settings(max_examples=60, deadline=None)
@given(random_circuit())
def test_output_tampering_is_rejected(circuit):
    optimized, proof = rewrite.rewrite(circuit)
    tampered = optimized.copy()
    tampered.x(0)
    assert checker.check(circuit, tampered, proof)["ok"] is False


@settings(max_examples=60, deadline=None)
@given(random_circuit())
def test_dropping_a_step_is_rejected(circuit):
    optimized, proof = rewrite.rewrite(circuit)
    if not proof["steps"]:
        return
    forged = copy.deepcopy(proof)
    forged["steps"].pop()
    assert checker.check(circuit, optimized, forged)["ok"] is False


def test_t_t_rule_fires():
    circuit = QuantumCircuit(3)
    circuit.t(0)
    circuit.t(0)
    optimized, proof = rewrite.rewrite(circuit)
    assert any(step["rule"] == "merge_T_T_to_S" for step in proof["steps"])
    assert checker.check(circuit, optimized, proof)["ok"] is True
    assert any(ins.operation.name == "s" for ins in optimized.data)


def test_commutation_rule_fires():
    circuit = QuantumCircuit(3)
    circuit.t(0)
    circuit.cx(0, 1)
    optimized, proof = rewrite.rewrite(circuit)
    assert any(step["rule"] == "commute_diag_cx" for step in proof["steps"])
    assert checker.check(circuit, optimized, proof)["ok"] is True
