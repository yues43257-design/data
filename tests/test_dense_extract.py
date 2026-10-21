"""Cross-validation of the gate-level extractor against the dense unitary."""

from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.proof_carrying import dense_extract, rewrite  # noqa: E402

GATES = st.sampled_from(["x", "cx", "t", "tdg", "s", "sdg", "z"])


@st.composite
def random_circuit(draw):
    n = draw(st.integers(min_value=3, max_value=6))
    length = draw(st.integers(min_value=1, max_value=20))
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


@settings(max_examples=50, deadline=None)
@given(random_circuit())
def test_gate_extractor_matches_dense(circuit):
    result = dense_extract.cross_validate(circuit)
    assert result["ok"] is True, result


def test_rewrite_keeps_dense_phase_function():
    circuit = rewrite.build_instance(6, 3)
    optimized, _ = rewrite.rewrite(circuit)
    assert dense_extract.cross_validate(circuit)["ok"] is True
    assert dense_extract.cross_validate(optimized)["ok"] is True


def test_dense_extractor_is_not_trivial():
    t_circuit = QuantumCircuit(2)
    t_circuit.t(0)
    tdg_circuit = QuantumCircuit(2)
    tdg_circuit.tdg(0)
    _, phases_t = dense_extract.dense_permutation_and_phases(t_circuit)
    _, phases_tdg = dense_extract.dense_permutation_and_phases(tdg_circuit)
    assert phases_t != phases_tdg
