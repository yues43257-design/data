"""Acceptance tests for toffoli_recover.py (written before the implementation)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import toffoli_recover as TR                # noqa: E402


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def standard_7t(target: int = 2) -> QuantumCircuit:
    """Qiskit's own 7-T Toffoli decomposition, re-targeted if requested."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    base = qc.decompose()
    if target == 2:
        return base
    # permute so the requested wire is the target
    perm = [0, 1, 2]
    perm[2], perm[target] = perm[target], perm[2]
    out = QuantumCircuit(3)
    for instruction in base.data:
        qubits = [base.find_bit(q).index for q in instruction.qubits]
        out.append(instruction.operation, [perm[q] for q in qubits])
    return out


def terminal_measure_circuit() -> QuantumCircuit:
    qc = standard_7t()
    qc.measure_all()
    return qc


def _equivalent_up_to_phase(a: QuantumCircuit, b: QuantumCircuit,
                            tol: float = 1e-9) -> bool:
    ua = np.asarray(Operator(a).data, dtype=complex)
    ub = np.asarray(Operator(b).data, dtype=complex)
    m = ua @ ub.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diag = np.diag(m)
    return off < tol and float(np.abs(diag - diag[0]).max()) < tol


def test_a_standard_7t_recovers_one_ccx():
    out, report = TR.recover(standard_7t())
    assert report["recovered_blocks"] == 1, report
    ccx = [ins for ins in out.data if ins.operation.name == "ccx"]
    assert len(ccx) == 1
    assert report["unmatched_reason"] is None


def test_a2_target_permutation_is_detected():
    out, report = TR.recover(standard_7t(target=0))
    assert report["recovered_blocks"] == 1, report
    ccx = [ins for ins in out.data if ins.operation.name == "ccx"][0]
    target = [out.find_bit(q).index for q in ccx.qubits][2]
    assert target == 0


def test_b_adder_n4_reports_honestly():
    qasm = (ROOT / "benchmarks" / "qasmbench" / "adder_n4.qasm").read_text()
    out, report = TR.recover(qasm)
    assert report["recovered_blocks"] == 0, report
    assert "no matching pattern" in report["unmatched_reason"]
    assert report["remaining_measurements"] == 0


def test_c_equivalence_after_stripping_terminal_measurements():
    original = terminal_measure_circuit()
    recovered, report = TR.recover(original, strip_terminal_measurements=True)
    assert report["removed_terminal_measurements"] == 3
    stripped, _ = TR.strip_terminal_measurements(original)
    assert _equivalent_up_to_phase(recovered, stripped)


def test_d_mid_circuit_measurement_is_not_removed():
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.measure(0, 0)          # mid-circuit: a gate follows it
    qc.cx(0, 1)
    out, report = TR.strip_terminal_measurements(qc)
    assert report["removed_terminal_measurements"] == 0
    assert report["mid_circuit_measurements"] == 1
    assert any(ins.operation.name == "measure" for ins in out.data)


def test_e_relative_phase_block_is_not_recovered():
    """A relative-phase Toffoli is NOT exactly a Toffoli -- must not match."""
    vch = _load("tr_vch", "variant_choice_headroom.py")
    relative = vch.relative_block(2)
    out, report = TR.recover(relative)
    assert report["recovered_blocks"] == 0, report
    assert not any(ins.operation.name == "ccx" for ins in out.data)
