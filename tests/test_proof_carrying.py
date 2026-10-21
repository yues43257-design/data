"""Tests for the toy proof-carrying pass."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.proof_carrying import checker, phase_poly, rewrite  # noqa: E402


def test_rewrite_cancels_pairs():
    circuit = rewrite.build_instance(5, 2)
    optimized, proof = rewrite.rewrite(circuit)
    assert len(proof["steps"]) == 2
    assert proof["t_before"] == 4
    assert proof["t_after"] == 0


def test_checker_accepts_valid_proof():
    circuit = rewrite.build_instance(5, 2)
    optimized, proof = rewrite.rewrite(circuit)
    result = checker.check(circuit, optimized, proof)
    assert result["ok"] is True, result


def test_checker_rejects_tampered_proof():
    circuit = rewrite.build_instance(5, 2)
    optimized, proof = rewrite.rewrite(circuit)
    bad = copy.deepcopy(proof)
    bad["steps"][0]["mask"] ^= 1
    result = checker.check(circuit, optimized, bad)
    assert result["ok"] is False


def test_checker_rejects_modified_output_circuit():
    circuit = rewrite.build_instance(5, 2)
    optimized, proof = rewrite.rewrite(circuit)
    modified = optimized.copy()
    modified.x(0)
    result = checker.check(circuit, modified, proof)
    assert result["ok"] is False


def test_rewrite_is_exactly_equivalent():
    circuit = rewrite.build_instance(6, 3)
    optimized, _ = rewrite.rewrite(circuit)
    a = np.asarray(Operator(circuit).data, dtype=complex)
    b = np.asarray(Operator(optimized).data, dtype=complex)
    m = a @ b.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diagonal = np.diag(m)
    assert off < 1e-9
    assert float(np.abs(diagonal - diagonal[0]).max()) < 1e-9


def test_canonical_merges_coefficients():
    circuit = rewrite.build_instance(4, 1)
    info = phase_poly.extract(circuit)
    # the T/Tdg pair cancels in the canonical form: 0 odd terms
    assert phase_poly.t_count(phase_poly.canonical(info["terms"])) == 0
