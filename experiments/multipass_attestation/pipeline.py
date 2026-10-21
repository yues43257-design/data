"""Minimal two-pass pipeline with a certificate chain.

    C0 --pass1 (relative-phase rewrite)--> C1 --pass2 (exact full_reduce)--> C2

pass1 emits a **bound** `diagonal-certificate-v1` for C1 (see
`support_tracker.certify`), extended with `input_hash`/`output_hash`.
pass2 emits an exactness certificate for C2 with `input_hash`/`output_hash`.

Nothing here modifies the main flow; this directory is self-contained.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402

PIPELINE_VERSION = "multipass-v1"

_VARIANTS = {
    0: ((0, 1, 2), (3, 4, 2)),
    1: ((1, 2, 3), (0, 4, 3)),
}


def _rccx(circuit: QuantumCircuit, c0: int, c1: int, target: int) -> None:
    for gate in RCW.LM._rccx_gates(c0, c1, target):
        if gate[0] == "cx":
            circuit.cx(gate[1], gate[2])
        else:
            getattr(circuit, gate[0])(gate[1])


def _make_c0(variant: int) -> QuantumCircuit:
    circuit = QuantumCircuit(5)
    blocks = _VARIANTS[variant]
    circuit.ccx(*blocks[0])
    circuit.cx(0, 3)
    circuit.ccx(*blocks[1])
    return circuit


def _emit_c1(variant: int) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    circuit = QuantumCircuit(5)
    ranges: List[Tuple[int, int]] = []
    for index, (c0, c1, target) in enumerate(_VARIANTS[variant]):
        start = len(circuit.data)
        _rccx(circuit, c0, c1, target)
        ranges.append((start, len(circuit.data)))
        if index == 0:
            circuit.cx(0, 3)
    return circuit, ranges


def full_reduce_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """pyZX full_reduce + extract, back to a qiskit circuit (fallback: copy)."""
    try:
        zx_circ = RB.LM.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()
        graph = zx_circ.to_graph()
        RB.LM.zx.simplify.full_reduce(graph)
        extracted = RB.LM.zx.extract_circuit(graph)
        return qasm2.loads(extracted.to_qasm())
    except Exception:                       # pragma: no cover - pyZX refusal
        return circuit.copy()


def build_pipeline(variant: int = 0) -> Dict[str, Any]:
    started = time.perf_counter()
    c0 = _make_c0(variant)
    t0 = time.perf_counter()
    c1, ranges = _emit_c1(variant)
    pass1_seconds = time.perf_counter() - t0

    schedule = RB._tracker_schedule(c0)
    mask = (1 << len(schedule["blocks"])) - 1
    t0 = time.perf_counter()
    cert1 = ST.certify(schedule, mask, circuit=c1, block_ranges=ranges)
    cert1["input_hash"] = ST.circuit_hash(c0)
    cert1["output_hash"] = ST.circuit_hash(c1)
    pass1_cert_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    c2 = full_reduce_circuit(c1)
    pass2_seconds = time.perf_counter() - t0
    cert2 = {
        "pass": "pyzx_full_reduce",
        "claim": "exact_equivalent",
        "input_hash": ST.circuit_hash(c1),
        "output_hash": ST.circuit_hash(c2),
    }
    return {
        "version": PIPELINE_VERSION,
        "variant": variant,
        "c0": c0, "c1": c1, "c2": c2,
        "schedule": schedule, "mask": mask, "ranges": ranges,
        "cert1": cert1, "cert2": cert2,
        "times": {"pass1": round(pass1_seconds, 5),
                  "pass1_cert": round(pass1_cert_seconds, 5),
                  "pass2": round(pass2_seconds, 5),
                  "total": round(time.perf_counter() - started, 5)},
    }
