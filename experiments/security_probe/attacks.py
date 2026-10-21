"""Attack constructions for the certificate-binding probe.

The threat model: the adversary **controls the compiler pass**.  It can emit a
circuit that differs from the one the certificate was computed for, while
keeping the schedule/certificate unchanged (or replaying another one).

Nothing here modifies ``support_tracker.py`` / ``relative_phase_criterion.py``;
the probe only *uses* them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import relative_phase_criterion as RPC  # noqa: E402
import support_tracker as ST          # noqa: E402

ATTACKS = ("normal", "block_substitution", "nonblock_x", "tail_h",
           "gate_smuggle_u3", "gate_smuggle_measure", "cert_forge", "cert_replay")


def _rccx(circuit: QuantumCircuit, c0: int, c1: int, target: int) -> None:
    for gate in RCW.LM._rccx_gates(c0, c1, target):
        if gate[0] == "cx":
            circuit.cx(gate[1], gate[2])
        else:
            getattr(circuit, gate[0])(gate[1])


def build_base() -> Dict[str, Any]:
    """Two downgraded blocks, one CNOT between them, N = 5 (no H in layers)."""
    circuit = QuantumCircuit(5)
    _rccx(circuit, 0, 1, 2)          # block 0: controls (0,1), target 2
    circuit.cx(0, 3)
    _rccx(circuit, 3, 4, 2)          # block 1: controls (3,4), target 2
    schedule = {
        "total": 5,
        "blocks": [
            {"controls": [0, 1], "ancillas": [], "target": 2},
            {"controls": [3, 4], "ancillas": [], "target": 2},
        ],
        "layers": [[], [("cx", [0, 3])], []],
    }
    mask = (1 << len(schedule["blocks"])) - 1
    certificate = ST.certify(schedule, mask)
    return {"circuit": circuit, "schedule": schedule, "mask": mask,
            "certificate": certificate}


def attack(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Return an entry: the emitted circuit plus the (unchanged) certificate."""
    entry = {"attack": name, "circuit": base["circuit"].copy(),
             "schedule": base["schedule"], "mask": base["mask"],
             "certificate": base["certificate"], "note": ""}
    if name == "normal":
        entry["note"] = "untampered baseline"
    elif name == "block_substitution":
        circuit = QuantumCircuit(5)
        _rccx(circuit, 0, 2, 1)      # swap control-1 and target of block 0
        circuit.cx(0, 3)
        _rccx(circuit, 3, 4, 2)
        entry["circuit"] = circuit
        entry["note"] = "block 0 re-mapped [0,1,2] -> [0,2,1]; schedule untouched"
    elif name == "nonblock_x":
        circuit = QuantumCircuit(5)
        _rccx(circuit, 0, 1, 2)
        circuit.x(3)                 # non-block gate, not in the schedule
        circuit.cx(0, 3)
        _rccx(circuit, 3, 4, 2)
        entry["circuit"] = circuit
        entry["note"] = "X on a data qubit, schedule untouched"
    elif name == "tail_h":
        circuit = base["circuit"].copy()
        circuit.h(2)                 # after the last block, not in the schedule
        entry["circuit"] = circuit
        entry["note"] = "H after the last block, schedule untouched"
    elif name == "gate_smuggle_u3":
        circuit = base["circuit"].copy()
        circuit.u(0.1, 0.2, 0.3, 3)
        entry["circuit"] = circuit
        entry["note"] = "u3 smuggled into the emitted circuit"
    elif name == "gate_smuggle_measure":
        circuit = QuantumCircuit(5, 1)
        circuit.compose(base["circuit"], inplace=True)
        circuit.measure(3, 0)
        entry["circuit"] = circuit
        entry["note"] = "measure smuggled into the emitted circuit"
    elif name == "cert_forge":
        forged = dict(base["certificate"])
        # NOTE: on this H-free base the genuine h_checks is already empty, so
        # clearing it would be a no-op.  Forge the verdict and the support.
        forged["accepted"] = not base["certificate"]["accepted"]
        forged["final_support"] = sorted(set(
            base["certificate"]["final_support"] + [999]))
        entry["certificate"] = forged
        entry["note"] = "accepted flipped and final_support padded"
    elif name == "cert_replay":
        entry["certificate"] = ST.certify(base["schedule"], 1)
        entry["note"] = "certificate from mask=1 replayed for mask=3"
    else:
        raise ValueError(name)
    return entry


def _global_phase_equal(a: QuantumCircuit, b: QuantumCircuit,
                        tol: float = 1e-9):
    if a.num_clbits or b.num_clbits:
        return None
    names = {ins.operation.name for ins in a.data} | {
        ins.operation.name for ins in b.data}
    if names & {"measure", "reset", "if_else"}:
        return None
    if a.num_qubits > 11 or b.num_qubits > 11:
        return None
    ua = np.asarray(Operator(a).data, dtype=complex)
    ub = np.asarray(Operator(b).data, dtype=complex)
    m = ua @ ub.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diagonal = np.diag(m)
    return bool(off < tol and float(np.abs(diagonal - diagonal[0]).max()) < tol)


def evaluate(base: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    attack_legal = _global_phase_equal(entry["circuit"], base["circuit"])
    started = time.perf_counter()
    error = None
    if entry["attack"].startswith("gate_smuggle"):
        # the honest checker rebuilds the schedule from the emitted circuit
        try:
            schedule = RB._tracker_schedule(entry["circuit"])
            certificate_accepts = bool(ST.certify(schedule, entry["mask"])["accepted"])
        except ST.UnsupportedGateError as exc:
            certificate_accepts = False
            error = type(exc).__name__
    else:
        certificate_accepts = bool(ST.verify_certificate(
            entry["schedule"], entry["mask"], entry["certificate"]))
    verify_seconds = time.perf_counter() - started
    detected = not certificate_accepts
    attack_success = int(certificate_accepts and attack_legal is False)
    return {
        "attack": entry["attack"],
        "note": entry["note"],
        "attack_legal": attack_legal,
        "certificate_accepts": certificate_accepts,
        "detected": detected,
        "attack_success": attack_success,
        "certificate_bytes": len(json.dumps(entry["certificate"])),
        "verify_seconds": round(verify_seconds, 6),
        "error": error,
    }
