"""Attack constructions for the certificate-binding probe (bound API).

Threat model: the adversary controls the compiler pass.  It emits a circuit that
differs from the certified one, while keeping the certificate.  The fix under
test binds the certificate to the emitted circuit via

    circuit_hash, schedule_hash, block_hashes

Nothing here modifies ``relative_phase_criterion.py``; the only change outside
``experiments/security_probe/`` is the additive, backward-compatible binding in
``support_tracker.certify`` / ``verify_certificate``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402

ATTACKS = ("normal", "block_substitution", "nonblock_x", "tail_h",
           "gate_smuggle_u3", "gate_smuggle_measure", "cert_forge", "cert_replay")

_BLOCKS = ((0, 1, 2), (3, 4, 2))       # controls and target of each block


def _rccx(circuit: QuantumCircuit, c0: int, c1: int, target: int) -> None:
    for gate in RCW.LM._rccx_gates(c0, c1, target):
        if gate[0] == "cx":
            circuit.cx(gate[1], gate[2])
        else:
            getattr(circuit, gate[0])(gate[1])


def _emit(blocks=_BLOCKS, between: Optional[Any] = None,
          tail: Optional[Any] = None) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    """Build the emitted (relative-phase) circuit and per-block gate ranges."""
    circuit = QuantumCircuit(5)
    ranges: List[Tuple[int, int]] = []
    for index, (c0, c1, target) in enumerate(blocks):
        start = len(circuit.data)
        _rccx(circuit, c0, c1, target)
        ranges.append((start, len(circuit.data)))
        if index == 0 and between is not None:
            between(circuit)
    if tail is not None:
        tail(circuit)
    return circuit, ranges


def build_base() -> Dict[str, Any]:
    ideal = QuantumCircuit(5)
    ideal.ccx(0, 1, 2)
    ideal.cx(0, 3)
    ideal.ccx(3, 4, 2)
    emitted, ranges = _emit(between=lambda qc: qc.cx(0, 3))
    schedule = RB._tracker_schedule(ideal)
    mask = (1 << len(schedule["blocks"])) - 1
    certificate = ST.certify(schedule, mask, circuit=emitted, block_ranges=ranges)
    return {"ideal": ideal, "circuit": emitted, "ranges": ranges,
            "schedule": schedule, "mask": mask, "certificate": certificate}


def attack(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
    entry = {"attack": name, "circuit": base["circuit"].copy(),
             "ranges": base["ranges"], "schedule": base["schedule"],
             "mask": base["mask"], "certificate": base["certificate"],
             "note": ""}
    if name == "normal":
        entry["note"] = "untampered baseline"
    elif name == "block_substitution":
        circuit, ranges = _emit(blocks=((0, 2, 1), (3, 4, 2)),
                                between=lambda qc: qc.cx(0, 3))
        entry.update(circuit=circuit, ranges=ranges,
                     note="block 0 re-mapped [0,1,2] -> [0,2,1]")
    elif name == "nonblock_x":
        circuit, ranges = _emit(
            between=lambda qc: (qc.x(3), qc.cx(0, 3)))
        entry.update(circuit=circuit, ranges=ranges,
                     note="X on a data qubit between the blocks")
    elif name == "tail_h":
        circuit, ranges = _emit(between=lambda qc: qc.cx(0, 3),
                                tail=lambda qc: qc.h(2))
        entry.update(circuit=circuit, ranges=ranges,
                     note="H after the last block")
    elif name == "gate_smuggle_u3":
        circuit, ranges = _emit(between=lambda qc: (qc.u(0.1, 0.2, 0.3, 3),
                                                    qc.cx(0, 3)))
        entry.update(circuit=circuit, ranges=ranges,
                     note="u3 smuggled into the emitted circuit")
    elif name == "gate_smuggle_measure":
        circuit = QuantumCircuit(5, 1)
        circuit.compose(base["circuit"], inplace=True)
        circuit.measure(3, 0)
        entry.update(circuit=circuit,
                     note="measure smuggled into the emitted circuit")
    elif name == "cert_forge":
        forged = dict(base["certificate"])
        forged["accepted"] = not base["certificate"]["accepted"]
        forged["final_support"] = sorted(set(
            base["certificate"]["final_support"] + [999]))
        entry["certificate"] = forged
        entry["note"] = "accepted flipped and final_support padded"
    elif name == "cert_replay":
        entry["certificate"] = ST.certify(
            base["schedule"], 1, circuit=base["circuit"],
            block_ranges=base["ranges"])
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
    ua = np.asarray(Operator(a).data, dtype=complex)
    ub = np.asarray(Operator(b).data, dtype=complex)
    m = ua @ ub.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    diagonal = np.diag(m)
    return bool(off < tol and float(np.abs(diagonal - diagonal[0]).max()) < tol)


def evaluate(base: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    attack_legal = _global_phase_equal(entry["circuit"], base["circuit"])
    error = None
    started = time.perf_counter()
    if entry["attack"].startswith("gate_smuggle"):
        try:
            schedule = RB._tracker_schedule(entry["circuit"])
            unbound = bool(ST.certify(schedule, entry["mask"])["accepted"])
            bound = bool(ST.certify(schedule, entry["mask"],
                                    circuit=entry["circuit"])["accepted"])
        except ST.UnsupportedGateError as exc:
            unbound = bound = False
            error = type(exc).__name__
    else:
        # Old (schedule-only) behaviour: compare the core fields against a fresh
        # unbound certificate.  Hashes/schema are deliberately ignored here, so
        # this reproduces what the pre-fix verifier would have accepted.
        fresh_unbound = ST.certify(entry["schedule"], entry["mask"])
        core = ("mask", "accepted", "final_support", "h_checks", "violations")
        unbound = all(entry["certificate"].get(k) == fresh_unbound.get(k)
                      for k in core)
        bound = bool(ST.verify_certificate(
            entry["schedule"], entry["mask"], entry["certificate"],
            circuit=entry["circuit"], block_ranges=entry["ranges"]))
    verify_seconds = time.perf_counter() - started
    detected = not bound
    return {
        "attack": entry["attack"],
        "note": entry["note"],
        "attack_legal": attack_legal,
        "unbound_accepts": unbound,
        "certificate_accepts": bound,
        "detected": detected,
        "attack_success": int(bound and attack_legal is False),
        "certificate_bytes": len(json.dumps(entry["certificate"])),
        "verify_seconds": round(verify_seconds, 6),
        "error": error,
    }
