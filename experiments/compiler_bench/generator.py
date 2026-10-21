"""Deterministic generator for compiler-bug cases.

Axes: family (two-block / CDKM adder / comparator) × width n × layout
(plain / with-H) × certificate version (v1 unbound / v2 bound) × attack × seed.

Every case carries three ground truths:

  L  legality  -- residual is diagonal (relative-phase criterion), dense oracle
  S  semantics -- exactly equal to the intended circuit (global phase), dense
  C  certificate -- the certificate is valid AND says "accepted"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CDKMRippleCarryAdder, IntegerComparator, VBERippleCarryAdder,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402

ATTACKS = ("legal", "h_on_support", "block_substitution", "nonblock_x", "tail_h",
           "gate_smuggle_u3", "gate_smuggle_measure", "cert_forge",
           "cert_replay")


def _rccx(circuit: QuantumCircuit, c0: int, c1: int, target: int) -> None:
    for gate in RCW.LM._rccx_gates(c0, c1, target):
        if gate[0] == "cx":
            circuit.cx(gate[1], gate[2])
        else:
            getattr(circuit, gate[0])(gate[1])


def _block_positions(circuit: QuantumCircuit) -> List[int]:
    return [i for i, ins in enumerate(circuit.data)
            if ins.operation.name == "ccx"]


def _emit(circuit: QuantumCircuit, modes: List[int]) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    """mode 0 = exact ccx, mode 1 = relative-phase block."""
    positions = _block_positions(circuit)
    modes = (modes + [1] * len(positions))[:len(positions)]
    out = QuantumCircuit(circuit.num_qubits)
    ranges: List[Tuple[int, int]] = []
    index = 0
    for position, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if position in positions:
            start = len(out.data)
            if modes[index] == 1:
                _rccx(out, *qubits)
            else:
                out.ccx(*qubits)
            ranges.append((start, len(out.data)))
            index += 1
        else:
            out.append(instruction.operation, qubits,
                       [circuit.find_bit(c).index for c in instruction.clbits])
    return out, ranges


def build_base(family: str, n: int, layout: str,
               seed: int = 0) -> Dict[str, Any]:
    """Return the ideal (ccx atoms), emitted (all-relative) and the schedule."""
    if family == "two_block":
        ideal = QuantumCircuit(5)
        ideal.ccx(0, 1, 2)
        ideal.cx(seed % 3, 3)
        if layout == "with_h":
            ideal.h(2)
        ideal.ccx(3, 4, 2)
    elif family == "hh_restore":
        # H ... H after the block: the correction is conjugated twice and comes
        # back to diagonal, but the tracker rejects at the first H.
        ideal = QuantumCircuit(3)
        ideal.ccx(0, 1, 2)
        ideal.h(2)
        ideal.h(2)
    elif family in ("adder", "vbe"):
        blueprint = (CDKMRippleCarryAdder(n) if family == "adder"
                     else VBERippleCarryAdder(n))
        ideal = RB.to_canonical(blueprint)
        if layout == "with_h":
            ideal = _insert_h(ideal, seed)
    elif family == "comparator":
        ideal = RB.to_canonical(IntegerComparator(n, value=2))
        if layout == "with_h":
            ideal = _insert_h(ideal, seed)
    else:
        raise ValueError(family)
    emitted, ranges = _emit(ideal, [1] * len(_block_positions(ideal)))
    schedule = RB._tracker_schedule(ideal)
    mask = (1 << len(schedule["blocks"])) - 1
    certificate = ST.certify(schedule, mask, circuit=emitted, block_ranges=ranges)
    return {"family": family, "n": n, "layout": layout, "seed": seed,
            "ideal": ideal, "emitted": emitted, "ranges": ranges,
            "schedule": schedule, "mask": mask, "certificate": certificate}


def _insert_h(circuit: QuantumCircuit, seed: int) -> QuantumCircuit:
    """Insert H before block ``seed % len(positions)`` on its target."""
    positions = _block_positions(circuit)
    if not positions:
        return circuit
    block = positions[seed % len(positions)]
    qubit = circuit.find_bit(circuit.data[block].qubits[-1]).index
    out = QuantumCircuit(circuit.num_qubits)
    for position, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if position == block:
            out.h(qubit)
        out.append(instruction.operation, qubits)
    return out


def apply_attack(base: Dict[str, Any], attack: str) -> Dict[str, Any]:
    ideal = base["ideal"]
    entry = {"ideal": ideal, "circuit": base["emitted"],
             "ranges": base["ranges"], "schedule": base["schedule"],
             "mask": base["mask"], "certificate": base["certificate"],
             "attack": attack}
    positions = _block_positions(ideal)
    if attack == "legal":
        return entry
    if attack == "h_on_support":
        emitted = base["emitted"]
        end = base["ranges"][0][1] if base["ranges"] else len(emitted.data)
        target = ideal.find_bit(ideal.data[positions[0]].qubits[-1]).index
        circuit = QuantumCircuit(emitted.num_qubits)
        for index, instruction in enumerate(emitted.data):
            if index == end:
                circuit.h(target)
            qubits = [emitted.find_bit(q).index for q in instruction.qubits]
            circuit.append(instruction.operation, qubits)
        entry["circuit"] = circuit
    elif attack == "block_substitution":
        modes = [1] * len(positions)
        emitted, ranges = _emit_with_remap(ideal, positions)
        entry.update(circuit=emitted, ranges=ranges)
    elif attack == "nonblock_x":
        circuit = base["emitted"].copy()
        circuit.x(0)
        entry["circuit"] = circuit
    elif attack == "tail_h":
        circuit = base["emitted"].copy()
        circuit.h(ideal.num_qubits - 1)
        entry["circuit"] = circuit
    elif attack == "gate_smuggle_u3":
        circuit = base["emitted"].copy()
        circuit.u(0.1, 0.2, 0.3, 0)
        entry["circuit"] = circuit
    elif attack == "gate_smuggle_measure":
        circuit = QuantumCircuit(base["emitted"].num_qubits, 1)
        circuit.compose(base["emitted"], inplace=True)
        circuit.measure(0, 0)
        entry["circuit"] = circuit
    elif attack == "cert_forge":
        forged = dict(base["certificate"])
        forged["accepted"] = not forged.get("accepted", True)
        entry["certificate"] = forged
    elif attack == "cert_replay":
        entry["certificate"] = ST.certify(
            base["schedule"], 1, circuit=base["emitted"],
            block_ranges=base["ranges"])
    else:
        raise ValueError(attack)
    return entry


def _emit_with_remap(ideal: QuantumCircuit, positions: List[int]):
    """Re-map the first block's controls/target (same arity, different unitary)."""
    out = QuantumCircuit(ideal.num_qubits)
    ranges: List[Tuple[int, int]] = []
    index = 0
    for position, instruction in enumerate(ideal.data):
        qubits = [ideal.find_bit(q).index for q in instruction.qubits]
        if position in positions:
            start = len(out.data)
            if index == 0 and len(qubits) == 3:
                _rccx(out, qubits[0], qubits[2], qubits[1])
            else:
                _rccx(out, *qubits)
            ranges.append((start, len(out.data)))
            index += 1
        else:
            out.append(instruction.operation, qubits)
    return out, ranges
