"""Read back the exact/relative choice of every Toffoli block."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402

_RCCX_NAMES = ["h", "t", "cx", "tdg", "cx", "t", "cx", "tdg", "h"]


def _qubits(circuit: QuantumCircuit, instruction) -> List[int]:
    return [circuit.find_bit(q).index for q in instruction.qubits]


def _rccx_at(circuit: QuantumCircuit, start: int) -> int:
    data = circuit.data
    if start + len(_RCCX_NAMES) > len(data):
        return 0
    window = data[start:start + len(_RCCX_NAMES)]
    if [ins.operation.name for ins in window] != _RCCX_NAMES:
        return 0
    qs = [_qubits(circuit, ins) for ins in window]
    target = qs[0][0]
    if qs[1] != [target] or qs[3] != [target] or qs[5] != [target] \
            or qs[7] != [target] or qs[8] != [target]:
        return 0
    if qs[2][1] != target or qs[4][1] != target or qs[6][1] != target:
        return 0
    if qs[2][0] == qs[4][0] or qs[6] != qs[2]:
        return 0
    return len(_RCCX_NAMES)


def detect_bits(circuit: QuantumCircuit) -> Tuple[List[int], int, int]:
    """Return (bits, blocks_found, blocks_expected_or_found)."""
    bits: List[int] = []
    index = 0
    data = circuit.data
    while index < len(data):
        name = data[index].operation.name
        if name == "ccx":
            bits.append(0)
            index += 1
            continue
        length = _rccx_at(circuit, index)
        if length:
            bits.append(1)
            index += length
            continue
        index += 1
    return bits, len(bits), len(bits)


def ber(bits: Sequence[int], key: Sequence[int]) -> float:
    if not key:
        return float("nan")
    n = min(len(bits), len(key))
    if n == 0:
        return 0.5
    errors = sum(1 for a, b in zip(bits[:n], key[:n]) if int(a) != int(b))
    return errors / n


def count_t(circuit: QuantumCircuit) -> int:
    """T-count treating a ccx as 7 T and an RCCX pattern as 4 T."""
    total = 0
    index = 0
    data = circuit.data
    while index < len(data):
        name = data[index].operation.name
        if name == "ccx":
            total += 7
            index += 1
            continue
        length = _rccx_at(circuit, index)
        if length:
            total += 4
            index += length
            continue
        if name in ("t", "tdg"):
            total += 1
        elif name in ("rz", "p", "u1") and data[index].operation.params:
            # pyZX extraction emits rz(pi/4) instead of t: count odd multiples
            # of pi/4; even multiples are Clifford (S, Z, ...).
            import math
            theta = float(data[index].operation.params[0])
            k = round(theta / (math.pi / 4))
            total += 1 if k % 2 else 0
        index += 1
    return total
