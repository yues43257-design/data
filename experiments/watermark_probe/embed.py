"""Embed a key into the exact/relative choice of each Toffoli block."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402

VARIANTS = {"cdkm3": 3, "cdkm4": 4}


def source_circuit(variant: str = "cdkm3") -> QuantumCircuit:
    return RB.to_canonical(CDKMRippleCarryAdder(VARIANTS[variant]))


def embed(key: Sequence[int], variant: str = "cdkm3") -> Dict[str, Any]:
    """bit 1 -> relative-phase block (4 T), bit 0 -> exact block (7 T)."""
    source = source_circuit(variant)
    positions = [i for i, ins in enumerate(source.data)
                 if ins.operation.name == "ccx"]
    capacity = len(positions)
    key_used = [int(b) & 1 for b in list(key)[:capacity]]
    key_used += [0] * (capacity - len(key_used))

    out = QuantumCircuit(source.num_qubits)
    index = 0
    for position, instruction in enumerate(source.data):
        qubits = [source.find_bit(q).index for q in instruction.qubits]
        if position in positions:
            if key_used[index] == 1:
                for gate in RCW.LM._rccx_gates(*qubits):
                    if gate[0] == "cx":
                        out.cx(gate[1], gate[2])
                    else:
                        getattr(out, gate[0])(gate[1])
            else:
                out.ccx(*qubits)
            index += 1
        else:
            out.append(instruction.operation, qubits,
                       [source.find_bit(c).index for c in instruction.clbits])

    schedule = RB._tracker_schedule(source)
    mask = sum(bit << i for i, bit in enumerate(key_used))
    legal = bool(ST.track_safe(schedule, mask))
    ones = sum(key_used)
    return {
        "variant": variant,
        "circuit": out,
        "source": source,
        "capacity": capacity,
        "key_used": key_used,
        "mask": mask,
        "tracker_safe": legal,
        "t_count": 7 * (capacity - ones) + 4 * ones,
        "t_overhead_vs_all_exact": -3 * ones,
        "t_overhead_vs_all_relative": 3 * (capacity - ones),
    }
