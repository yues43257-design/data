"""QASMBench in-domain cases (the 4/42 that `toffoli_recover` brings into the
block-level tracker domain).

For each circuit: recover the Toffoli blocks, build the all-relative rewrite,
and run four attacks through the frozen protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402
import toffoli_recover as TR          # noqa: E402

DIR = ROOT / "benchmarks" / "qasmbench_all"
REAL = ("fredkin_n3", "sat_n7", "simon_n6", "toffoli_n3")
ATTACKS = ("legal", "tail_h", "cert_forge", "cert_replay")


def _rccx(circuit: QuantumCircuit, c0: int, c1: int, target: int) -> None:
    for gate in RCW.LM._rccx_gates(c0, c1, target):
        if gate[0] == "cx":
            circuit.cx(gate[1], gate[2])
        else:
            getattr(circuit, gate[0])(gate[1])


def _all_relative(ideal: QuantumCircuit) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    positions = {i for i, ins in enumerate(ideal.data)
                 if ins.operation.name == "ccx"}
    out = QuantumCircuit(ideal.num_qubits)
    ranges: List[Tuple[int, int]] = []
    for position, instruction in enumerate(ideal.data):
        qubits = [ideal.find_bit(q).index for q in instruction.qubits]
        if position in positions:
            start = len(out.data)
            _rccx(out, *qubits)
            ranges.append((start, len(out.data)))
        else:
            out.append(instruction.operation, qubits)
    return out, ranges


def real_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for name in REAL:
        path = DIR / f"{name}.qasm"
        if not path.exists():
            cases.append({"name": f"real/{name}", "error": "qasm file missing"})
            continue
        try:
            source = qasm2.loads(path.read_text())
            recovered, report = TR.recover(source)
        except Exception as exc:
            cases.append({"name": f"real/{name}",
                          "error": f"recover failed: {type(exc).__name__}: {exc}"})
            continue
        if report["remaining_ccx"] == 0:
            cases.append({"name": f"real/{name}",
                          "error": "no ccx blocks after recovery"})
            continue
        ideal = recovered
        emitted, ranges = _all_relative(ideal)
        schedule = RB._tracker_schedule(ideal)
        mask = (1 << len(schedule["blocks"])) - 1
        certificate = ST.certify(schedule, mask, circuit=emitted,
                                 block_ranges=ranges)
        base = {"ideal": ideal, "emitted": emitted, "ranges": ranges,
                "schedule": schedule, "mask": mask, "certificate": certificate}
        for attack in ATTACKS:
            entry = dict(base, attack=attack, circuit=emitted,
                         certificate=certificate)
            if attack == "tail_h":
                circuit = emitted.copy()
                circuit.h(ideal.num_qubits - 1)
                entry["circuit"] = circuit
            elif attack == "cert_forge":
                forged = dict(certificate)
                forged["accepted"] = not forged.get("accepted", True)
                entry["certificate"] = forged
            elif attack == "cert_replay":
                # mask=0 (all-exact certificate) differs from the all-relative
                # mask even when the circuit has a single block.
                entry["certificate"] = ST.certify(
                    schedule, 0, circuit=emitted, block_ranges=ranges)
            cases.append({
                "name": f"real/{name}/{attack}", "family": "real_qasmbench",
                "n": ideal.num_qubits, "layout": "recovered", "seed": 0,
                "attack": attack, "qubits": ideal.num_qubits,
                "cert_property": None,      # assigned after L is known
                "ground_truth": {"legal": "none", "tail_h": "tail H after the last block",
                                 "cert_forge": "accepted flipped",
                                 "cert_replay": "cert replayed from mask 1"}[attack],
                "truth_L": None, "truth_S": None, "entry": entry,
                "real_circuit": name,
            })
    return cases
