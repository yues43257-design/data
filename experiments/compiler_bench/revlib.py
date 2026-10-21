"""Minimal RevLib 2.0 ``.real`` parser → qiskit QuantumCircuit.

Supported gate lines: ``t1 q`` (X), ``t2 c t`` (CX), ``tk c1 .. ck t``
(multi-controlled X).  RevLib semantics: the line applies X to the LAST
argument, controlled on the preceding ones.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from qiskit import QuantumCircuit


def parse(path: Path) -> QuantumCircuit:
    names: List[str] = []
    gates: List[List[str]] = []
    inside = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(".variables"):
            names = line.split()[1:]
        elif line == ".begin":
            inside = True
        elif line == ".end":
            inside = False
        elif inside:
            tokens = line.split()
            if re.fullmatch(r"t\d+", tokens[0]):
                gates.append(tokens[1:])
    if not names:
        raise ValueError("no .variables line")
    index: Dict[str, int] = {name: i for i, name in enumerate(names)}
    circuit = QuantumCircuit(len(names))
    for args in gates:
        qubits = [index[a] for a in args]
        if len(qubits) == 1:
            circuit.x(qubits[0])
        elif len(qubits) == 2:
            circuit.cx(qubits[0], qubits[1])
        else:
            circuit.mcx(qubits[:-1], qubits[-1])
    return circuit
