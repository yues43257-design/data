#!/usr/bin/env python3
"""Recover exact Toffoli sub-circuits into ``ccx`` blocks.

Why: the support tracker works at **block granularity** (a Toffoli/MCX is one
permutation).  Gate-level QASM files -- including QASMBench -- usually ship the
7-T decomposition instead, so there are no block sites to certify.  This script
rewrites a gate-level circuit back into blocks, but **only** when the sub-circuit
is *exactly* a Toffoli:

    U(window) = e^{i phi} . U(ccx)      (global phase allowed)

Relative-phase variants (RCCX) are therefore **not** matched -- they differ by a
diagonal operator, which this tool must not silently erase.

What it does
  * strips **terminal** measurements only (flag-controlled); a measurement with
    any non-measure gate after it is mid-circuit and is preserved;
  * scans contiguous windows of <= ``max_window`` gates acting on exactly three
    qubits, and replaces a matching window by one ``ccx``;
  * every non-matching gate is copied unchanged.

What it does NOT do
  * no relative-phase matching, no approximate equivalence (tolerance 1e-9);
  * no re-synthesis front-end: if a circuit has no exact Toffoli window, the
    output is simply the (measurement-stripped) input, and the report says
    ``why: no matching pattern``;
  * it never edits ``support_tracker.py`` / ``relative_phase_criterion.py``.

Usage
-----
    out, report = recover(circuit_or_qasm_string)
    .venv/bin/python toffoli_recover.py            # census over benchmarks/qasmbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent
RECOVER_VERSION = "toffoli-recover-v1"
DEFAULT_MAX_WINDOW = 20
DEFAULT_TOL = 1e-9
_IGNORED = frozenset({"barrier", "id"})
# gates the shipped DiagonalTracker actually models (see support_tracker.py)
_TRACKER_GATES = frozenset({"h", "cx", "swap", "x", "z", "s", "sdg", "t",
                            "tdg", "cz", "rz", "id", "barrier", "ccx"})

# Qiskit's own 7-T decomposition (15 gates).  Kept as a fast-path label only;
# the generic window search below is what actually decides.
KNOWN_PATTERNS: Dict[str, List[str]] = {
    "qiskit_7t": ["h", "cx", "tdg", "cx", "t", "cx", "t", "tdg", "cx",
                  "cx", "t", "tdg", "cx", "t", "h"],
}


def _toffoli_matrix(target: int) -> np.ndarray:
    controls = [q for q in range(3) if q != target]
    circuit = QuantumCircuit(3)
    circuit.ccx(controls[0], controls[1], target)
    return np.asarray(Operator(circuit).data, dtype=complex)


_TOFFOLI = {t: _toffoli_matrix(t) for t in range(3)}


def _as_circuit(circuit_or_qasm: Any) -> QuantumCircuit:
    if isinstance(circuit_or_qasm, QuantumCircuit):
        return circuit_or_qasm
    if isinstance(circuit_or_qasm, str):
        return qasm2.loads(circuit_or_qasm)
    raise TypeError("expected a QuantumCircuit or a QASM string")


def _empty_like(circuit: QuantumCircuit) -> QuantumCircuit:
    try:
        return circuit.copy_empty_like()
    except Exception:                       # pragma: no cover - old qiskit
        return QuantumCircuit(circuit.num_qubits, circuit.num_clbits)


def _append(out: QuantumCircuit, circuit: QuantumCircuit, instruction) -> None:
    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
    clbits = [circuit.find_bit(c).index for c in instruction.clbits]
    out.append(instruction.operation, qubits, clbits)


def strip_terminal_measurements(
        circuit: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[str, int]]:
    """Remove measurements that form the trailing suffix; keep mid-circuit ones."""
    data = list(circuit.data)
    last_unitary = -1
    for index, instruction in enumerate(data):
        if instruction.operation.name not in ("measure", "barrier", "id"):
            last_unitary = index
    out = _empty_like(circuit)
    removed = mid = 0
    for index, instruction in enumerate(data):
        if instruction.operation.name == "measure":
            if index > last_unitary:
                removed += 1
                continue
            mid += 1
        _append(out, circuit, instruction)
    return out, {"removed_terminal_measurements": removed,
                 "mid_circuit_measurements": mid}


def _phase_equal(u: np.ndarray, v: np.ndarray, tol: float) -> bool:
    """True iff ``u = e^{i phi} v`` (no relative phase)."""
    m = u @ v.conj().T
    off = float(np.abs(m - np.diag(np.diag(m))).max())
    if off >= tol:
        return False
    diagonal = np.diag(m)
    return float(np.abs(diagonal - diagonal[0]).max()) < tol


def _match_window(circuit: QuantumCircuit, start: int, length: int,
                  tol: float) -> Optional[Dict[str, Any]]:
    """Try to match ``circuit.data[start:start+length]`` to one Toffoli."""
    window = list(circuit.data[start:start + length])
    # An existing single ccx is already a block: copying it is a no-op, and
    # counting it as "recovered" would inflate the report.
    if len(window) == 1 and window[0].operation.name == "ccx":
        return None
    if any(ins.clbits for ins in window):
        return None
    qubits: List[int] = []
    for instruction in window:
        for q in instruction.qubits:
            index = circuit.find_bit(q).index
            if index not in qubits:
                qubits.append(index)
        if len(qubits) > 3:
            return None
    if len(qubits) != 3:
        return None
    qubits.sort()
    local_index = {q: i for i, q in enumerate(qubits)}
    local = QuantumCircuit(3)
    for instruction in window:
        if instruction.operation.name in _IGNORED:
            continue
        local.append(instruction.operation,
                     [local_index[circuit.find_bit(q).index]
                      for q in instruction.qubits])
    matrix = np.asarray(Operator(local).data, dtype=complex)
    for target in range(3):
        if _phase_equal(matrix, _TOFFOLI[target], tol):
            controls = [q for q in range(3) if q != target]
            names = [ins.operation.name for ins in window]
            label = next((name for name, pattern in KNOWN_PATTERNS.items()
                          if names == pattern), "search")
            return {"qubits": qubits, "target_local": target,
                    "controls_local": controls, "pattern": label,
                    "gates": length}
    return None


def recover(circuit_or_qasm: Any, *,
            strip_terminal_measurements: bool = True,
            max_window: int = DEFAULT_MAX_WINDOW,
            tol: float = DEFAULT_TOL,
            timeout_s: float = 30.0) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """Rewrite exact-Toffoli windows into ``ccx`` blocks."""
    import time

    started = time.perf_counter()
    circuit = _as_circuit(circuit_or_qasm)
    strip_report = {"removed_terminal_measurements": 0,
                    "mid_circuit_measurements": 0}
    if strip_terminal_measurements:
        circuit, strip_report = strip_terminal_measurements_fn(circuit)
    data = list(circuit.data)
    out = _empty_like(circuit)
    matches: List[Dict[str, Any]] = []
    windows_tried = 0
    timed_out = False
    index = 0
    while index < len(data):
        if time.perf_counter() - started > timeout_s:
            timed_out = True
            for instruction in data[index:]:
                _append(out, circuit, instruction)
            break
        matched = None
        longest = min(max_window, len(data) - index)
        for length in range(longest, 0, -1):
            windows_tried += 1
            matched = _match_window(circuit, index, length, tol)
            if matched:
                break
        if matched:
            controls = [matched["qubits"][i] for i in matched["controls_local"]]
            target = matched["qubits"][matched["target_local"]]
            out.ccx(controls[0], controls[1], target)
            matches.append({**matched, "start": index,
                            "global_qubits": matched["qubits"],
                            "global_target": target})
            index += matched["gates"]
        else:
            _append(out, circuit, data[index])
            index += 1

    names = [ins.operation.name for ins in out.data]
    remaining_measure = names.count("measure")
    remaining_ccx = names.count("ccx")
    dynamic = any(name in ("measure", "reset", "if_else") for name in names)
    unsupported = sorted({name for name in names
                          if name not in _TRACKER_GATES})
    report: Dict[str, Any] = {
        "version": RECOVER_VERSION,
        **strip_report,
        "input_gates": len(data),
        "recovered_blocks": len(matches),
        "windows_tried": windows_tried,
        "seconds": round(time.perf_counter() - started, 4),
        "timed_out": timed_out,
        "matches": matches,
        "remaining_gates": len(out.data),
        "remaining_ccx": remaining_ccx,
        "remaining_measurements": remaining_measure,
        "in_tracker_domain": bool(remaining_ccx > 0 and not dynamic
                                  and not unsupported),
        "unsupported_gates": unsupported,
        "unmatched_reason": (None if matches else
                             "why: no matching pattern"),
    }
    if out.num_qubits <= 10 and not dynamic:
        recovered_u = np.asarray(Operator(out).data, dtype=complex)
        original_u = np.asarray(Operator(circuit).data, dtype=complex)
        m = recovered_u @ original_u.conj().T
        off = float(np.abs(m - np.diag(np.diag(m))).max())
        diagonal = np.diag(m)
        report["equivalence_checked"] = True
        report["equivalence_max_error"] = max(
            off, float(np.abs(diagonal - diagonal[0]).max()))
    else:
        report["equivalence_checked"] = False
        report["equivalence_max_error"] = None
        report["equivalence_skipped_because"] = (
            "non-unitary (measure/reset/if_else)" if dynamic else "wider than 10 qubits")
    return out, report


# keep the public name and the helper distinct despite the shadowing parameter
strip_terminal_measurements_fn = strip_terminal_measurements


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasm-dir", default=str(ROOT / "benchmarks" / "qasmbench"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "toffoli_recover.json"))
    args = parser.parse_args(argv)
    rows = []
    print("=" * 100)
    print(f"  toffoli_recover  {RECOVER_VERSION}")
    print("=" * 100)
    print(f"  {'circuit':<16}{'in':>5}{'ccx':>6}{'out':>6}{'meas':>6}{'in-domain':>11}  reason")
    for path in sorted(Path(args.qasm_dir).glob("*.qasm")):
        out, report = recover(path.read_text())
        rows.append({"circuit": path.stem, **report})
        print(f"  {path.stem:<16}{report['input_gates']:>5}"
              f"{report['recovered_blocks']:>6}{report['remaining_gates']:>6}"
              f"{report['remaining_measurements']:>6}"
              f"{str(report['in_tracker_domain']):>11}  "
              f"{report['unmatched_reason'] or 'ok'}")
    Path(args.out).write_text(json.dumps({"version": RECOVER_VERSION, "rows": rows},
                                         indent=2), encoding="utf-8")
    print("=" * 100)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
