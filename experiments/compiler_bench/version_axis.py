#!/usr/bin/env python3
"""Compiler-version axis for the compiler-bug benchmark.

The same measurement-free circuits are evaluated in isolated Python
environments.  Each worker records four costs:

* Qiskit L0 and L3 on the Clifford+T basis;
* pyZX ``full_reduce`` plus ``extract_circuit``;
* TODD via pyZX ``phase_block_optimize``.

The worker deliberately imports no repository modules, so an old Qiskit
environment does not inherit the current environment's benchmark code.  The
parent launches one worker per environment and emits a single JSON artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

AXIS_VERSION = "compiler-version-axis-v1"
CLIFFORD_T_BASIS = ["cx", "h", "x", "z", "s", "sdg", "t", "tdg"]
CANONICAL_BASIS = ["ccx", "cx", "h", "x", "z", "s", "sdg", "t", "tdg"]
DEFAULT_CIRCUITS = (
    "adder_n3",
    "adder_n4",
    "vbe_n3",
    "comparator_n3",
    "two_block_h",
    "qasmbench/fredkin_n3",
    "qasmbench/sat_n7",
    "qasmbench/simon_n6",
    "qasmbench/toffoli_n3",
    "qasmbench/adder_n4",
    "qasmbench/adder_n10",
    "qasmbench/qec_en_n5",
    "qasmbench/teleportation_n3",
)
QASMBENCH = (
    "fredkin_n3", "sat_n7", "simon_n6", "toffoli_n3",
    "adder_n4", "adder_n10", "qec_en_n5", "teleportation_n3",
)
ROOT = Path(__file__).resolve().parents[2]


def t_count_from_gates(gates: Iterable[Any]) -> int:
    """Count T/Tdg gates, pyZX phase gates, and explicit Toffolis.

    pyZX represents a pi/4 phase as a ``ZPhase`` with ``phase == 1/4``
    (fractions are in units of pi), and pi/2 or pi phases are Clifford.
    Non-Clifford phases not on the pi/4 grid are rounded up conservatively.
    """
    total = 0
    for gate in gates:
        name = getattr(gate, "name", "")
        if name in ("T", "Tdg", "t", "tdg"):
            total += 1
            continue
        if name in ("Tof", "Toffoli", "toffoli", "ccx"):
            total += 7
            continue
        if name not in ("ZPhase", "ParityPhase"):
            continue
        phase = getattr(gate, "phase", None)
        if phase is None:
            total += 1
            continue
        units = abs(float(phase) * 4.0)
        nearest = round(units)
        if abs(units - nearest) <= 1e-9:
            total += int(nearest) % 2
        else:
            total += max(1, int(math.ceil(units - 1e-9)))
    return total


def _build_circuit(name: str, root: Path):
    from qiskit import QuantumCircuit, qasm2
    from qiskit.circuit.library import (
        CDKMRippleCarryAdder, IntegerComparator, VBERippleCarryAdder,
    )

    if name == "adder_n3":
        circuit = CDKMRippleCarryAdder(3)
        family = "adder"
    elif name == "adder_n4":
        circuit = CDKMRippleCarryAdder(4)
        family = "adder"
    elif name == "vbe_n3":
        circuit = VBERippleCarryAdder(3)
        family = "vbe"
    elif name == "comparator_n3":
        circuit = IntegerComparator(3, value=2)
        family = "comparator"
    elif name == "two_block_h":
        circuit = QuantumCircuit(5)
        circuit.ccx(0, 1, 2)
        circuit.h(2)
        circuit.ccx(3, 4, 2)
        family = "h_interleaved"
    elif name.startswith("qasmbench/"):
        stem = name.split("/", 1)[1]
        if stem not in QASMBENCH:
            raise ValueError(f"unknown QASMBench circuit: {stem}")
        path = root / "benchmarks" / "qasmbench_all" / f"{stem}.qasm"
        circuit = qasm2.loads(
            path.read_text(encoding="utf-8"),
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
        family = "qasmbench"
    else:
        raise ValueError(f"unknown circuit: {name}")
    return circuit, family


def _strip_terminal_measurements(circuit):
    """Return a unitary-only copy; reject dynamic operations rather than hide them."""
    if any(instruction.operation.name == "measure" for instruction in circuit.data):
        circuit = circuit.copy()
    else:
        # Some Qiskit 1.3 library objects lose constructor state on copy
        # (IntegerComparator.value is one example), so copy only when needed.
        circuit = circuit
    circuit.remove_final_measurements(inplace=True)
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("measure", "reset", "if_else"):
            raise ValueError(f"{name} is not supported by the compiler-version axis")
    return circuit


def _pyzx_ready(circuit):
    """Remove drawing-only directives before crossing the Qiskit/pyZX boundary."""
    circuit = circuit.copy()
    output = circuit.__class__(circuit.num_qubits)
    for instruction in circuit.data:
        if instruction.operation.name in ("barrier", "id"):
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        output.append(instruction.operation, qubits)
    return output


def _canonical_hash(circuit) -> str:
    gates = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("barrier", "id"):
            continue
        gates.append({
            "name": name,
            "qubits": [circuit.find_bit(q).index for q in instruction.qubits],
            "params": [round(float(param), 12)
                       for param in instruction.operation.params],
        })
    payload = json.dumps(gates, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _ToolTimeout(Exception):
    pass


def _timed(fn, timeout: float):
    started = time.perf_counter()
    previous = None

    def _handler(signum, frame):
        raise _ToolTimeout()

    if timeout:
        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        value = fn()
        return {"t_count": int(value), "seconds": round(time.perf_counter() - started, 6)}
    except _ToolTimeout:
        return {
            "t_count": None,
            "seconds": round(time.perf_counter() - started, 6),
            "error": f"timeout after {timeout:g}s",
        }
    except Exception as exc:
        return {
            "t_count": None,
            "seconds": round(time.perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if timeout:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def _worker(name: str, root: Path, tool_timeout: float) -> Dict[str, Any]:
    import qiskit
    import pyzx as zx
    from qiskit import qasm2
    from qiskit.compiler import transpile

    circuit, family = _build_circuit(name, root)
    unitary = _strip_terminal_measurements(circuit)
    source = transpile(
        unitary, basis_gates=CANONICAL_BASIS, optimization_level=0,
    )

    def qiskit_count(level: int) -> int:
        compiled = transpile(
            source, basis_gates=CLIFFORD_T_BASIS, optimization_level=level,
        )
        return sum(
            1 for instruction in compiled.data
            if instruction.operation.name in ("t", "tdg")
        )

    source_for_zx = _pyzx_ready(source)

    def from_qiskit():
        return zx.Circuit.from_qasm(qasm2.dumps(source_for_zx))

    def full_reduce() -> int:
        graph_circuit = from_qiskit()
        graph = graph_circuit.to_graph()
        zx.simplify.full_reduce(graph)
        extracted = zx.extract_circuit(graph)
        return t_count_from_gates(extracted.gates)

    def todd() -> int:
        basic = from_qiskit().to_basic_gates()
        optimised = zx.optimize.phase_block_optimize(basic.copy(), quiet=True)
        return t_count_from_gates(optimised.gates)

    zx_input = from_qiskit()
    outputs = {
        "qiskit_l0": _timed(lambda: qiskit_count(0), tool_timeout),
        "qiskit_l3": _timed(lambda: qiskit_count(3), tool_timeout),
        "pyzx_full_reduce": _timed(full_reduce, tool_timeout),
        "todd": _timed(todd, tool_timeout),
    }
    return {
        "name": name,
        "family": family,
        "qubits": source.num_qubits,
        "input": {
            "gates": len(source.data),
            "gate_counts": dict(sorted(Counter(
                instruction.operation.name for instruction in source.data
            ).items())),
            "pyzx_gates": len(zx_input.gates),
            "canonical_hash": _canonical_hash(source),
        },
        "outputs": outputs,
        "versions": {
            "qiskit_version": qiskit.__version__,
            "pyzx_version": zx.__version__,
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
    }


def worker_report(names: Sequence[str], root: Path,
                  tool_timeout: float = 30.0) -> Dict[str, Any]:
    import qiskit
    import pyzx as zx

    circuits: List[Dict[str, Any]] = []
    for name in names:
        try:
            circuits.append(_worker(name, root, tool_timeout))
        except Exception as exc:
            circuits.append({
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
                "outputs": {},
            })
    return {
        "qiskit_version": qiskit.__version__,
        "pyzx_version": zx.__version__,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "circuits": circuits,
    }


def _run_env(python: Path, names: Sequence[str], root: Path,
             timeout: float, tool_timeout: float) -> Dict[str, Any]:
    proc = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--worker",
         "--root", str(root), "--circuits", ",".join(names),
         "--tool-timeout", str(tool_timeout)],
        cwd=root, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker {python} failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[:1000]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker {python} emitted invalid JSON: {exc}; "
            f"stdout={proc.stdout[:500]!r}; stderr={proc.stderr[:500]!r}"
        ) from exc


def _older_python(explicit: Optional[str]) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("MCX_OLD_PYTHON"):
        candidates.append(Path(os.environ["MCX_OLD_PYTHON"]))
    candidates.extend((
        Path.home() / "venvs" / "oldver" / "bin" / "python",
        Path("/tmp/oldver/bin/python"),
        ROOT / ".venv-qiskit13" / "bin" / "python",
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "older compiler environment not found; create /tmp/oldver with "
        "`python3 -m venv /tmp/oldver && /tmp/oldver/bin/pip install "
        "qiskit==1.3.0 pyzx==0.7.0`, or set MCX_OLD_PYTHON"
    )


def run_axis(names: Sequence[str] = DEFAULT_CIRCUITS,
             old_python: Optional[str] = None,
             timeout: float = 180.0,
             tool_timeout: float = 30.0) -> Dict[str, Any]:
    current_python = Path(sys.executable)
    old = _older_python(old_python)
    current = _run_env(current_python, names, ROOT, timeout, tool_timeout)
    older = _run_env(old, names, ROOT, timeout, tool_timeout)

    by_env = {"current": current, "qiskit-1.3": older}
    circuits: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    input_mismatches: List[Dict[str, Any]] = []
    changes: Dict[str, int] = {key: 0 for key in
                               ("qiskit_l0", "qiskit_l3",
                                "pyzx_full_reduce", "todd")}
    deltas: Dict[str, List[int]] = {key: [] for key in changes}

    for index, name in enumerate(names):
        cells: Dict[str, Any] = {}
        base: Dict[str, Any] = {}
        for env_name, env_report in by_env.items():
            cell = env_report["circuits"][index]
            if cell.get("name") != name:
                raise RuntimeError(
                    f"worker ordering mismatch: expected {name}, got {cell.get('name')}"
                )
            cells[env_name] = cell
            if cell.get("error"):
                failures.append({"circuit": name, "environment": env_name,
                                 "error": cell["error"]})
            for tool, output in cell.get("outputs", {}).items():
                if output.get("error"):
                    failures.append({
                        "circuit": name, "environment": env_name, "tool": tool,
                        "error": output["error"],
                    })
            if not base and "input" in cell:
                base = cell["input"]
        input_hashes = {
            env_name: cell.get("input", {}).get("canonical_hash")
            for env_name, cell in cells.items()
        }
        if len(set(input_hashes.values())) != 1 or None in input_hashes.values():
            input_mismatches.append({"circuit": name, "hashes": input_hashes})
        for key in changes:
            left = cells["current"].get("outputs", {}).get(key, {}).get("t_count")
            right = cells["qiskit-1.3"].get("outputs", {}).get(key, {}).get("t_count")
            if left is not None and right is not None:
                delta = int(right) - int(left)
                deltas[key].append(abs(delta))
                changes[key] += int(delta != 0)
        circuits.append({
            "name": name,
            "qubits": cells["current"].get("qubits"),
            "family": cells["current"].get("family"),
            "input": base,
            "environments": cells,
        })

    environments = [
        {
            "name": "current",
            "python": current["python"],
            "python_version": current["python_version"],
            "qiskit_version": current["qiskit_version"],
            "pyzx_version": current["pyzx_version"],
        },
        {
            "name": "qiskit-1.3",
            "python": older["python"],
            "python_version": older["python_version"],
            "qiskit_version": older["qiskit_version"],
            "pyzx_version": older["pyzx_version"],
        },
    ]
    total_cells = len(names) * len(environments) * 4
    # A failed environment/circuit contributes four failed tool cells.
    successful_cells = total_cells - sum(
        4 if "tool" not in failure else 1 for failure in failures
    )
    return {
        "version": AXIS_VERSION,
        "environments": environments,
        "circuits": circuits,
        "summary": {
            "circuits": len(names),
            "total_cells": total_cells,
            "successful_cells": successful_cells,
            "failures": failures,
            "changed_circuits": changes,
            "cross_environment_input_mismatches": input_mismatches,
            "max_abs_t_count_delta": {
                key: max(values) if values else None
                for key, values in deltas.items()
            },
            "status": "measured" if not failures else "partial",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true",
                        help="internal: run one environment and print JSON")
    parser.add_argument("--circuits", default=",".join(DEFAULT_CIRCUITS))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--old-python", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--tool-timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "version_axis.json")
    args = parser.parse_args(argv)
    names = tuple(name.strip() for name in args.circuits.split(",") if name.strip())

    if args.worker:
        print(json.dumps(worker_report(names, args.root, args.tool_timeout),
                         sort_keys=True))
        return 0

    report = run_axis(names, args.old_python, args.timeout, args.tool_timeout)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True),
                           encoding="utf-8")
    print("=" * 100)
    print(f"  compiler-version axis  {AXIS_VERSION}")
    print("=" * 100)
    for env in report["environments"]:
        print(f"  {env['name']:<12} Qiskit {env['qiskit_version']:<8} "
              f"pyZX {env['pyzx_version']}")
    print(f"  {'circuit':<25}{'L0':>8}{'L3':>8}{'FR':>8}{'TODD':>8}")
    for circuit in report["circuits"]:
        current = circuit["environments"]["current"]["outputs"]
        print(f"  {circuit['name']:<25}"
              f"{str(current['qiskit_l0'].get('t_count')):>8}"
              f"{str(current['qiskit_l3'].get('t_count')):>8}"
              f"{str(current['pyzx_full_reduce'].get('t_count')):>8}"
              f"{str(current['todd'].get('t_count')):>8}")
    summary = report["summary"]
    print(f"  status={summary['status']} cells="
          f"{summary['successful_cells']}/{summary['total_cells']}")
    print(f"  changed={summary['changed_circuits']}")
    print(f"  wrote {args.output}")
    return 0 if summary["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
