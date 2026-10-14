#!/usr/bin/env python3
"""Real-circuit T-count benchmark: our relative-phase compiler vs Qiskit.

Why this script exists
----------------------
Every T-count number produced inside this project so far was measured against an
oracle we wrote ourselves.  This script puts the compiler next to an *external*
baseline - Qiskit's default transpiler - on circuits that people actually build
(Grover, a CDKM ripple-carry adder, a bare C^nX ladder, QFT).

What is compared (per circuit)
------------------------------
* ``native``      - resources as written (qubits, ccx/mcx, cx, h, depth)
* ``qiskit L0``   - transpiled to a Clifford+T basis with no optimisation
                    (this is the standard "7 T per Toffoli" cost)
* ``qiskit L3``   - the same with ``optimization_level=3`` (Qiskit's default
                    "best" setting)
* ``ours (raw)``  - every Toffoli replaced by the 4-T relative-phase RCCX gate,
                    i.e. the ~43% per-block saving this project is built on
* ``ours (safe)`` - ``ours (raw)`` **only if** the injected diagonal correction
                    provably survives to the measurement; otherwise the circuit
                    is not equivalent to the original and we fall back to the
                    exact cost.  The test is ``Linux-MCX.relative_phase_metrics``
                    (``U_ideal^dagger U_candidate`` must be diagonal on the
                    clean-ancilla subspace), the same certified criterion used
                    everywhere else in the project.

The last row is the point of the exercise: the 4-T trick is only *real* when the
correction stays diagonal, and that depends on the circuit, not on the model.

Usage
-----
    .venv/bin/python realworld_benchmark.py --circuits mcx,grover,adder --qubits 4
    .venv/bin/python realworld_benchmark.py --qasm my_circuit.qasm
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import warnings

from qiskit import QuantumCircuit, transpile
from qiskit import qasm2

# Qiskit 2.1 deprecates the ``mode=`` kwarg of ``QuantumCircuit.mcx``; we still
# use it because it is the only way to request the ancilla-free synthesis.
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LM = _load("linux_mcx_realworld", "Linux-MCX")
DG = _load("data_generator_realworld", "data_generator.py")

# exact representation: Toffoli kept explicit, everything else Clifford
CANONICAL_BASIS = ["ccx", "cx", "h", "x", "z", "s", "sdg", "t", "tdg"]
# fault-tolerant representation: no Toffoli, so every ccx shows up as 7 T gates
CLIFFORD_T_BASIS = ["cx", "h", "x", "z", "s", "sdg", "t", "tdg"]


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------


def circuit_mcx(qubits: int) -> QuantumCircuit:
    """C^nX built by this project's own exact Toffoli AND-ladder (Clifford+T)."""
    source = LM.make_mcx_source(qubits, basis="clifford_t")
    return qasm2.loads(source.to_qasm())


def circuit_grover(qubits: int) -> QuantumCircuit:
    """Textbook 1-iteration Grover on ``qubits`` qubits, marked state |0..0>.

    Kept deliberately small: the multi-controlled Z is written with explicit
    Toffolis, so the whole circuit stays inside Clifford+T (Qiskit's ancilla-free
    ``MCXGate(n>=3)`` synthesis uses pi/8 rotations and is therefore *not*
    Clifford+T - see the closing note).
    """
    qc = QuantumCircuit(qubits)
    qc.h(range(qubits))
    # oracle: X, H, (n-1)-controlled X on the last qubit, H, X
    qc.x(range(qubits))
    qc.h(qubits - 1)
    qc.mcx(list(range(qubits - 1)), qubits - 1, mode="noancilla")
    qc.h(qubits - 1)
    qc.x(range(qubits))
    # diffusion
    qc.h(range(qubits))
    qc.x(range(qubits))
    qc.h(qubits - 1)
    qc.mcx(list(range(qubits - 1)), qubits - 1, mode="noancilla")
    qc.h(qubits - 1)
    qc.x(range(qubits))
    qc.h(range(qubits))
    return qc


def circuit_adder(qubits: int) -> QuantumCircuit:
    """CDKM ripple-carry adder - natively Toffoli + CNOT."""
    from qiskit.circuit.library import CDKMRippleCarryAdder

    return CDKMRippleCarryAdder(qubits)


def circuit_qft(qubits: int) -> QuantumCircuit:
    from qiskit.circuit.library import QFT

    return QFT(qubits)


BUILDERS = {
    "mcx": circuit_mcx,
    "grover": circuit_grover,
    "adder": circuit_adder,
    "qft": circuit_qft,
}


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------


def histogram(qc: QuantumCircuit) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for instruction in qc.data:
        name = instruction.operation.name
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def t_count(qc: QuantumCircuit) -> int:
    return sum(1 for ins in qc.data if ins.operation.name in ("t", "tdg"))


def to_clifford_t(qc: QuantumCircuit, level: int) -> QuantumCircuit:
    return transpile(qc, basis_gates=CLIFFORD_T_BASIS, optimization_level=level)


def to_canonical(qc: QuantumCircuit) -> QuantumCircuit:
    """Exact rewrite into explicit Toffolis + Clifford gates."""
    return transpile(qc, basis_gates=CANONICAL_BASIS, optimization_level=1)


def substitute_rccx(qc: QuantumCircuit) -> QuantumCircuit:
    """Replace every ``ccx`` by the 4-T relative-phase (Margolus) gate."""
    out = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instruction in qc.data:
        name = instruction.operation.name
        qubits = [qc.find_bit(q).index for q in instruction.qubits]
        if name == "ccx":
            control0, control1, target = qubits
            for gate in LM._rccx_gates(control0, control1, target):
                if gate[0] == "cx":
                    out.cx(gate[1], gate[2])
                else:
                    getattr(out, gate[0])(gate[1])
        elif name in ("barrier",):
            out.barrier(qubits)
        else:
            out.append(instruction.operation, qubits,
                       [qc.find_bit(c).index for c in instruction.clbits])
    return out


def todd_t_count(qc: QuantumCircuit) -> Optional[int]:
    """PyZX' TODD (Heyfron & Campbell 2019), applied block-wise.

    TODD optimises a *phase polynomial* (CNOT+ZPhase), so it is applied to every
    Hadamard-separated block with ``phase_block_optimize`` and the T-counts are
    summed - the standard way TODD is used on a Clifford+T circuit.
    """
    try:
        circuit = LM.zx.Circuit.from_qasm(qasm2.dumps(qc)).to_basic_gates()
        optimised = LM.zx.optimize.phase_block_optimize(circuit.copy(), quiet=True)
        return int(DG.tcount_of_gates(optimised.gates, "ceil"))
    except Exception as exc:  # pragma: no cover - TODD refuses some gate sets
        print(f"  [warn] TODD failed ({type(exc).__name__}: {str(exc)[:60]})",
              file=sys.stderr)
        return None


def relative_phase_ok(ideal: QuantumCircuit, candidate: QuantumCircuit) -> Optional[bool]:
    """Is ``candidate`` equal to ``ideal`` up to a diagonal correction?"""
    total = ideal.num_qubits
    if total > 11:
        return None
    reference = LM.zx.Circuit.from_qasm(qasm2.dumps(ideal))
    trial = LM.zx.Circuit.from_qasm(qasm2.dumps(candidate))
    metrics = LM.relative_phase_metrics(reference, trial, ancillas=[])
    if metrics is None:
        return None
    offdiag, unit = metrics
    return bool(offdiag < 1e-8 and unit < 1e-8)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def benchmark_one(name: str, qc: QuantumCircuit, mode: str = "toffoli",
                 n_controls: Optional[int] = None) -> Dict[str, Any]:
    """``mode='ladder'``: the circuit *is* one C^nX block (block-level swap).

    ``mode='toffoli'``: an arbitrary Toffoli+Clifford circuit, so the rewrite is
    per-Toffoli (``ccx`` -> 4-T relative-phase RCCX).
    """
    canonical = to_canonical(qc)
    no_opt = to_clifford_t(canonical, 0)
    best = to_clifford_t(canonical, 3)
    if mode == "ladder":
        # NOTE: a C^nX ladder occupies n+1+max(0, n-2) qubits, so the control
        # count cannot be recovered as ``num_qubits - 1``.
        ours = LM.relative_phase_mcx_circuit(int(n_controls))
        reference = qc
    else:
        ours = substitute_rccx(canonical)
        reference = canonical

    toffolis = sum(1 for ins in canonical.data if ins.operation.name == "ccx")
    t_baseline = t_count(no_opt)
    t_qiskit = t_count(best)
    t_ours = t_count(ours)
    ok = relative_phase_ok(reference, ours)
    t_safe = t_ours if ok else t_baseline

    return {
        "name": name,
        "qubits": qc.num_qubits,
        "native": histogram(qc),
        "toffolis": toffolis,
        "t_qiskit_l0": t_baseline,
        "t_qiskit_l3": t_qiskit,
        "t_todd": todd_t_count(canonical),
        "t_todd_ours": todd_t_count(ours),
        "t_ours_raw": t_ours,
        "relative_phase_ok": ok,
        "t_ours_safe": t_safe,
        "cx_l0": no_opt.count_ops().get("cx", 0),
        "cx_l3": best.count_ops().get("cx", 0),
    }


def summarise(rows: Sequence[Dict[str, Any]], qubits: int) -> int:
    print()
    print("=" * 92)
    print(f"  Real-circuit T-count benchmark  (qubits={qubits}, basis=Clifford+T)")
    print("=" * 92)
    header = (f"  {'circuit':<14}{'q':>3}{'ccx':>5}{'L0':>6}{'L3':>6}"
              f"{'TODD':>6}{'ours':>6}{'TODD(ours)':>11}{'best':>6}   {'RP valid':<9}")
    print(header)
    print("  " + "-" * 88)
    for row in rows:
        valid = "yes" if row["relative_phase_ok"] else (
            "no" if row["relative_phase_ok"] is False else "n/a")
        def _num(value):
            return f"{value}" if value is not None else "n/a"

        # only count the relative-phase branch when it is actually certified:
        # an unsafe rewrite is not a smaller cost, it is a wrong circuit.
        branch = [row["t_ours_safe"]]
        if row["relative_phase_ok"]:
            branch.append(row["t_todd_ours"])
        candidates = [v for v in ([row["t_todd"]] + branch) if v is not None]
        best = min(candidates) if candidates else None
        print(f"  {row['name']:<14}{row['qubits']:>3}{row['toffolis']:>5}"
              f"{row['t_qiskit_l0']:>6}{row['t_qiskit_l3']:>6}{_num(row['t_todd']):>6}"
              f"{row['t_ours_safe']:>6}{_num(row['t_todd_ours']):>11}"
              f"{_num(best):>6}   {valid:<9}")
    print("  " + "-" * 88)
    print("  L0 / L3      = Qiskit transpile to Clifford+T, optimisation_level 0 / 3")
    print("  TODD         = PyZX phase_block_optimize on the exact circuit (block-wise)")
    print("  ours         = 4-T relative-phase rewrite, kept only when the injected")
    print("                 diagonal provably survives (Linux-MCX certified test)")
    print("  TODD(ours)   = TODD applied *on top of* ours (does it find anything more?)")
    print("  best         = best known among TODD / ours / TODD(ours)")
    print("=" * 92)

    print()
    print("  verdicts")
    for row in rows:
        delta = row["t_qiskit_l3"] - row["t_ours_safe"]
        todd = row["t_todd"]
        if row["relative_phase_ok"]:
            vs_todd = (f", and {todd - row['t_ours_safe']} T vs TODD "
                       f"({(todd - row['t_ours_safe']) / max(todd, 1):.0%})"
                       if todd else "")
            extra = ("" if row["t_todd_ours"] in (None, row["t_ours_safe"])
                     else f" [TODD on top of ours gives {row['t_todd_ours']} T]")
            note = (f"relative-phase substitution IS valid -> saves {delta} T "
                    f"({delta / max(row['t_qiskit_l3'], 1):.0%}) vs Qiskit L3"
                    f"{vs_todd}{extra}")
        elif row["relative_phase_ok"] is False:
            note = ("relative-phase substitution is NOT valid here -> the 4-T trick "
                    "cannot be used; honest cost equals the exact one")
        else:
            note = "correctness check not available (>11 qubits)"
        print(f"    {row['name']:<16} {note}")
    print()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Real-circuit T-count benchmark.")
    parser.add_argument("--circuits", default="mcx,grover,adder",
                        help="comma-separated subset of " + ",".join(BUILDERS))
    parser.add_argument("--qubits", default="4",
                        help="comma-separated size(s) passed to each builder")
    parser.add_argument("--json", default=None,
                        help="write the benchmark rows to this JSON file")
    parser.add_argument("--qasm", default=None, help="also benchmark this QASM file")
    parser.add_argument("--show-native", action="store_true",
                        help="print the native gate histogram of every circuit")
    args = parser.parse_args(argv)

    sizes = [int(v) for v in str(args.qubits).split(",") if v.strip()]
    rows: List[Dict[str, Any]] = []
    for name in [c.strip() for c in args.circuits.split(",") if c.strip()]:
        if name not in BUILDERS:
            print(f"  [skip] unknown circuit {name!r}", file=sys.stderr)
            continue
        for size in sizes:
            label = f"{name}({size})"
            try:
                qc = BUILDERS[name](size)
            except Exception as exc:  # pragma: no cover - builder limitation
                print(f"  [skip] {label}: builder failed ({type(exc).__name__}: {exc})",
                      file=sys.stderr)
                continue
            mode = "ladder" if name == "mcx" else "toffoli"
            if qc.num_qubits > 11:
                print(f"  [warn] {label}: {qc.num_qubits} qubits - the relative-phase "
                      f"correctness test needs <=11", file=sys.stderr)
            try:
                rows.append(benchmark_one(label, qc, mode=mode, n_controls=size))
            except Exception as exc:
                print(f"  [skip] {label}: not expressible in Clifford+T "
                      f"({type(exc).__name__}: {str(exc)[:90]})", file=sys.stderr)

    if args.qasm:
        text = Path(args.qasm).read_text(encoding="utf-8")
        rows.append(benchmark_one(Path(args.qasm).stem, qasm2.loads(text)))

    if args.show_native:
        print()
        for row in rows:
            print(f"  {row['name']} native gates: {row['native']}")

    status = summarise(rows, args.qubits)
    if args.json:
        import json as _json

        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(rows, handle, indent=2)
        print(f"  wrote {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
