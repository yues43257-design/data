#!/usr/bin/env python3
"""Measure the *relative-phase* T-count headroom of multi-controlled X ladders.

The routing-policy experiment showed zero headroom: for a Clifford+T block the
T-count is fixed by its phase polynomial (the number of odd Fourier
coefficients) regardless of how it is routed.  The one lever that *does* change
the T-count is allowing **relative phase**: replacing each exact 7-T Toffoli by
the 4-T Margolus (RCCX) gate.

This script builds both variants of C^nX, reports their FTQC costs, and measures
how far the relative-phase variant is from the ideal gate -- so you can decide
whether "choose the low-T relative-phase variant per block" is a worthwhile
learning target for the GNN before training anything.

Usage
-----
    .venv/bin/python relative_phase_headroom.py --max-controls 6
    .venv/bin/python relative_phase_headroom.py --max-controls 4 --json /tmp/rp.json
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
LINUX_MCX_PATH = ROOT / "Linux-MCX"


def load_linux_mcx():
    loader = importlib.machinery.SourceFileLoader("linux_mcx_relphase", str(LINUX_MCX_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {LINUX_MCX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402
from qiskit.circuit.library import MCXGate  # noqa: E402


def rccx(qc: QuantumCircuit, control0: int, control1: int, target: int) -> None:
    """4-T Margolus / RCCX relative-phase Toffoli (exact Clifford+T)."""
    qc.h(target)
    qc.t(target)
    qc.cx(control1, target)
    qc.tdg(target)
    qc.cx(control0, target)
    qc.t(target)
    qc.cx(control1, target)
    qc.tdg(target)
    qc.h(target)


def relative_phase_ladder(num_controls: int) -> QuantumCircuit:
    """The same clean-ancilla AND ladder as the exact variant, with RCCX gates."""
    n_ancillas = max(0, num_controls - 2)
    total = num_controls + 1 + n_ancillas
    controls = list(range(num_controls))
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    target = total - 1
    qc = QuantumCircuit(total)
    if num_controls == 1:
        qc.cx(controls[0], target)
        return qc
    if num_controls == 2:
        rccx(qc, controls[0], controls[1], target)
        return qc
    rccx(qc, controls[0], controls[1], ancillas[0])
    for i in range(1, n_ancillas):
        rccx(qc, ancillas[i - 1], controls[i + 1], ancillas[i])
    rccx(qc, ancillas[-1], controls[-1], target)
    for i in reversed(range(1, n_ancillas)):
        rccx(qc, ancillas[i - 1], controls[i + 1], ancillas[i])
    rccx(qc, controls[0], controls[1], ancillas[0])
    return qc


def unitary_error(candidate: QuantumCircuit, num_controls: int, max_dense_qubits: int = 9):
    """Distance between a candidate C^nX and the ideal gate, up to a global phase.

    The ladders place the *target on the highest qubit index* with the clean
    ancillas below it, so the reference must act on ``controls + [last qubit]``.
    (Using ``range(num_controls + 1)`` compares the wrong target and reports a
    spurious error of 1.0 -- exactly what the first version of this script did.)

    Dense operators are used when the circuit is small enough; otherwise a
    statevector comparison is used in which **half of the sampled states
    trigger the gate** (a uniform sample triggers with probability 2^-(n+1) and
    would make a wrong circuit look exact).
    """
    total = candidate.num_qubits
    n_ancillas = max(0, num_controls - 2)
    controls = list(range(num_controls))
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    target = total - 1
    reference = QuantumCircuit(total)
    reference.append(MCXGate(num_controls), controls + [target])

    if total <= max_dense_qubits:
        keep = [x for x in range(2 ** total) if all(not (x >> a) & 1 for a in ancillas)]
        actual = Operator(candidate).data[np.ix_(keep, keep)]
        ideal = Operator(reference).data[np.ix_(keep, keep)]
        scale = complex(np.vdot(ideal, actual) / max(np.vdot(ideal, ideal).real, 1e-12))
        return float(np.max(np.abs(actual - ideal * scale))), "dense"

    rng = np.random.default_rng(0)
    free = [q for q in range(total) if q not in ancillas and q not in controls and q != target]
    ratios = []
    triggers = 0
    for index in range(64):
        basis = 0
        if index % 2 == 0:                       # guaranteed trigger
            for q in controls + [target]:
                basis |= 1 << q
            triggers += 1
        else:                                    # random control pattern
            for q in controls + [target]:
                if rng.random() < 0.5:
                    basis |= 1 << q
        for q in free:
            if rng.random() < 0.5:
                basis |= 1 << q
        a = Statevector.from_int(basis, 2 ** total).evolve(candidate).data
        b = Statevector.from_int(basis, 2 ** total).evolve(reference).data
        ia, ib = int(np.argmax(np.abs(a))), int(np.argmax(np.abs(b)))
        if ia != ib:
            return 1.0, "sampled(index mismatch)"
        ratios.append(a[ia] / b[ib])
    ratios = np.array(ratios)
    return float(np.max(np.abs(ratios / ratios[0] - 1.0))), f"sampled({triggers} triggers)"


def relative_phase_error(candidate: QuantumCircuit, num_controls: int, max_dense_qubits: int = 11):
    """Check the *relative-phase* criterion: U_ideal^dagger U_candidate must be diagonal with +-1 entries.

    This is the correct acceptance test for the Margolus/RCCX family: those
    circuits differ from the exact gate by a **diagonal Clifford correction**
    (phases on basis states), not by a global phase.  Comparing only up to a
    global phase (as ``unitary_error`` does) therefore reports a large number
    even for a perfectly usable relative-phase construction.
    """
    total = candidate.num_qubits
    n_ancillas = max(0, num_controls - 2)
    controls = list(range(num_controls))
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    target = total - 1
    reference = QuantumCircuit(total)
    reference.append(MCXGate(num_controls), controls + [target])
    if total > max_dense_qubits:
        return None
    keep = [x for x in range(2 ** total) if all(not (x >> a) & 1 for a in ancillas)]
    actual = Operator(candidate).data[np.ix_(keep, keep)]
    ideal = Operator(reference).data[np.ix_(keep, keep)]
    correction = ideal.conj().T @ actual
    offdiag = float(np.max(np.abs(correction - np.diag(np.diag(correction)))))
    diagonal = np.diag(correction)
    unit_phase = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
    return offdiag, unit_phase


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Relative-phase T-count headroom for C^nX.")
    parser.add_argument("--max-controls", type=int, default=6)
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-8, help="'exact' threshold for the error")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    rows: "list[dict[str, Any]]" = []
    print(
        f"{'gate':>5} | {'exact T':>7} {'exact dep':>9} {'exact err':>10} | "
        f"{'RCCX T':>6} {'RCCX dep':>8} {'RCCX err':>9} | {'verdict':>22}"
    )
    for num_controls in range(args.min_controls, args.max_controls + 1):
        exact = LINUX_MCX.make_mcx_source(num_controls, basis="clifford_t")
        exact_metrics = LINUX_MCX.compute_ftq_metrics(exact)
        exact_err, exact_mode = unitary_error(
            qasm2.loads(exact.to_qasm()), num_controls
        )

        rel_circuit = relative_phase_ladder(num_controls)
        rel_source = LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(rel_circuit)).to_basic_gates()
        rel_metrics = LINUX_MCX.compute_ftq_metrics(rel_source)
        rel_err, rel_mode = unitary_error(rel_circuit, num_controls)

        rp = relative_phase_error(rel_circuit, num_controls)
        if rel_err <= args.tolerance:
            verdict = "exact"
        elif rp is not None and rp[0] <= args.tolerance and rp[1] <= args.tolerance:
            verdict = "relative-phase OK"
        else:
            verdict = "NOT reversible-equivalent"
        print(
            f"C^{num_controls}X | {exact_metrics.t_count:7d} {exact_metrics.t_depth:9d} "
            f"{exact_err:10.2e} | {rel_metrics.t_count:6d} {rel_metrics.t_depth:8d} "
            f"{rel_err:9.2e} | {verdict:>22}"
        )
        rows.append(
            {
                "num_controls": num_controls,
                "exact": {
                    "t_count": exact_metrics.t_count,
                    "t_depth": exact_metrics.t_depth,
                    "cx_depth": exact_metrics.cx_depth,
                    "error": exact_err,
                    "mode": exact_mode,
                },
                "relative_phase": {
                    "t_count": rel_metrics.t_count,
                    "t_depth": rel_metrics.t_depth,
                    "cx_depth": rel_metrics.cx_depth,
                    "error": rel_err,
                    "mode": rel_mode,
                },
                "relative_phase_ok": bool(
                    rp is not None and rp[0] <= args.tolerance and rp[1] <= args.tolerance
                ),
                "relative_phase_offdiag": None if rp is None else rp[0],
                "relative_phase_unit": None if rp is None else rp[1],
                "delta_t_count": exact_metrics.t_count - rel_metrics.t_count,
                "delta_t_depth": exact_metrics.t_depth - rel_metrics.t_depth,
                "verdict": verdict,
            }
        )

    gains = [row["delta_t_count"] for row in rows if row.get("relative_phase_ok")]
    print(
        f"\n    [headroom] T-count gain from the relative-phase ladder: "
        f"min={min(gains) if gains else 0}, max={max(gains) if gains else 0} "
        f"(positive = the RCCX variant uses fewer T gates)"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
        print(f"    [json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
