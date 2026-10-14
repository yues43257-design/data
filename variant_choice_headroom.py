#!/usr/bin/env python3
"""Strategy-B headroom: how many MCX blocks may safely be *downgraded* to the
relative-phase (RCCX) variant inside a real circuit?

The relative-phase ladder is much cheaper (C^4X: 20 T instead of 35 T) but it
differs from the exact gate by a **diagonal +-1 correction**.  Inside a larger
circuit that correction is harmless only when it can be absorbed by the
surrounding Clifford layers (equivalently: when the composed correction of the
whole circuit is still a Z-type correction, i.e. diagonal +-1).  Which subset of
blocks may be downgraded is therefore a *combinatorial* question -- exactly the
kind of structure a learned policy could exploit, and exactly what has to be
measured before training anything.

This script answers it by brute force on a small instance: ``k`` C^3X blocks
separated by random Clifford layers on the data qubits, all 2^k variant
assignments, each classified as

    exact           U_mixed == U_exact (up to global phase)
    z-correction    U_exact^dagger U_mixed is diagonal with +-1 entries
                    (usable: a Z correction, absorbable into the final Z-basis
                    measurement / subsequent Clifford gates)
    other           any other correction -> NOT usable

and reports the T-count of the best usable assignment (the headroom).

Usage
-----
    .venv/bin/python variant_choice_headroom.py --blocks 3 --seeds 5
    .venv/bin/python variant_choice_headroom.py --blocks 4 --seeds 3 --json /tmp/vc.json
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
LINUX_MCX_PATH = ROOT / "Linux-MCX"


def load_linux_mcx():
    loader = importlib.machinery.SourceFileLoader("linux_mcx_variant", str(LINUX_MCX_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {LINUX_MCX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402


def exact_block(num_controls: int) -> QuantumCircuit:
    source = LINUX_MCX.make_mcx_source(num_controls, basis="clifford_t")
    return qasm2.loads(source.to_qasm())


def relative_block(num_controls: int) -> QuantumCircuit:
    return LINUX_MCX.relative_phase_mcx_circuit(num_controls)


def clifford_layer(
    qc: QuantumCircuit,
    qubits: Sequence[int],
    rng: random.Random,
    interconnect: str = "clifford",
) -> None:
    """Random single-/two-qubit layer on ``qubits``.

    ``interconnect="clifford"`` includes Hadamards; ``"h_free"`` restricts the
    layer to {X, Z, S, CNOT}, which is the regime where a diagonal (Z-type)
    correction stays diagonal and a relative-phase downgrade is *usable*.  The
    regime matters: with Hadamards after a downgraded block the correction turns
    into an X-type error and destroys Z-basis measurements.
    """
    gate_set = ("h", "s", "x", "z", "cx") if interconnect == "clifford" else ("s", "x", "z", "cx")
    for _ in range(rng.randint(1, 3) * len(qubits)):
        choice = rng.choice(gate_set)
        q = rng.choice(qubits)
        if choice == "cx":
            other = rng.choice([p for p in qubits if p != q])
            qc.cx(q, other)
        else:
            getattr(qc, choice)(q)


def collect_blocks(num_controls: int, blocks: int, seed: int):
    """Build the data-qubit indices, the exact reference and the layer plan."""
    n_ancillas = max(0, num_controls - 2)
    total = num_controls + 1 + n_ancillas
    data_qubits = list(range(num_controls + 1))
    rng = random.Random(seed)
    return total, data_qubits, n_ancillas, rng


def classify_reference(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> str:
    """Classify ``candidate`` against ``reference`` on the data subspace."""
    scale = complex(np.vdot(reference, candidate) / max(np.vdot(reference, reference).real, 1e-12))
    if float(np.max(np.abs(candidate - reference * scale))) <= tolerance:
        return "exact"
    correction = reference.conj().T @ candidate
    offdiag = float(np.max(np.abs(correction - np.diag(np.diag(correction)))))
    unit = float(np.max(np.abs(np.abs(np.diag(correction)) - 1.0)))
    if offdiag <= tolerance and unit <= tolerance:
        return "z-correction"
    return "other"


def measure_instance(
    num_controls: int,
    blocks: int,
    seed: int,
    tolerance: float = 1e-8,
    interconnect: str = "clifford",
):
    """Brute-force every variant assignment and return the headroom table."""
    total, data_qubits, n_ancillas, rng = collect_blocks(num_controls, blocks, seed)
    exact_reference = QuantumCircuit(total)
    relative_only = QuantumCircuit(total)
    layers = []
    for index in range(blocks):
        for circuit in (exact_reference, relative_only):
            if index:
                clifford_layer(circuit, data_qubits, rng, interconnect)
            if circuit is exact_reference:
                circuit.compose(exact_block(num_controls), range(total), inplace=True)
            else:
                circuit.compose(relative_block(num_controls), range(total), inplace=True)
        layer_seed = rng.randrange(1 << 30)
        layers.append(layer_seed)

    keep = [x for x in range(2 ** total) if all(not (x >> a) & 1 for a in range(num_controls + 1, total))]
    reference_matrix = Operator(exact_reference).data[np.ix_(keep, keep)]
    t_exact_per_block = LINUX_MCX.compute_ftq_metrics(
        LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(exact_block(num_controls))).to_basic_gates()
    ).t_count
    t_relative_per_block = LINUX_MCX.compute_ftq_metrics(
        LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(relative_block(num_controls))).to_basic_gates()
    ).t_count

    rows = []
    for mask in range(1 << blocks):
        circuit = QuantumCircuit(total)
        layer_rng = random.Random(seed)
        for index in range(blocks):
            if index:
                clifford_layer(circuit, data_qubits, layer_rng, interconnect)
            block = exact_block(num_controls) if (mask >> index) & 1 == 0 else relative_block(num_controls)
            circuit.compose(block, range(total), inplace=True)
        matrix = Operator(circuit).data[np.ix_(keep, keep)]
        verdict = classify_reference(reference_matrix, matrix, tolerance)
        downgraded = bin(mask).count("1")
        t_count = (blocks - downgraded) * t_exact_per_block + downgraded * t_relative_per_block
        rows.append(
            {
                "mask": mask,
                "downgraded": downgraded,
                "verdict": verdict,
                "t_count": t_count,
                "usable": verdict in ("exact", "z-correction"),
            }
        )
    usable = [row for row in rows if row["usable"]]
    best = min(usable, key=lambda row: row["t_count"]) if usable else None
    return {
        "num_controls": num_controls,
        "blocks": blocks,
        "seed": seed,
        "interconnect": interconnect,
        "t_per_block": {"exact": t_exact_per_block, "relative": t_relative_per_block},
        "rows": rows,
        "best_usable": best,
        "all_exact_t": rows[0]["t_count"],
        "all_relative_usable": bool(rows[-1]["usable"]),
        "headroom_t": (rows[0]["t_count"] - best["t_count"]) if best else 0,
    }


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy-B headroom (MCX variant downgrading).")
    parser.add_argument("--num-controls", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--interconnect",
        choices=("clifford", "h_free"),
        default="clifford",
        help="'clifford' includes Hadamards after each block; 'h_free' keeps the "
        "correction diagonal (the regime where relative-phase downgrades are safe)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    results = []
    print(f"    interconnect regime: {args.interconnect}", flush=True)
    print(
        f"{'seed':>4} | {'T(per block)':>13} | {'all-exact':>9} | {'best usable':>11} | "
        f"{'downgraded':>10} | {'all-relative usable':>19} | {'headroom T':>10}"
    )
    for seed in range(args.seeds):
        result = measure_instance(
            args.num_controls, args.blocks, seed, interconnect=args.interconnect
        )
        results.append(result)
        per_block = result["t_per_block"]
        best = result["best_usable"]
        print(
            f"{seed:>4} | {per_block['exact']:>6}/{per_block['relative']:<6} | "
            f"{result['all_exact_t']:>9} | {best['t_count'] if best else '-':>11} | "
            f"{best['downgraded'] if best else '-':>10} | "
            f"{str(result['all_relative_usable']):>19} | {result['headroom_t']:>10}"
        )
    ok = [r for r in results if r["headroom_t"] > 0]
    fragmented = [r for r in results if r["headroom_t"] > 0 and not r["all_relative_usable"]]
    print(
        f"\n    [headroom] instances with a positive gain: {len(ok)}/{len(results)}; "
        f"of those, fragmented (all-relative NOT usable, so a real choice is needed): "
        f"{len(fragmented)}/{len(ok) if ok else 0}"
    )
    print(
        "    [reading] positive + fragmented  -> a learned per-block policy has room to beat "
        "a fixed heuristic;\n              positive + all-relative usable -> just always "
        "downgrade (no learning needed);\n              0 -> no headroom, do not train."
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"    [json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
