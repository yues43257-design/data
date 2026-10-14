#!/usr/bin/env python3
"""Strategy-B gate, v2: is downgrading an MCX block to the relative-phase
variant *safe for the final measurement statistics*?

The v1 gate asked "is the correction diagonal after conjugation by the block?",
which is not the physically relevant question and reported a hard zero.  What
actually matters is whether the **Z-basis measurement statistics** change: a
diagonal (+-1) correction only flips amplitudes' signs, so it never changes
probabilities *at the measurement point*, but the gates in between (Hadamards in
particular) can turn it into an X-type error that does.

This gate therefore simulates the circuit directly: for every computational
basis input (ancillas kept in |0>) it compares the output probability
distribution of a candidate variant assignment against the all-exact circuit.
Assignments whose maximum total-variation distance stays below the tolerance are
*usable*.

The interesting outcome is **fragmented** headroom: a positive T-count saving
where *not all* blocks may be downgraded.  Then the per-block choice is a real
combinatorial problem and a learned policy has room to beat a fixed heuristic.
If instead every block may be downgraded, the saving is unconditional (apply it
and never train).  If nothing is downgradable, there is no headroom.

Usage
-----
    .venv/bin/python variant_choice_gate.py --num-controls 3 --blocks 3 --seeds 5
    .venv/bin/python variant_choice_gate.py --blocks 3 --h-after-blocks always --json /tmp/gate.json
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_linux_mcx():
    loader = importlib.machinery.SourceFileLoader("linux_mcx_gate", str(ROOT / "Linux-MCX"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("cannot build an import spec for Linux-MCX")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator, Statevector  # noqa: E402


def exact_block(num_controls: int) -> QuantumCircuit:
    return qasm2.loads(LINUX_MCX.make_mcx_source(num_controls, basis="clifford_t").to_qasm())


def relative_block(num_controls: int) -> QuantumCircuit:
    return LINUX_MCX.relative_phase_mcx_circuit(num_controls)


def block_t_counts(num_controls: int):
    def count(circuit: QuantumCircuit) -> int:
        source = LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()
        return LINUX_MCX.compute_ftq_metrics(source).t_count

    return count(exact_block(num_controls)), count(relative_block(num_controls))


def interconnect_layer(
    qc: QuantumCircuit,
    qubits: Sequence[int],
    rng: random.Random,
    h_after_blocks: str,
) -> None:
    """Clifford layer between blocks; ``h_after_blocks`` controls Hadamard use."""
    gate_set = ["s", "x", "z", "cx"]
    if h_after_blocks == "always":
        for qubit in qubits:
            qc.h(qubit)
        return
    if h_after_blocks == "random":
        gate_set.append("h")
    for _ in range(rng.randint(1, 2) * len(qubits)):
        choice = rng.choice(gate_set)
        qubit = rng.choice(qubits)
        if choice == "cx":
            other = rng.choice([q for q in qubits if q != qubit])
            qc.cx(qubit, other)
        else:
            getattr(qc, choice)(qubit)


def build_circuit(
    num_controls: int,
    blocks: int,
    mask: int,
    seed: int,
    h_after_blocks: str,
    total: int,
) -> QuantumCircuit:
    data_qubits = list(range(num_controls)) + [total - 1]   # controls + target
    rng = random.Random(seed)
    circuit = QuantumCircuit(total)
    for index in range(blocks):
        if index:
            interconnect_layer(circuit, data_qubits, rng, h_after_blocks)
        block = relative_block(num_controls) if (mask >> index) & 1 else exact_block(num_controls)
        circuit.compose(block, range(total), inplace=True)
    return circuit


def measurement_distributions(
    circuit: QuantumCircuit,
    num_controls: int,
    total: int,
) -> "Dict[int, np.ndarray]":
    """Z-basis output distribution for every data-qubit basis input."""
    n_ancillas = max(0, num_controls - 2)
    # controls 0..n-1, ancillas n..n+a-1, target on the highest qubit
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    matrix = Operator(circuit).data
    keep = (
        np.array([x for x in range(2 ** total) if all(not (x >> a) & 1 for a in ancillas)])
        if ancillas
        else None
    )
    input_indices = keep if keep is not None else np.arange(2 ** total)
    distributions: "Dict[int, np.ndarray]" = {}
    for basis in input_indices:
        amplitudes = matrix[:, int(basis)]
        probabilities = np.abs(amplitudes) ** 2
        if keep is not None:
            probabilities = probabilities[keep]
        probabilities = probabilities / max(probabilities.sum(), 1e-30)
        distributions[int(basis)] = probabilities
    return distributions


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    size = max(len(a), len(b))
    aa = np.zeros(size)
    bb = np.zeros(size)
    aa[: len(a)] = a
    bb[: len(b)] = b
    return float(0.5 * np.sum(np.abs(aa - bb)))


def measure_instance(
    num_controls: int,
    blocks: int,
    seed: int,
    h_after_blocks: str,
    tolerance: float,
) -> "Dict[str, Any]":
    n_ancillas = max(0, num_controls - 2)
    total = num_controls + 1 + n_ancillas
    t_exact, t_relative = block_t_counts(num_controls)
    reference = build_circuit(num_controls, blocks, 0, seed, h_after_blocks, total)
    reference_distributions = measurement_distributions(reference, num_controls, total)

    rows: "List[Dict[str, Any]]" = []
    for mask in range(1 << blocks):
        circuit = build_circuit(num_controls, blocks, mask, seed, h_after_blocks, total)
        distributions = measurement_distributions(circuit, num_controls, total)
        deviation = max(
            total_variation(reference_distributions[basis], distributions[basis])
            for basis in reference_distributions
        )
        downgraded = bin(mask).count("1")
        rows.append(
            {
                "mask": mask,
                "downgraded": downgraded,
                "t_count": (blocks - downgraded) * t_exact + downgraded * t_relative,
                "tv_distance": deviation,
                "usable": deviation <= tolerance,
            }
        )
    usable = [row for row in rows if row["usable"]]
    best = min(usable, key=lambda row: row["t_count"]) if usable else None
    all_relative = rows[-1]
    return {
        "num_controls": num_controls,
        "blocks": blocks,
        "seed": seed,
        "h_after_blocks": h_after_blocks,
        "t_per_block": {"exact": t_exact, "relative": t_relative},
        "all_exact_t": rows[0]["t_count"],
        "best_usable": best,
        "all_relative_usable": bool(all_relative["usable"]),
        "headroom_t": (rows[0]["t_count"] - best["t_count"]) if best else 0,
        "fragmented": bool(best and 0 < best["downgraded"] < blocks),
        "rows": rows,
    }


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy-B gate v2 (measurement statistics).")
    parser.add_argument("--num-controls", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--h-after-blocks",
        choices=("none", "random", "always"),
        default="random",
        help="Hadamards in the Clifford layers between blocks (they are what makes "
        "a diagonal correction matter for the final statistics)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    results = []
    print(f"    interconnect Hadamards: {args.h_after_blocks} | tolerance: {args.tolerance:g}")
    print(
        f"{'seed':>4} | {'T/block':>9} | {'all-exact':>9} | {'best usable':>11} | "
        f"{'downgraded':>10} | {'all-rel usable':>14} | {'fragmented':>10} | {'headroom':>8}"
    )
    for seed in range(args.seeds):
        result = measure_instance(
            args.num_controls, args.blocks, seed, args.h_after_blocks, args.tolerance
        )
        results.append(result)
        best = result["best_usable"]
        print(
            f"{seed:>4} | {result['t_per_block']['exact']:>3}/{result['t_per_block']['relative']:<5} | "
            f"{result['all_exact_t']:>9} | {best['t_count'] if best else '-':>11} | "
            f"{best['downgraded'] if best else '-':>10} | "
            f"{str(result['all_relative_usable']):>14} | {str(result['fragmented']):>10} | "
            f"{result['headroom_t']:>8}"
        )
    positive = [r for r in results if r["headroom_t"] > 0]
    fragmented = [r for r in positive if r["fragmented"]]
    unconditional = [r for r in positive if r["all_relative_usable"]]
    print(
        f"\n    [gate] positive headroom: {len(positive)}/{len(results)} | "
        f"fragmented (real choice): {len(fragmented)} | "
        f"unconditional all-relative: {len(unconditional)}"
    )
    if fragmented:
        print("    [verdict] POSITIVE + FRAGMENTED -> a learned per-block policy has room; train here.")
    elif unconditional:
        print("    [verdict] positive but unconditional -> always downgrade; do NOT train.")
    else:
        print("    [verdict] no headroom under this regime -> do not train.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"    [json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
