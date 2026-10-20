#!/usr/bin/env python3
"""Task 3: random tail-H fuzz + end-to-end differential against the dense oracle.

The tail bug survived every earlier test because **no generator ever put a
Hadamard after the last block**.  This harness exists to make that class
impossible to miss again: every random circuit ends with a tail layer that
contains Hadamards on random qubits.

For each circuit we enumerate all masks and compare

    tracker (new, tail applied)   vs   dense ``relative_phase_ok``

and also run the OLD (tail-dropped) replay as a **negative control**: if the
fuzz never produces an old-unsound acceptance, it is not exercising the bug.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
import realcircuit_window as RCW          # noqa: E402
import relative_phase_criterion as RPC    # noqa: E402
import support_tracker as ST              # noqa: E402

FUZZ_VERSION = "tail-fuzz-v1"


def replace_positions(circuit: QuantumCircuit,
                      positions: Sequence[int]) -> QuantumCircuit:
    wanted = set(positions)
    out = QuantumCircuit(circuit.num_qubits)
    for index, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if index in wanted and instruction.operation.name == "ccx":
            c0, c1, target = qubits
            for gate in RCW.LM._rccx_gates(c0, c1, target):
                if gate[0] == "cx":
                    out.cx(gate[1], gate[2])
                else:
                    getattr(out, gate[0])(gate[1])
        else:
            out.append(instruction.operation, qubits,
                       [circuit.find_bit(c).index for c in instruction.clbits])
    return out


def random_circuit(rng: random.Random, blocks: int) -> QuantumCircuit:
    n = rng.randint(3, 7)
    circuit = QuantumCircuit(n)
    for _ in range(blocks):
        layer = QuantumCircuit(n)
        for _ in range(rng.randint(0, 3)):
            if n > 1 and rng.random() < 0.6:
                c, t = rng.sample(range(n), 2)
                layer.cx(c, t)
            else:
                layer.__getattribute__(rng.choice(["h", "x", "z", "s"]))(
                    rng.randrange(n))
        circuit.compose(layer, inplace=True)
        circuit.ccx(*rng.sample(range(n), 3))
    # the tail: Hadamards AFTER the last block (the previously untested class)
    tail = QuantumCircuit(n)
    for _ in range(rng.randint(1, 2)):
        tail.h(rng.randrange(n))
    if n > 1 and rng.random() < 0.5:
        c, t = rng.sample(range(n), 2)
        tail.cx(c, t)
    circuit.compose(tail, inplace=True)
    return circuit


def old_track_safe(schedule: Dict[str, Any], mask: int) -> bool:
    tracker = ST.DiagonalTracker(schedule["total"])
    for index, block in enumerate(schedule["blocks"]):
        tracker.apply_layer(schedule["layers"][index])
        tracker.apply_block(block)
        if (mask >> index) & 1:
            tracker.inject_correction(block)
    return tracker.is_safe()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "tail_fuzz.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    total_masks = accepted = new_unsound = old_unsound = flips = 0
    exercised = 0
    for _ in range(args.circuits):
        blocks = rng.randint(1, 4)
        circuit = random_circuit(rng, blocks)
        schedule = RB._tracker_schedule(circuit)
        positions = [i for i, ins in enumerate(circuit.data)
                     if ins.operation.name == "ccx"]
        tail = schedule["layers"][-1] if len(schedule["layers"]) > blocks else []
        tail_h = sum(1 for name, _ in tail if name == "h")
        case_new = case_old = case_flip = 0
        for mask in range(1 << blocks):
            candidate = replace_positions(
                circuit, [positions[i] for i in range(blocks) if (mask >> i) & 1])
            legal = bool(RPC.relative_phase_ok(circuit, candidate))
            new = bool(ST.track_safe(schedule, mask))
            old = old_track_safe(schedule, mask)
            total_masks += 1
            accepted += int(new)
            case_new += int(new and not legal)
            case_old += int(old and not legal)
            case_flip += int(old != new)
        new_unsound += case_new
        old_unsound += case_old
        flips += case_flip
        exercised += int(case_old > 0)
        rows.append({"qubits": circuit.num_qubits, "blocks": blocks,
                     "tail_h": tail_h, "masks": 1 << blocks,
                     "flips": case_flip, "old_unsound": case_old,
                     "new_unsound": case_new})
    print("=" * 96)
    print(f"  tail fuzz  version={FUZZ_VERSION}  circuits={args.circuits}")
    print(f"  masks={total_masks}  tracker-accepted={accepted}")
    print(f"  OLD (tail dropped) unsound accepts = {old_unsound}  "
          f"in {exercised} circuits   <- negative control")
    print(f"  NEW (tail applied) unsound accepts = {new_unsound}")
    print(f"  old/new verdict flips              = {flips}")
    print(f"  ({time.perf_counter() - started:.1f}s)")
    print("=" * 96)
    ok = new_unsound == 0 and old_unsound > 0
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'} "
          f"({'fuzz exercises the bug and the fix closes it' if ok else 'not discriminative or still unsound'})")
    out = Path(args.out)
    out.write_text(json.dumps({
        "fuzz_version": FUZZ_VERSION, "seed": args.seed,
        "circuits": args.circuits, "masks": total_masks,
        "accepted": accepted, "old_unsound": old_unsound,
        "new_unsound": new_unsound, "flips": flips,
        "circuits_exercising_bug": exercised, "verdict": "PASS" if ok else "FAIL",
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
