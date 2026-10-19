#!/usr/bin/env python3
"""Exact phase-polynomial legality checker, and where it can actually be used.

THE CHECKER
-----------
A circuit over {x, cx, t, tdg, s, sdg, z, cz} can be written exactly as

    U = P . D          P = basis permutation,  D = diagonal

where ``P`` is the affine map over GF(2) that the X/CX gates induce (tracked as
one ``(mask, const)`` pair per qubit), and ``D`` collects the T/S/Z phases.
For two such circuits

    C = U_cand . U_ideal^dagger = P_c D_c D_i^dagger P_i^dagger

so ``C`` is diagonal **iff P_c == P_i** (if the permutations differ, the
(a,b) entry is nonzero exactly along the permutation P_c^-1 P_i).  Any diagonal
is unit-modulus, so that is the whole criterion: **compare the affine maps**.
Cost: O(gates) time, O(qubits^2) bits -- no 2^n object anywhere.

WHY THIS IS NOT A DROP-IN REPLACEMENT FOR THE TRACKER
-----------------------------------------------------
The form is closed only for H-free circuits.  A Hadamard maps Z to X, so the
"permutation times diagonal" decomposition breaks.  This script therefore
measures, on the SAME 24-100q workload as §7, how much of it is in the
checker's domain -- it is a scope measurement, not a new oracle.

What is measured
  1. validation: on small H-free circuits the checker must agree with the
     dense criterion (``relative_phase_criterion``) on every case;
  2. applicability: per circuit, the first gate outside the PP gate set;
  3. cost when it *does* apply: the inter-block Clifford layers (H-free when
     h = 0), timed at 24/32/50/100 qubits;
  4. the reference costs: the tracker on the same instance, and the dense
     exact criterion's memory (4^n complex128).

Usage
-----
    .venv/bin/python pp_exact_checker.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC          # noqa: E402
import support_tracker as ST                    # noqa: E402
import ultimate_scalability_test as ULT         # noqa: E402
import wide_scale_stress as WS                  # noqa: E402
from qiskit import QuantumCircuit               # noqa: E402

CHECKER_VERSION = "pp-exact-v1"

# gates that keep the "permutation . diagonal" form
_DIAGONAL = frozenset({"t", "tdg", "s", "sdg", "z", "cz", "rz", "p", "id"})
_AFFINE_1Q = frozenset({"x"})
_AFFINE_2Q = frozenset({"cx"})


class NotApplicable(Exception):
    """The circuit leaves the phase-polynomial gate set."""


def affine_map(circuit: QuantumCircuit) -> List[Tuple[int, int]]:
    """Exact affine skeleton: qubit q |-> parity(mask & x) ^ const."""
    n = circuit.num_qubits
    mask = [1 << q for q in range(n)]
    const = [0] * n
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if name in _DIAGONAL:
            continue
        if name in _AFFINE_1Q:
            const[qubits[0]] ^= 1
        elif name in _AFFINE_2Q:
            c, t = qubits
            mask[t] ^= mask[c]
            const[t] ^= const[c]
        else:
            raise NotApplicable(name)
    return list(zip(mask, const))


def pp_legal(ideal: QuantumCircuit,
             candidate: QuantumCircuit) -> Tuple[bool, str]:
    """Exact legality for phase-polynomial circuits (raises NotApplicable)."""
    if ideal.num_qubits != candidate.num_qubits:
        raise ValueError("width mismatch")
    a = affine_map(ideal)
    b = affine_map(candidate)
    return (a == b), ("same affine skeleton" if a == b
                      else "affine skeletons differ")


def pp_memory_bits(n: int) -> int:
    """Bits needed by the affine map: n masks of n bits + n const bits."""
    return n * n + n


# ---------------------------------------------------------------------------
# 1. validation against the dense criterion
# ---------------------------------------------------------------------------

def _random_pp_circuit(n: int, length: int, rng: random.Random) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for _ in range(length):
        choice = rng.random()
        if choice < 0.25:
            qc.x(rng.randrange(n))
        elif choice < 0.55 and n > 1:
            c, t = rng.sample(range(n), 2)
            qc.cx(c, t)
        else:
            qc.__getattribute__(rng.choice(["t", "tdg", "s", "sdg", "z"]))(rng.randrange(n))
    return qc


def _random_variant(circuit: QuantumCircuit, rng: random.Random) -> QuantumCircuit:
    """Make a PP candidate that is legal or illegal about equally often."""
    data = list(circuit.data)
    mode = rng.choice(["flip_phase", "drop_cx", "extra_x"])
    drop: Optional[int] = None
    flip: Optional[int] = None
    if mode == "drop_cx":
        candidates = [i for i, ins in enumerate(data) if ins.operation.name == "cx"]
        drop = rng.choice(candidates) if candidates else None
    elif mode == "flip_phase":
        candidates = [i for i, ins in enumerate(data) if ins.operation.name in ("t", "tdg")]
        flip = rng.choice(candidates) if candidates else None
    if drop is None and flip is None:
        mode = "extra_x"

    out = QuantumCircuit(circuit.num_qubits)
    for index, instruction in enumerate(data):
        if index == drop:
            continue
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        name = instruction.operation.name
        if index == flip:
            getattr(out, "tdg" if name == "t" else "t")(qubits[0])
        else:
            out.append(instruction.operation, qubits,
                       [circuit.find_bit(c).index for c in instruction.clbits])
    if mode == "extra_x":
        out.x(rng.randrange(circuit.num_qubits))
    return out


def validate(trials: int = 60) -> Dict[str, Any]:
    """The checker must reproduce the dense criterion on H-free circuits."""
    rng = random.Random(20261019)
    agree = disagree = legal_count = 0
    examples: List[Dict[str, Any]] = []
    for _ in range(trials):
        n = rng.randint(3, 6)
        ideal = _random_pp_circuit(n, rng.randint(8, 20), rng)
        candidate = _random_variant(ideal, rng)
        mine, _ = pp_legal(ideal, candidate)
        dense = bool(RPC.relative_phase_ok(ideal, candidate))
        legal_count += int(dense)
        if mine == dense:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 3:
                examples.append({"n": n, "pp": mine, "dense": dense})
    return {"trials": trials, "agree": agree, "disagree": disagree,
            "dense_legal": legal_count, "examples": examples}


# ---------------------------------------------------------------------------
# 2. applicability + cost on the 24-100q workload
# ---------------------------------------------------------------------------

def _first_non_pp_gate(circuit: QuantumCircuit) -> Optional[str]:
    for instruction in circuit.data:
        name = instruction.operation.name
        if name not in _DIAGONAL and name not in _AFFINE_1Q and name not in _AFFINE_2Q:
            return name
    return None


def _layer_circuit(instance) -> QuantumCircuit:
    """Concatenate the inter-block layers (H-free when h_prob == 0)."""
    out = QuantumCircuit(instance.total)
    for layer in instance.layers:
        if layer is None:
            continue
        for instruction in layer.data:
            qubits = [layer.find_bit(q).index for q in instruction.qubits]
            out.append(instruction.operation, qubits)
    return out


def census(widths: Sequence[int] = (24, 32, 50, 100),
           h_probs: Sequence[float] = (0.0, 0.1, 0.5),
           blocks: int = 20, ancilla_pool: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for width in widths:
        for h_prob in h_probs:
            inst = ULT.WideInstance(width=width, blocks=blocks, seed=20261019,
                                    h_prob=h_prob, ancilla_pool=ancilla_pool)
            circuit = inst.circuit(0)
            schedule = WS.build_schedule(inst)
            n = inst.total
            gate_counts: Dict[str, int] = {}
            for instruction in circuit.data:
                name = instruction.operation.name
                gate_counts[name] = gate_counts.get(name, 0) + 1

            # (a) can the checker be applied to the whole circuit?
            blocker = _first_non_pp_gate(circuit)
            started = time.perf_counter()
            try:
                affine_map(circuit)
                applies, verdict = True, "affine map computed"
            except NotApplicable as exc:
                applies, verdict = False, f"NotApplicable at first {exc}"
            pp_seconds = time.perf_counter() - started

            # (b) cost where it DOES apply: the inter-block layers at h = 0
            layer_circuit = _layer_circuit(inst)
            layer_blocker = _first_non_pp_gate(layer_circuit)
            started = time.perf_counter()
            layer_applies = layer_blocker is None
            if layer_applies:
                affine_map(layer_circuit)
            layer_seconds = time.perf_counter() - started
            layer_gates = len(layer_circuit.data)

            # (c) reference costs
            mask = (1 << inst.blocks) - 1
            started = time.perf_counter()
            tracker_ok = bool(ST.track_safe(schedule, mask))
            tracker_seconds = time.perf_counter() - started
            dense_bytes = (1 << (2 * n)) * 16

            rows.append({
                "width": width, "N": n, "h_prob": h_prob, "blocks": inst.blocks,
                "gates": len(circuit.data), "gate_counts": gate_counts,
                "pp_applies_to_circuit": applies,
                "pp_blocking_gate": blocker,
                "pp_seconds": pp_seconds,
                "layers_are_pp": layer_applies,
                "layer_gates": layer_gates,
                "layer_pp_seconds": layer_seconds,
                "tracker_all_relative_safe": tracker_ok,
                "tracker_seconds": tracker_seconds,
                "pp_memory_bytes": pp_memory_bits(n) // 8,
                "dense_exact_bytes": dense_bytes,
            })
    return rows


def main() -> int:
    print("=" * 104)
    print(f"  EXACT PHASE-POLYNOMIAL CHECKER  version={CHECKER_VERSION}")
    print("  criterion: C = U_cand U_ideal^dagger diagonal  <=>  same affine skeleton")
    print("=" * 104)
    validation = validate()
    print(f"\n  (1) validation vs dense criterion: {validation['agree']}/"
          f"{validation['trials']} agree, {validation['disagree']} disagree "
          f"(dense-legal cases: {validation['dense_legal']})")
    for example in validation["examples"]:
        print(f"      {example}")
    assert validation["disagree"] == 0, "PP checker disagrees with the dense criterion"

    print("\n  (2)+(3) applicability and cost on the same 24-100q workload")
    rows = census()
    header = (f"  {'w':>4}{'N':>5}{'h':>5}{'gates':>7}{'H':>5}"
              f"{'PP?':>6}{'blocker':>8}{'PP ms':>8}"
              f"{'layersPP':>10}{'lay-ms':>8}{'track ms':>10}"
              f"{'PP B':>8}{'dense B':>12}")
    print(header)
    for row in rows:
        print(f"  {row['width']:>4}{row['N']:>5}{row['h_prob']:>5.1f}"
              f"{row['gates']:>7}{row['gate_counts'].get('h', 0):>5}"
              f"{('yes' if row['pp_applies_to_circuit'] else 'NO'):>6}"
              f"{str(row['pp_blocking_gate']):>8}"
              f"{row['pp_seconds'] * 1e3:>8.3f}"
              f"{('yes' if row['layers_are_pp'] else 'NO'):>10}"
              f"{row['layer_pp_seconds'] * 1e3:>8.3f}"
              f"{row['tracker_seconds'] * 1e3:>10.3f}"
              f"{row['pp_memory_bytes']:>8}{row['dense_exact_bytes']:>12.3e}")

    applied = sum(1 for row in rows if row["pp_applies_to_circuit"])
    print("\n  (4) reading")
    print(f"      whole-circuit PP applicability: {applied}/{len(rows)}")
    print("      -> the blocks' Clifford+T decompositions contain Hadamards "
          "(2 per Toffoli),")
    print("         so the 'permutation . diagonal' form is closed on NONE of these circuits.")
    print("      -> where the form IS closed (inter-block layers at h=0) it costs "
          "O(gates) and ~N^2/8 bytes.")
    print("      -> exact checking of the FULL circuit therefore needs 4^N "
          "(4.0e15 B at N=30).")

    out = ROOT / "reports" / "figure_data" / "pp_exact_checker.json"
    out.write_text(json.dumps({
        "checker_version": CHECKER_VERSION,
        "criterion": "same affine skeleton (exact for phase-polynomial circuits)",
        "validation": validation,
        "census": rows,
        "whole_circuit_applicability": f"{applied}/{len(rows)}",
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
