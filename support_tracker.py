#!/usr/bin/env python3
"""Sound, polynomial-time safety tracker for relative-phase downgrades.

The exact criterion ("are the post-selected Z-basis statistics preserved?") is a
*measurement-subspace* statement: at width ``n`` it needs a ``2**n`` object, which
is why the reduced verifier dies past ~19 qubits.  This module implements the
cheap sound criterion instead:

    a mask is accepted only if the accumulated correction stays **diagonal**.

If the correction ``C = U_RL . U_exact^dagger`` is diagonal then
``U_RL = U_exact . C``, so every Z-basis probability is unchanged - that is a
*proof* of safety.  The tracker therefore never accepts an unsafe mask.  It is
**incomplete**: a diagonal correction is sufficient, not necessary, so it rejects
masks that the exact verifier would accept (measured cost in
``reports/strategy_b_experiments.md`` §5.4 / the run below).

Tracking the correction
-----------------------
Start from ``E = I`` and sweep the circuit forward.  ``E`` is pushed to the
measurement, so each gate conjugates it ``E <- G E G^dagger``:

* a **downgraded block** injects its diagonal correction: ``E <- D . E``, so the
  support grows by that block's qubits;
* **diagonal** gates (Z, S, T, Rz, CZ, CCZ ...) commute with ``E`` -> no change;
* **X / CNOT** are permutations: ``Z_b -> Z_a Z_b``, so a CNOT only adds its
  control to the support when the target is already supported;
* a **whole MCX block** is a permutation, and conjugating the target adds the
  controls *and* the clean ancillas (the AND chain), while conjugating a control
  or ancilla changes nothing.  This is why the tracker must work at **block**
  granularity: sweeping the ladder's internal H/T/CX gates one by one would flag
  a perfectly diagonal block;
* a **Hadamard** on a supported qubit turns the correction non-diagonal
  (``Z -> X``) -> the mask is rejected permanently.

Cost: one pass over the schedule, ``O(gates)`` time and ``O(qubits)`` space - no
matrix of size ``2**n`` anywhere.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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


GEN = _load("variant_generator_tracker", "generate_variant_dataset.py")


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


def homogeneous_schedule(num_controls: int, blocks: int, seed: int, h_prob: float) -> Dict[str, Any]:
    """Block-level schedule of a homogeneous instance (what the tracker consumes)."""
    controls, ancillas, target = GEN.qubit_layout(num_controls)
    total = num_controls + 1 + len(ancillas)
    layers = GEN.build_instance_layers(num_controls, blocks, seed, h_prob)
    layer_gates: List[List[Tuple[str, List[int]]]] = []
    for layer in layers:
        if layer is None:
            layer_gates.append([])
            continue
        layer_gates.append(
            [
                (ins.operation.name, [layer.find_bit(q).index for q in ins.qubits])
                for ins in layer.data
            ]
        )
    block = {"controls": list(controls), "ancillas": list(ancillas), "target": int(target)}
    return {
        "total": total,
        "layers": layer_gates,
        "blocks": [dict(block) for _ in range(blocks)],
    }


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------


# Gate names the tracker's model actually covers, plus the ones that are provably
# harmless to it (diagonal / sign-flip only).  ANYTHING ELSE must be rejected
# explicitly: silently ignoring an unmodelled gate (``measure``, ``reset``,
# ``if_else``, ...) would let the tracker certify a circuit it does not model.
_MODELLED = frozenset({"h", "cx", "swap"})
_HARMLESS = frozenset({"x", "z", "s", "sdg", "t", "tdg", "cz", "rz", "id", "barrier"})


class UnsupportedGateError(ValueError):
    """Raised when a layer contains a gate the tracker does not model.

    This is deliberately fatal rather than silent: a measurement-based or
    classically-controlled circuit is *outside* the tracker's declared domain,
    and returning a verdict for it would be unsound.
    """


class DiagonalTracker:
    """``O(gates)`` time / ``O(qubits)`` space soundness check for one mask."""

    __slots__ = ("support", "broken", "qubits")

    def __init__(self, total: int):
        self.qubits = int(total)
        self.support: Set[int] = set()
        self.broken = False

    # -- gate rules ---------------------------------------------------------

    def apply_layer(self, gates: Sequence[Tuple[str, Sequence[int]]]) -> None:
        """Conjugate the accumulated correction by an inter-block Clifford layer."""
        for name, qubits in gates:
            if name == "h":
                if qubits[0] in self.support:
                    self.broken = True
            elif name == "cx":
                if qubits[1] in self.support:
                    self.support.add(qubits[0])
            elif name == "swap":
                a, b = qubits[0], qubits[1]
                in_a, in_b = a in self.support, b in self.support
                if in_a:
                    self.support.add(b)
                if in_b:
                    self.support.add(a)
            elif name in _HARMLESS:
                # diagonal or sign-flip only: conjugation leaves the correction
                # diagonal and does not move its support
                continue
            else:
                raise UnsupportedGateError(
                    f"support_tracker does not model the gate {name!r}. "
                    "Circuits containing measurements, resets or classical "
                    "control are OUTSIDE the declared domain (no ancilla, no "
                    "measurement, no classical control) and must not be "
                    "certified by this tracker."
                )

    def apply_block(self, block: Dict[str, Any]) -> None:
        """Conjugate by an MCX block (a permutation): only the target spreads."""
        if block["target"] in self.support:
            self.support.update(block["controls"])
            self.support.update(block["ancillas"])

    def inject_correction(self, block: Dict[str, Any]) -> None:
        """Multiply by the block's diagonal correction (support = all its qubits)."""
        self.support.add(block["target"])
        self.support.update(block["controls"])
        self.support.update(block["ancillas"])

    # -- driver -------------------------------------------------------------

    def is_safe(self) -> bool:
        return not self.broken


def replay(schedule: Dict[str, Any], mask: int) -> "DiagonalTracker":
    """Replay a schedule through the tracker, **including the tail layer**.

    ``layers`` is either ``blocks`` long (one layer before each block, no gates
    after the last block) or ``blocks + 1`` long, in which case ``layers[-1]``
    holds the gates *after* the last block.  Omitting that tail was a real
    soundness bug: ``CCX; H(target)`` with the CCX downgraded was accepted while
    its residual had off-diagonal weight 1.0 (see ``test_tracker_soundness``).
    """
    layers = schedule["layers"]
    blocks = schedule["blocks"]
    if len(layers) not in (len(blocks), len(blocks) + 1):
        raise ValueError(
            f"schedule has {len(layers)} layers for {len(blocks)} blocks")
    tracker = DiagonalTracker(schedule["total"])
    for index, block in enumerate(blocks):
        tracker.apply_layer(layers[index])
        tracker.apply_block(block)
        if (mask >> index) & 1:
            tracker.inject_correction(block)
    if len(layers) == len(blocks) + 1:
        tracker.apply_layer(layers[-1])
    return tracker


def _cancel_hh(layer: Sequence[Tuple[str, Sequence[int]]]):
    """Drop H,H pairs that are adjacent **on the same qubit** (H H = I).

    Only exact cancellations: a gate touching the qubit between the two H's
    flushes the pending one, so nothing is cancelled across a CX or a block.
    """
    pending: Dict[int, int] = {}
    out: List[Any] = []
    for name, qubits in layer:
        if name == "h":
            q = int(qubits[0])
            if q in pending:
                out[pending.pop(q)] = None          # remove the first H
                continue                            # and drop this one
            pending[q] = len(out)
            out.append((name, qubits))
        else:
            for q in qubits:
                pending.pop(int(q), None)
            out.append((name, qubits))
    return [gate for gate in out if gate is not None]


def peephole(schedule: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``schedule`` with exact H-H cancellations applied per layer."""
    return {**schedule, "layers": [_cancel_hh(layer) for layer in schedule["layers"]]}


def track_safe(schedule: Dict[str, Any], mask: int,
               cancel_hh: bool = False) -> bool:
    """True only when the accumulated correction provably stays diagonal."""
    return replay(peephole(schedule) if cancel_hh else schedule, mask).is_safe()


CERTIFICATE_SCHEMA = "diagonal-certificate-v1"


def certify(schedule: Dict[str, Any], mask: int,
            cancel_hh: bool = False) -> Dict[str, Any]:
    """Replay a mask and return a **machine-checkable witness**.

    The theorem is one line: conjugating a diagonal operator by a basis
    permutation keeps it diagonal, and diagonal gates commute with it, so the
    accumulated correction stays diagonal **as long as no Hadamard ever acts on
    a qubit in its support**.  The tracker only ever returns a boolean; this
    function returns the evidence for that precondition:

        final_support        S, the qubits the correction can touch
        h_checks             for every H: its stage, qubit, and the support
                             at that moment
        violations           the H's that hit the support (empty <=> accepted)

    A reader can check ``S ∩ {H qubits} = ∅`` directly from the record, and
    ``verify_certificate`` re-derives it independently.  The certificate
    witnesses the *precondition*; the diagonal-conjugation lemma is the proof
    that the precondition suffices.
    """
    if cancel_hh:
        schedule = peephole(schedule)
    layers, blocks = schedule["layers"], schedule["blocks"]
    n_blocks = len(blocks)
    if len(layers) not in (n_blocks, n_blocks + 1):
        raise ValueError(
            f"schedule has {len(layers)} layers for {n_blocks} blocks")
    tracker = DiagonalTracker(schedule["total"])
    checks: List[Dict[str, Any]] = []

    def run_layer(index: int, gates: Sequence[Tuple[str, Sequence[int]]]) -> None:
        for name, qubits in gates:
            if name == "h":
                checks.append({
                    "stage": f"layer {index}", "qubit": int(qubits[0]),
                    "in_support": int(qubits[0]) in tracker.support,
                    "support_before": sorted(tracker.support),
                })
            tracker.apply_layer([(name, qubits)])

    for index, block in enumerate(blocks):
        run_layer(index, layers[index])
        tracker.apply_block(block)
        if (mask >> index) & 1:
            tracker.inject_correction(block)
    if len(layers) == n_blocks + 1:
        run_layer(n_blocks, layers[-1])

    violations = [{"stage": c["stage"], "qubit": c["qubit"]}
                  for c in checks if c["in_support"]]
    return {
        "schema": CERTIFICATE_SCHEMA,
        "mask": int(mask),
        "blocks": n_blocks,
        "accepted": not tracker.broken,
        "final_support": sorted(tracker.support),
        "h_checks": checks,
        "violations": violations,
    }


def verify_certificate(schedule: Dict[str, Any], mask: int,
                       certificate: Dict[str, Any]) -> bool:
    """Independently re-derive the witness and compare every claim."""
    fresh = certify(schedule, mask)
    if certificate.get("schema") != CERTIFICATE_SCHEMA:
        return False
    if certificate.get("mask") != fresh["mask"]:
        return False
    if certificate.get("accepted") != fresh["accepted"]:
        return False
    if sorted(certificate.get("final_support", [])) != fresh["final_support"]:
        return False
    recorded = [(c["stage"], c["qubit"], c["in_support"])
                for c in certificate.get("h_checks", [])]
    expected = [(c["stage"], c["qubit"], c["in_support"]) for c in fresh["h_checks"]]
    return recorded == expected


def tracker_greedy(schedule: Dict[str, Any], probs: Sequence[float],
                   threshold: float = 0.0) -> int:
    """Greedy decode with the tracker standing in for the exact verifier."""
    order = sorted(
        (i for i in range(len(schedule["blocks"])) if probs[i] >= threshold),
        key=lambda i: -probs[i],
    )
    accepted = 0
    for block in order:
        candidate = accepted | (1 << block)
        if track_safe(schedule, candidate):
            accepted = candidate
    return accepted


def mask_from_probs(probs: Sequence[float], threshold: float = 0.5) -> int:
    mask = 0
    for index, prob in enumerate(probs):
        if prob >= threshold:
            mask |= 1 << index
    return mask


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _self_test() -> int:
    """Soundness: the tracker must never accept a mask the exact verifier rejects."""
    import numpy as np

    SYM = _load("variant_symbolic_tracker", "symbolic_verifier.py")
    cases = [(2, 4, 11), (3, 4, 22), (4, 3, 33), (3, 5, 44), (2, 6, 55), (4, 4, 66)]
    total_masks = accepted = wrong = 0
    for num_controls, blocks, seed in cases:
        schedule = homogeneous_schedule(num_controls, blocks, seed, 0.7)
        verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks, seed, 0.7)
        case_bad = 0
        for mask in range(1 << blocks):
            exact = verifier.is_safe(mask)
            tracked = track_safe(schedule, mask)
            total_masks += 1
            accepted += int(tracked)
            if tracked and not exact:
                wrong += 1
                case_bad += 1
        print(f"  [{'ok  ' if case_bad == 0 else 'FAIL'}] C^{num_controls}X x{blocks} "
              f"(seed {seed}): {1 << blocks} masks, tracker-accepted, exact-rejected={case_bad}")
    print()
    if wrong:
        print(f"tracker self-test FAILED: {wrong}/{total_masks} unsound acceptances")
        return 1
    print(f"tracker self-test passed (sound on {total_masks} masks; "
          f"accepted {accepted} = {accepted / total_masks:.0%})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sound support/diagonality tracker.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
