#!/usr/bin/env python3
"""The three remaining optional items, in one place.

A) Weak-definition robustness: the same Toffoli sites scored under BOTH
   definitions, stratified by H spacing:
     strict = ``C = U_cand . U_ideal^dagger`` diagonal   (declared definition)
     weak   = entrywise ``|U_cand| == |U_ideal|``          (basis-inputs only)
   If the tracker is "near-tight" only under the strict definition, that shows
   up as a large conservativeness gap between the two columns.

B) Why are the two MISS sites legal?  Compute ``C`` and extract the set of
   qubits its phase actually depends on (the TRUE support, after removing a
   global phase).  If that set avoids the downstream H's while the tracker's
   support bound does not, the miss is a *bounding* artefact (fixable); if it
   overlaps the H's, the gap is structural (sufficient vs necessary).

C) Directed adversarial search for the two remaining stress axes:
     * H acting on ANCILLA qubits (the earlier search only had H on data),
     * ancilla reuse (already in the WideInstance layout) combined with above.

Usage
-----
    .venv/bin/python remaining_options.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
import realcircuit_window as RCW          # noqa: E402
import wide_h_window as WH                # noqa: E402
import support_tracker as ST              # noqa: E402
import relative_phase_criterion as RPC    # noqa: E402
import experiment_setup as SETUP          # noqa: E402

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


ULT = _load("ro_ult", "ultimate_scalability_test.py")
GEN = ULT.GEN
WS = _load("ro_ws", "wide_scale_stress.py")


def wilson(a: int, b: int, z: float = 1.96):
    if not b:
        return float("nan"), float("nan"), float("nan")
    c = (a + z * z / 2) / (b + z * z)
    h = z * math.sqrt(c * (1 - c) / (b + z * z))
    return a / b, max(0.0, c - h), min(1.0, c + h)


def weak_ok(ideal: QuantumCircuit, candidate: QuantumCircuit) -> bool:
    a = np.asarray(Operator(ideal).data, dtype=complex)
    b = np.asarray(Operator(candidate).data, dtype=complex)
    return bool(np.allclose(np.abs(a), np.abs(b), atol=1e-9))


def true_support(circuit: QuantumCircuit, position: int) -> List[int]:
    """Qubits the correction's phase actually depends on (global phase removed)."""
    base = np.asarray(Operator(circuit).data, dtype=complex)
    cand = np.asarray(Operator(RCW.replace_one_ccx(circuit, position)).data, dtype=complex)
    c = cand @ base.conj().T
    d = np.diag(c)
    d = d / d[0]
    total = circuit.num_qubits
    out = []
    for q in range(total):
        shifted = d[np.arange(len(d)) ^ (1 << q)]
        if np.any(np.abs(d - shifted) > 1e-9):
            out.append(q)
    return out


def part_a_b() -> Dict[str, Any]:
    print("=" * 100)
    print("  (A) strict vs weak definition, stratified by H spacing")
    print("=" * 100)
    print(f"  {'H spacing':<14}{'sites':>6}{'legal_strict':>13}{'legal_weak':>11}"
          f"{'tracker':>8}{'MISS_strict':>12}{'conserv':>9}{'95% CI':>18}")
    tiers: Dict[int, Dict[str, int]] = {}
    misses: List[Dict[str, Any]] = []
    for n in (3, 4):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        for every in (0, 1, 2, 3, 4, 5, 6):
            for h_qubits in (canon.num_qubits, max(2, canon.num_qubits // 2)):
                if every == 0 and h_qubits != canon.num_qubits:
                    continue
                circuit = (WH.interleave_h_layers(canon, every, h_qubits)
                           if every else canon)
                schedule = RB._tracker_schedule(circuit)
                positions = [i for i, ins in enumerate(circuit.data)
                             if ins.operation.name == "ccx"]
                tier = tiers.setdefault(every, {"sites": 0, "ls": 0, "lw": 0,
                                                "tr": 0, "miss": 0})
                for site, position in enumerate(positions):
                    mask = 1 << site
                    accepted = bool(ST.track_safe(schedule, mask))
                    ls = bool(RPC.relative_phase_ok(
                        circuit, RCW.replace_one_ccx(circuit, position)))
                    lw = weak_ok(circuit, RCW.replace_one_ccx(circuit, position))
                    tier["sites"] += 1
                    tier["ls"] += ls
                    tier["lw"] += lw
                    tier["tr"] += accepted
                    if ls and not accepted:
                        tier["miss"] += 1
                        misses.append({"adder": n, "every": every, "site": site,
                                       "mask": mask,
                                       "true_support": true_support(circuit, position),
                                       "N": circuit.num_qubits})
    for every in sorted(tiers):
        t = tiers[every]
        c, lo, hi = wilson(t["miss"], t["ls"])
        tag = "H-free" if every == 0 else f"every {every} ccx"
        print(f"  {tag:<14}{t['sites']:>6}{t['ls']:>13}{t['lw']:>11}{t['tr']:>8}"
              f"{t['miss']:>12}{c:>9.3f}   [{lo:.3f}, {hi:.3f}]")
    print("=" * 100)
    print("  (B) the MISS sites: what does the correction actually depend on?")
    for m in misses:
        print(f"    adder{m['adder']} H/{m['every']}ccx site {m['site']}: "
              f"true support {m['true_support']} of {m['N']} qubits")
    return {"tiers": tiers, "misses": misses}


def build_wide_with_ancilla_h(width: int, blocks: int, seed: int, h_prob: float,
                              pool: int = 3):
    """WideInstance whose Clifford layers act on ALL qubits, ancillas included."""
    inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed, h_prob=h_prob,
                            ancilla_pool=pool)
    rng = random.Random(seed + 1)
    for index in range(1, inst.blocks):
        layer = QuantumCircuit(inst.total)
        GEN.clifford_layer(layer, list(range(inst.total)), rng, h_prob)
        inst.layers[index] = layer
    return inst


def part_c() -> Dict[str, Any]:
    print("=" * 100)
    print("  (C) directed adversarial search: H on ANCILLAS + ancilla reuse")
    print("=" * 100)
    print(f"  {'N':>4}{'blocks':>7}{'h':>5}{'masks':>7}{'track':>7}{'legal':>7}"
          f"{'COUNTEREX':>11}")
    total = counter = track_total = 0
    examples: List[Dict[str, Any]] = []
    for width in (4, 5):
        for blocks in (3, 4):
            for h_prob in (0.5, 1.0):
                for seed_index in range(2):
                    seed = SETUP.seed_for(seed_index) * 37 + width * 7 + blocks
                    inst = build_wide_with_ancilla_h(width, blocks, seed, h_prob)
                    if inst.total > 11:
                        continue
                    schedule = WS.build_schedule(inst)
                    ancillas = list(range(width, inst.total))
                    base = inst.circuit(0)
                    t = c = m = 0
                    for mask in range(1 << blocks):
                        m += 1
                        if not ST.track_safe(schedule, mask):
                            continue
                        t += 1
                        ok = RPC.relative_phase_ok(base, inst.circuit(mask), ancillas)
                        if not ok:
                            c += 1
                            examples.append({"width": width, "blocks": blocks,
                                             "h": h_prob, "seed": seed, "mask": mask})
                    print(f"  {inst.total:>4}{blocks:>7}{h_prob:>5.1f}{m:>7}{t:>7}"
                          f"{t - c:>7}{c:>11}")
                    total += m
                    track_total += t
                    counter += c
    print("=" * 100)
    print(f"  TOTAL masks={total}  tracker-accepted={track_total}  "
          f"COUNTEREXAMPLES={counter}")
    return {"masks": total, "accepted": track_total, "counterexamples": counter,
            "examples": examples[:20]}


def main() -> int:
    a = {"note": "reuse previous run (already recorded)"}
    c = part_c()
    out = ROOT / "reports" / "figure_data" / "remaining_options.json"
    out.write_text(json.dumps({"setup_version": SETUP.SETUP_VERSION,
                               "part_a_b": a, "part_c": c}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
