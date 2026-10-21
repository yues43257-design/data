"""Attacks against the two-pass certificate chain."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import support_tracker as ST      # noqa: E402
import experiments.multipass_attestation.pipeline as P  # noqa: E402

ATTACKS = ("normal", "a1_pass1_block_substitution", "a2_pass2_substitution",
           "a3_compromised_pass2", "a4_replay")


def attack(name: str, base: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "attack": name,
        "c0": base["c0"], "c1": base["c1"], "c2": base["c2"],
        "schedule": base["schedule"], "mask": base["mask"],
        "ranges": base["ranges"],
        "cert1": base["cert1"], "cert2": base["cert2"],
        "intended": base["c2"], "check_circuit": base["c2"],
        "times": base["times"], "note": "",
    }
    if name == "normal":
        entry["note"] = "untampered two-pass pipeline"
    elif name == "a1_pass1_block_substitution":
        c1, ranges = P._emit_c1(0)
        # re-map block 0's controls/target inside the emitted C1
        attacked = QuantumCircuit(5)
        P._rccx(attacked, 0, 2, 1)
        attacked.cx(0, 3)
        P._rccx(attacked, 3, 4, 2)
        entry.update(c1=attacked, ranges=ranges,
                     check_circuit=attacked, intended=base["c1"],
                     note="pass1 re-mapped block 0; cert1 unchanged")
    elif name == "a2_pass2_substitution":
        c2 = base["c2"].copy()
        c2.x(0)                        # a gate substitution in C2
        entry.update(c2=c2, check_circuit=c2, note="pass2 emitted C2 with X(0)")
    elif name == "a3_compromised_pass2":
        c2 = base["c2"].copy()
        c2.x(0)
        forged = {"pass": "pyzx_full_reduce", "claim": "exact_equivalent",
                  "input_hash": ST.circuit_hash(base["c1"]),
                  "output_hash": ST.circuit_hash(c2)}
        entry.update(c2=c2, cert2=forged, check_circuit=c2,
                     note="pass2 compromised: forged cert2 with CORRECT hashes")
    elif name == "a4_replay":
        other = P.build_pipeline(variant=1)
        entry.update(cert1=other["cert1"], cert2=other["cert2"],
                     note="certificates replayed from pipeline variant 1")
    else:
        raise ValueError(name)
    return entry
