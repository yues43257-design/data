"""Generated benchmark suite with separate L / S / C ground truths.

  L legality   -- is the emitted circuit legal (residual diagonal)?
  S semantics  -- is it exactly equal to the intended circuit (global phase)?
  C certificate-- split into named properties:
                   positive  a genuine certificate must be accepted
                   binding   a changed circuit must be rejected
                   tamper    a forged/replayed certificate must be rejected
                   chain     a 2-pass chain must reject a wrong C2 (unless the
                             pass writes correct hashes)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC    # noqa: E402
import experiments.compiler_bench.generator as G  # noqa: E402
import experiments.multipass_attestation.attacks as PA   # noqa: E402
import experiments.multipass_attestation.pipeline as PP  # noqa: E402

FAMILIES = (("two_block", 0), ("adder", 2), ("adder", 3), ("vbe", 2),
            ("vbe", 3), ("comparator", 3))
LAYOUTS = ("plain", "with_h")
SEEDS = (0, 1)
DENSE_LIMIT = 8          # keep the suite runtime bounded

BINDING_ATTACKS = {"block_substitution", "nonblock_x", "tail_h",
                   "gate_smuggle_u3", "gate_smuggle_measure"}
TAMPER_ATTACKS = {"cert_forge", "cert_replay"}


def _unitary(entry: Dict[str, Any]) -> bool:
    circuit = entry["circuit"]
    if circuit.num_clbits or entry["ideal"].num_qubits > DENSE_LIMIT:
        return False
    return not any(ins.operation.name in ("measure", "reset", "if_else")
                   for ins in circuit.data)


def dense_legal(entry: Dict[str, Any]) -> Optional[bool]:
    if not _unitary(entry):
        return None
    try:
        return bool(RPC.relative_phase_ok(entry["ideal"], entry["circuit"]))
    except Exception:
        return None


def dense_exact(entry: Dict[str, Any]) -> Optional[bool]:
    if not _unitary(entry):
        return None
    try:
        a = np.asarray(Operator(entry["ideal"]).data, dtype=complex)
        b = np.asarray(Operator(entry["circuit"]).data, dtype=complex)
        m = a @ b.conj().T
        off = float(np.abs(m - np.diag(np.diag(m))).max())
        diagonal = np.diag(m)
        return bool(off < 1e-9 and float(np.abs(diagonal - diagonal[0]).max()) < 1e-9)
    except Exception:
        return None


def cert_property(attack: str, truth_l: Optional[bool]) -> str:
    if attack == "legal":
        if truth_l is True:
            return "positive"
        if truth_l is False:
            return "illegal_rewrite"
        return "undecided_legal"
    if attack in BINDING_ATTACKS:
        return "binding"
    if attack in TAMPER_ATTACKS:
        return "tamper"
    return "chain"


def all_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for family, n in FAMILIES:
        for layout in LAYOUTS:
            for seed in SEEDS:
                try:
                    base = G.build_base(family, n, layout, seed)
                except Exception as exc:
                    cases.append({"name": f"{family}/{n}/{layout}/{seed}",
                                  "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if base["ideal"].num_qubits > 11:
                    continue
                for attack in G.ATTACKS:
                    entry = G.apply_attack(base, attack)
                    truth_l = dense_legal(entry)
                    truth_s = dense_exact(entry)
                    cases.append({
                        "name": f"{family}{n or ''}/{layout}/s{seed}/{attack}",
                        "family": family, "n": n, "layout": layout,
                        "seed": seed, "attack": attack,
                        "qubits": base["ideal"].num_qubits,
                        "cert_property": cert_property(attack, truth_l),
                        "ground_truth": _location(attack),
                        "truth_L": truth_l, "truth_S": truth_s,
                        "entry": entry,
                    })
    pass2 = PP.build_pipeline(0)
    for name, attack_name in (("pass2_substitution", "a2_pass2_substitution"),
                              ("pass2_forgery", "a3_compromised_pass2")):
        entry = PA.attack(attack_name, pass2)
        entry["kind"] = "pass2"
        cases.append({"name": f"pass2/{name}", "family": "pass2", "n": 0,
                      "layout": "-", "seed": 0, "attack": name,
                      "qubits": entry["c1"].num_qubits if "c1" in entry else 5,
                      "cert_property": "chain",
                      "ground_truth": "C2 after full_reduce is tampered",
                      "truth_L": None, "truth_S": None,
                      "kind": "pass2", "entry": entry})
    return cases


def _location(attack: str) -> str:
    return {
        "legal": "none",
        "block_substitution": "block 0 remapped (same schedule metadata)",
        "nonblock_x": "outside the blocks: X inserted",
        "tail_h": "after the last block: H",
        "gate_smuggle_u3": "outside the blocks: u3",
        "gate_smuggle_measure": "after the last block: measure",
        "cert_forge": "certificate.accepted flipped",
        "cert_replay": "certificate replayed from a different mask",
        "pass2_substitution": "C2 tampered after full_reduce",
        "pass2_forgery": "C2 tampered + forged cert2 with correct hashes",
    }.get(attack, "unknown")
