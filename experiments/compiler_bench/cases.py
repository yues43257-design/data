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
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC    # noqa: E402
import support_tracker as ST              # noqa: E402
import experiments.compiler_bench.generator as G  # noqa: E402
import experiments.multipass_attestation.attacks as PA   # noqa: E402
import experiments.multipass_attestation.pipeline as PP  # noqa: E402

FAMILIES = (("two_block", 0), ("adder", 2), ("adder", 3), ("vbe", 2),
            ("vbe", 3), ("comparator", 3))
LAYOUTS = ("plain", "with_h")
SEEDS = (0, 1, 2, 3)
DENSE_LIMIT = 8          # keep the suite runtime bounded
SUITE_ATTACKS = ("legal", "h_on_support", "block_substitution", "tail_h",
                 "gate_smuggle_u3", "cert_forge", "cert_replay")

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
    if attack in ("legal", "h_on_support"):
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


@lru_cache(maxsize=1)
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
                for attack in SUITE_ATTACKS:
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
    n_qubits = pass2["c2"].num_qubits
    for i in range(5):                       # detected: cert unchanged
        c2 = pass2["c2"].copy()
        c2.x(i % n_qubits)
        entry = dict(pass2)
        entry.update(c2=c2, check_circuit=c2, intended=pass2["c2"], kind="pass2")
        cases.append({"name": f"pass2/substitution{i}", "family": "pass2", "n": 0,
                      "layout": "-", "seed": i, "attack": "pass2_substitution",
                      "qubits": n_qubits, "cert_property": "chain",
                      "ground_truth": "C2 tampered, cert unchanged",
                      "truth_L": None, "truth_S": None, "entry": entry})
    for i in range(20):                      # the structural gap: correct hashes
        c2 = pass2["c2"].copy()
        c2.x(i % n_qubits)
        entry = dict(pass2)
        entry.update(c2=c2, check_circuit=c2, intended=pass2["c2"], kind="pass2",
                     cert2={"pass": "pyzx_full_reduce", "claim": "exact_equivalent",
                            "input_hash": ST.circuit_hash(pass2["c1"]),
                            "output_hash": ST.circuit_hash(c2)})
        cases.append({"name": f"pass2/forgery{i}", "family": "pass2", "n": 0,
                      "layout": "-", "seed": i, "attack": "pass2_forgery",
                      "qubits": n_qubits, "cert_property": "chain",
                      "ground_truth": "C2 tampered + forged cert2 (correct hashes)",
                      "truth_L": None, "truth_S": None, "entry": entry})
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
