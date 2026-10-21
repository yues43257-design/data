"""Ground-truth cases for the compiler correctness/safety benchmark.

Each case has an explicit ``expected_legal`` (is the emitted circuit legal w.r.t.
the intended one?) and ``expected_cert`` (must the certificate be accepted?),
plus a ``ground_truth`` string naming the bug location.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import support_tracker as ST          # noqa: E402
import experiments.multipass_attestation.attacks as PA   # noqa: E402
import experiments.multipass_attestation.pipeline as PP  # noqa: E402
import experiments.security_probe.attacks as SA          # noqa: E402


def _base_with_h() -> Dict[str, Any]:
    """Same two-block shape, but with an H on a supported qubit in the layer."""
    ideal = QuantumCircuit(5)
    ideal.ccx(0, 1, 2)
    ideal.cx(0, 3)
    ideal.h(2)                     # H on the target of block 0 -> illegal
    ideal.ccx(3, 4, 2)
    emitted = QuantumCircuit(5)
    ranges = []
    SA._rccx(emitted, 0, 1, 2)
    ranges.append((0, len(emitted.data)))
    emitted.cx(0, 3)
    emitted.h(2)
    start = len(emitted.data)
    SA._rccx(emitted, 3, 4, 2)
    ranges.append((start, len(emitted.data)))
    schedule = RB._tracker_schedule(ideal)
    mask = (1 << len(schedule["blocks"])) - 1
    certificate = ST.certify(schedule, mask, circuit=emitted, block_ranges=ranges)
    return {"ideal": ideal, "circuit": emitted, "ranges": ranges,
            "schedule": schedule, "mask": mask, "certificate": certificate}


def all_cases() -> List[Dict[str, Any]]:
    base = SA.build_base()
    h_base = _base_with_h()
    pass2 = PP.build_pipeline(0)
    cases: List[Dict[str, Any]] = [
        {"name": "legal_control", "expected_legal": True, "expected_cert": True,
         "ground_truth": "none", "kind": "rp", **base},
        {"name": "illegal_rtof_mask", "expected_legal": False, "expected_cert": False,
         "ground_truth": "layer 1: H(2) hits the support of block 0",
         "kind": "rp", **h_base},
    ]
    for name, truth in (
            ("block_substitution", "block 0: remapped [0,1,2] -> [0,2,1]"),
            ("nonblock_x", "layer 1: X(3) inserted"),
            ("tail_h", "after block 1: H(2)"),
            ("gate_smuggle_u3", "layer 1: u3 inserted"),
            ("gate_smuggle_measure", "after block 1: measure inserted"),
            ("cert_forge", "certificate.accepted/final_support tampered"),
            ("cert_replay", "certificate from mask=1 replayed for mask=3")):
        entry = SA.attack(name, base)
        cases.append({"name": name, "expected_legal": False,
                      "expected_cert": False, "ground_truth": truth,
                      "kind": "rp", "ideal": base["ideal"],
                      **{k: entry[k] for k in
                         ("circuit", "ranges", "schedule", "mask",
                          "certificate")}})
    # the two certificate-only cases are legal circuits with a bad certificate
    for case in cases:
        if case["name"] in ("cert_forge", "cert_replay"):
            case["expected_legal"] = True
    for name, attack_name, truth in (
            ("pass2_substitution", "a2_pass2_substitution",
             "C2: X(0) added after full_reduce"),
            ("pass2_forgery", "a3_compromised_pass2",
             "C2: X(0) + forged cert2 with correct hashes")):
        entry = PA.attack(attack_name, pass2)
        cases.append({"name": name, "expected_legal": None,
                      "expected_cert": False, "ground_truth": truth,
                      "kind": "pass2", "entry": entry})
    return cases
