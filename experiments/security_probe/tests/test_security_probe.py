"""Acceptance tests for the certificate-binding security probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import experiments.security_probe.attacks as A   # noqa: E402
import experiments.security_probe.probe as P     # noqa: E402


def test_normal_circuit_is_accepted_and_legal():
    base = A.build_base()
    entry = A.attack("normal", base)
    metrics = A.evaluate(base, entry)
    assert metrics["certificate_accepts"] is True
    assert metrics["attack_legal"] is True
    assert metrics["attack_success"] == 0


def test_block_substitution_is_detected_after_binding():
    base = A.build_base()
    metrics = A.evaluate(base, A.attack("block_substitution", base))
    assert metrics["attack_legal"] is False
    assert metrics["unbound_accepts"] is True       # the old API still accepts
    assert metrics["certificate_accepts"] is False  # bound API rejects
    assert metrics["detected"] is True
    assert metrics["attack_success"] == 0


def test_nonblock_x_is_detected_after_binding():
    base = A.build_base()
    metrics = A.evaluate(base, A.attack("nonblock_x", base))
    assert metrics["attack_legal"] is False
    assert metrics["unbound_accepts"] is True
    assert metrics["certificate_accepts"] is False
    assert metrics["detected"] is True
    assert metrics["attack_success"] == 0


def test_tail_h_is_detected_after_binding():
    base = A.build_base()
    metrics = A.evaluate(base, A.attack("tail_h", base))
    assert metrics["attack_legal"] is False
    assert metrics["unbound_accepts"] is True
    assert metrics["certificate_accepts"] is False
    assert metrics["detected"] is True
    assert metrics["attack_success"] == 0


def test_gate_smuggle_is_detected():
    base = A.build_base()
    for name in ("gate_smuggle_u3", "gate_smuggle_measure"):
        metrics = A.evaluate(base, A.attack(name, base))
        assert metrics["detected"] is True, (name, metrics)
        assert metrics["attack_success"] == 0


def test_cert_forge_is_detected():
    base = A.build_base()
    metrics = A.evaluate(base, A.attack("cert_forge", base))
    assert metrics["certificate_accepts"] is False
    assert metrics["detected"] is True


def test_cert_replay_is_detected():
    base = A.build_base()
    metrics = A.evaluate(base, A.attack("cert_replay", base))
    assert metrics["certificate_accepts"] is False
    assert metrics["detected"] is True


def test_probe_writes_all_attacks():
    results = P.run(write=False)
    names = {r["attack"] for r in results}
    for expected in ("normal", "block_substitution", "nonblock_x", "tail_h",
                     "gate_smuggle_u3", "gate_smuggle_measure",
                     "cert_forge", "cert_replay"):
        assert expected in names
    successes = {r["attack"] for r in results if r["attack_success"]}
    assert successes == set(), successes
    assert all(r["detected"] for r in results if r["attack"] != "normal")
