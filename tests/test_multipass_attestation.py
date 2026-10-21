"""Acceptance tests for the two-pass attestation probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.multipass_attestation.attacks as A       # noqa: E402
import experiments.multipass_attestation.pipeline as P      # noqa: E402
import experiments.multipass_attestation.verify_chain as V  # noqa: E402


def test_normal_two_pass_chain_verifies():
    base = P.build_pipeline()
    result = V.verify_chain(base)
    assert result["ok"] is True, result
    assert result["reasons"] == []


def test_a1_pass1_block_substitution_is_detected():
    base = P.build_pipeline()
    entry = A.attack("a1_pass1_block_substitution", base)
    result = V.verify_chain(entry)
    assert result["ok"] is False
    assert result["detected"] is True
    assert result["attack_success"] == 0


def test_a2_pass2_substitution_is_detected():
    base = P.build_pipeline()
    entry = A.attack("a2_pass2_substitution", base)
    result = V.verify_chain(entry)
    assert result["ok"] is False
    assert result["attack_success"] == 0


def test_a3_compromised_pass2_can_forge_the_chain():
    """The expected failure mode: correct hashes, wrong circuit."""
    base = P.build_pipeline()
    entry = A.attack("a3_compromised_pass2", base)
    result = V.verify_chain(entry)
    assert result["ok"] is True, result
    assert result["attack_success"] == 1


def test_a4_replay_is_detected():
    base = P.build_pipeline()
    entry = A.attack("a4_replay", base)
    result = V.verify_chain(entry)
    assert result["ok"] is False
    assert result["attack_success"] == 0


def test_chain_links_are_checked():
    base = P.build_pipeline()
    # break only the CHAIN link, keeping both per-pass hashes self-consistent
    entry = dict(base)
    entry["cert2"] = dict(base["cert2"])
    entry["cert2"]["input_hash"] = "0" * 32
    result = V.verify_chain(entry)
    assert result["ok"] is False
    assert any("chain" in r or "input_hash" in r for r in result["reasons"])
