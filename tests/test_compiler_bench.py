"""Tests for the generated suite and the L/S/C scoring protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compiler_bench.cases as C          # noqa: E402
import experiments.compiler_bench.run_bench as R      # noqa: E402


def test_generated_suite_has_l_s_and_properties():
    cases = C.all_cases()
    assert len(cases) >= 10
    attacks = {c.get("attack") for c in cases}
    for expected in ("legal", "block_substitution", "nonblock_x", "tail_h",
                     "gate_smuggle_u3", "gate_smuggle_measure", "cert_forge",
                     "cert_replay", "pass2_substitution", "pass2_forgery"):
        assert expected in attacks, expected
    properties = {c.get("cert_property") for c in cases if "error" not in c}
    for expected in ("positive", "binding", "tamper", "chain",
                     "illegal_rewrite", "undecided_legal"):
        assert expected in properties, expected
    for case in cases:
        if "error" in case:
            continue
        assert "truth_L" in case and "truth_S" in case and case["ground_truth"]


def test_legal_control_is_legal_but_not_exactly_equivalent():
    cases = {c["name"]: c for c in C.all_cases() if "error" not in c}
    legal = cases["two_block/plain/s0/legal"]
    assert legal["truth_L"] is True
    assert legal["truth_S"] is False       # differs by a diagonal
    assert legal["cert_property"] == "positive"


def test_l_and_c_metrics():
    report = R.run(with_qcec=False)
    L = report["L"]["dense"]
    assert L["soundness"] == 1.0
    assert L["completeness"] == 1.0
    Cm = report["C"]
    assert Cm["positive_accept"]["rate"] == 1.0
    assert Cm["binding_reject"]["rate"] == 1.0
    assert Cm["tamper_reject"]["rate"] == 1.0
    # the known structural gap: one of the two chain cases is accepted
    assert Cm["chain_reject"]["correct"] == 1


def test_semantics_verifier_uses_common_subset():
    report = R.run(with_qcec=True, qcec_limit=8)
    assert report["S"]["common_subset"] >= 1
    assert report["S"]["accuracy"] == 1.0
    assert report["S"]["ci"][0] is not None


def test_abstentions_are_not_false_positives():
    report = R.run(with_qcec=False)
    rows = [r for r in report["rows"] if "error" not in r]
    abstained = [r for r in rows if r["truth_L"] is None]
    assert abstained and all(r["dense_legal"] is None for r in abstained)
    assert report["C"]["undecided_legal_cases"] == sum(
        1 for r in rows if r["cert_property"] == "undecided_legal")


def test_deterministic_verdicts():
    first = {r["case"]: (r.get("dense_legal"), r.get("cert_ok"))
             for r in R.run()["rows"]}
    second = {r["case"]: (r.get("dense_legal"), r.get("cert_ok"))
              for r in R.run()["rows"]}
    assert first == second
