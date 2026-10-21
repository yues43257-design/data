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
    for expected in ("legal", "h_on_support", "block_substitution", "tail_h",
                     "gate_smuggle_u3", "cert_forge",
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
    assert Cm["positive_accept"]["rate"] < 1.0      # certificate false negatives
    assert Cm["illegal_rewrite_reject"]["rate"] == 1.0
    assert Cm["binding_reject"]["rate"] == 1.0
    assert Cm["tamper_reject"]["rate"] == 1.0
    # the known structural gap: 10 substitution cases rejected, 45 forgeries not
    assert Cm["chain_reject"]["correct"] == 10
    assert Cm["chain_reject"]["cases"] == 55


def test_second_divergence_certificate_false_negatives():
    """Legal rewrites the tracker rejects (H...H restores the phase)."""
    report = R.run(with_qcec=False)
    assert report["C"]["positive_accept"]["rate"] < 1.0
    fn = [r for r in report["rows"]
          if r.get("cert_property") == "positive" and r.get("truth_L") is True
          and r.get("cert_ok") is False]
    assert fn and all("hh_restore" in r["case"] for r in fn)


def test_d1_reproduces_across_parameters():
    report = R.run(with_qcec=False)
    by_family = report["C"]["positive_by_family"]
    assert by_family["hh_restore"]["cases"] >= 9
    assert by_family["hh_restore"]["rate"] == 0.0
    for family in ("two_block", "adder", "vbe", "comparator"):
        assert by_family[family]["rate"] == 1.0


def test_real_qasmbench_family_and_d1_scope():
    """D1 is mechanism-specific: the real in-domain circuits do not show it."""
    report = R.run(with_qcec=False)
    by_family = report["C"]["positive_by_family"]
    assert "real_qasmbench" in by_family
    assert by_family["real_qasmbench"]["rate"] == 1.0
    assert by_family["hh_restore"]["rate"] == 0.0


def test_chain_metrics_are_split():
    report = R.run(with_qcec=False)
    Cm = report["C"]
    assert Cm["chain_substitution"]["rate"] == 1.0
    assert Cm["chain_substitution"]["cases"] == 10
    assert Cm["chain_forgery"]["rate"] == 0.0
    assert Cm["chain_forgery"]["cases"] == 45


def test_protocol_is_frozen():
    report = R.run(with_qcec=False)
    assert report["protocol_version"] == "compiler-bench-protocol-v1"
    assert len(report["protocol_fingerprint"]) == 16


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
