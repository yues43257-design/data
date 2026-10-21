"""Sanity, scoring-protocol and determinism tests for the benchmark suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compiler_bench.cases as C          # noqa: E402
import experiments.compiler_bench.run_bench as R      # noqa: E402


def test_generated_suite_has_three_ground_truths():
    cases = C.all_cases()
    assert len(cases) >= 10
    attacks = {c.get("attack") for c in cases}
    for expected in ("legal", "block_substitution", "nonblock_x", "tail_h",
                     "gate_smuggle_u3", "gate_smuggle_measure", "cert_forge",
                     "cert_replay", "pass2_substitution", "pass2_forgery"):
        assert expected in attacks, expected
    for case in cases:
        if "error" in case:
            continue
        for key in ("truth_L", "truth_S", "truth_C", "ground_truth"):
            assert key in case
        assert case["ground_truth"]


def test_legal_control_and_relative_phase_semantics():
    cases = {(c["name"]): c for c in C.all_cases() if "error" not in c}
    legal = cases["two_block/plain/legal"]
    assert legal["truth_L"] is True
    assert legal["truth_S"] is False        # differs by a diagonal
    assert legal["truth_C"] is True


def test_scoring_protocol_excludes_abstentions():
    report = R.run(with_qcec=False)
    metrics = report["metrics"]
    # dense is the oracle for L
    assert metrics["dense"]["wrong"] == 0
    # unbound certificate misses circuit attacks; bound catches them
    assert metrics["cert_v1"]["wrong"] > metrics["cert_v2"]["wrong"]
    assert metrics["cert_v2"]["wrong"] == 1
    wrong = [r for r in report["rows"] if r.get("score_cert_v2") == "wrong"]
    assert len(wrong) == 1 and "pass2_forgery" in wrong[0]["case"]
    # n/a cases are abstentions, not false positives
    abstain = [r for r in report["rows"] if r.get("score_dense") == "abstain"]
    assert abstain and all(r.get("dense") is None for r in abstain)


def test_deterministic_verdicts():
    first = {r["case"]: (r.get("dense"), r.get("cert_v1"), r.get("cert_v2"))
             for r in R.run()["rows"]}
    second = {r["case"]: (r.get("dense"), r.get("cert_v1"), r.get("cert_v2"))
              for r in R.run()["rows"]}
    assert first == second
