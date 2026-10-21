"""Sanity + determinism tests for the compiler benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.compiler_bench.cases as C          # noqa: E402
import experiments.compiler_bench.run_bench as R      # noqa: E402


def test_at_least_ten_cases_with_ground_truth():
    cases = C.all_cases()
    assert len(cases) >= 10
    names = {c["name"] for c in cases}
    for expected in ("legal_control", "illegal_rtof_mask", "block_substitution",
                     "nonblock_x", "tail_h", "gate_smuggle_u3",
                     "gate_smuggle_measure", "cert_forge", "cert_replay",
                     "pass2_substitution", "pass2_forgery"):
        assert expected in names
    for case in cases:
        assert "ground_truth" in case and case["ground_truth"]
        assert case["expected_cert"] in (True, False)


def test_verifier_outcomes_are_distinguishable():
    report = R.run()
    rows = {r["case"]: r for r in report["rows"]}
    assert rows["legal_control"]["dense_legal"] is True
    assert rows["legal_control"]["cert_ok"] is True
    assert rows["tail_h"]["dense_legal"] is False
    assert rows["tail_h"]["cert_ok"] is False
    # dense/QCEC answer legality and are blind to certificate tampering
    assert rows["cert_forge"]["dense_legal"] is True
    assert rows["cert_forge"]["cert_ok"] is False
    # the known gap: a compromised pass with correct hashes is accepted
    assert rows["pass2_forgery"]["cert_ok"] is True
    assert report["metrics"]["certificate"]["correct"] == 10


def test_deterministic_verdicts():
    first = {r["case"]: (r["dense_legal"], r["qcec_legal"], r["cert_ok"])
             for r in R.run()["rows"]}
    second = {r["case"]: (r["dense_legal"], r["qcec_legal"], r["cert_ok"])
              for r in R.run()["rows"]}
    assert first == second
