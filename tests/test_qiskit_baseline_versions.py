"""Tests for the Qiskit L3 sensitivity artifact."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "figure_data" / "qiskit_baseline_sensitivity.json"


def test_qiskit_baseline_sensitivity_covers_all_claims():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["version"] == "qiskit-baseline-sensitivity-v1"
    assert report["primary_qiskit"] == "2.5.2"
    assert report["sensitivity_qiskit"] == "1.3.0"
    rows = {row["circuit"]: row for row in report["rows"]}
    expected = {
        "mcx(3)", "mcx(4)", "mcx(5)",
        "adder(3)", "adder(4)", "adder(8)", "adder(16)", "adder(32)",
        "two_adders(3)", "two_adders(4)",
        "comparator(3)", "comparator(4)",
        "grover(3)",
    }
    assert expected <= set(rows)
    for name in expected:
        row = rows[name]
        assert row["qiskit_l3_primary"] > 0
        assert row["qiskit_l3_sensitivity"] > 0
        assert row["ours_safe"] > 0
        if row["legal_rewrite"]:
            assert row["saving_pct_primary"] is not None
            assert row["saving_pct_sensitivity"] is not None
        else:
            assert row["saving_pct_primary"] is None


def test_claim_percentages_are_version_specific():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["rows_with_legal_rewrite"] == 12
    assert -40.0 < summary["primary_min_pct"] <= summary["primary_max_pct"] < -30.0
    assert summary["sensitivity_max_pct"] <= summary["primary_min_pct"]
    assert summary["sensitivity_max_pct"] < -40.0
    assert summary["max_abs_pp_change"] > 0
