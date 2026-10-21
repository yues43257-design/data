"""Sanity + determinism tests for the application probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.application_probe.run_probe as P  # noqa: E402


def test_sanity_known_families():
    rows = {r["family"]: r for r in (P.measure(s) for s in P._families())}
    cdkm = rows["cdkm_adder_3"]
    assert cdkm["in_domain"] is True
    assert cdkm["blocks"] == 6
    assert cdkm["legality"] == "legal"
    # the first paper's known numbers
    assert cdkm["relative_t"] == 24
    assert cdkm["full_reduce_t"] == 24

    vbe = rows["vbe_adder_3"]
    assert vbe["in_domain"] is True
    assert vbe["relative_t"] == 4 * vbe["blocks"]
    assert vbe["full_reduce_t"] <= vbe["relative_t"]      # FR wins -> NO-GO


def test_deterministic_results():
    first = [P.measure(s) for s in P._families()]
    second = [P.measure(s) for s in P._families()]
    for a, b in zip(first, second):
        for key in ("family", "blocks", "t_count", "t_depth", "relative_t",
                    "full_reduce_t", "legality", "in_domain"):
            assert a.get(key) == b.get(key), (a["family"], key)


def test_out_of_domain_families_are_recorded():
    rows = {r["family"]: r for r in (P.measure(s) for s in P._families())}
    assert rows["modular_adder_3"]["in_domain"] is False
    assert rows["modular_adder_3"]["blocks"] == 0
    # QFT-family transpilation is expected to fail rather than be forced
    for name in ("qft_4", "draper_adder_3", "multiplier_2"):
        assert rows[name]["in_domain"] is False
        assert "error" in rows[name]
