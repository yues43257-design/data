"""Acceptance tests for the relative-phase watermark probe."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments.watermark_probe.attacks as A   # noqa: E402
import experiments.watermark_probe.detect as D    # noqa: E402
import experiments.watermark_probe.embed as E     # noqa: E402

KEY = [1, 0, 1, 1, 0, 1, 0, 0]        # 8 bits; cdkm3 uses the first 6


def test_embed_then_detect_gives_zero_ber():
    watermarked = E.embed(KEY)
    bits, found, total = D.detect_bits(watermarked["circuit"])
    assert found == total == watermarked["capacity"]
    assert bits == watermarked["key_used"]
    assert D.ber(bits, watermarked["key_used"]) == 0.0


def test_embedded_mask_is_tracker_safe():
    watermarked = E.embed(KEY)
    assert watermarked["tracker_safe"] is True
    assert watermarked["capacity"] == 6          # CDKM 3-bit has 6 Toffolis


def test_all_exact_without_watermark_is_not_decoded_as_key():
    watermarked = E.embed(KEY)
    plain = E.source_circuit("cdkm3")
    bits, found, total = D.detect_bits(plain)
    assert found == total == 6
    assert bits == [0] * 6
    assert D.ber(bits, watermarked["key_used"]) > 0


def test_block_deletion_increases_ber():
    watermarked = E.embed(KEY)
    attacked = A.flip_blocks(watermarked, count=2)
    bits, _, _ = D.detect_bits(attacked)
    assert D.ber(bits, watermarked["key_used"]) > 0


def test_run_all_reports_every_attack():
    results = A.run_all(E.embed(KEY), write=False)
    names = {r["attack"] for r in results}
    for expected in ("none", "pyzx_full_reduce", "todd", "flip_one",
                     "all_exact", "false_positive"):
        assert expected in names


def test_optimizers_destroy_or_preserve_is_recorded():
    results = A.run_all(E.embed(KEY), write=False)
    for r in results:
        assert "ber" in r and "blocks_found" in r and "t_count" in r
