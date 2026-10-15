"""Frozen canonical setup for every recall / soundness number in the report.

Every recall figure quoted in ``reports/strategy_b_experiments.md`` must come
from this configuration.  Earlier rounds used different seeds, which is why the
same ``h_prob`` produced different legal counts (e.g. single-block legal 67 vs 57
vs 72).  If a measurement changes the setup, bump ``SETUP_VERSION`` and label the
number with it.
"""

from __future__ import annotations

from typing import Dict

# v2 = seeds fixed to ``seed_index * 7 + 11`` (the original tracker_verification.py
#      convention) and the CNOT X-spread direction corrected in the v2 diagnostics.
SETUP_VERSION = "canonical-setup-v2"

CANONICAL: Dict[str, object] = {
    "setup_version": SETUP_VERSION,
    "family": "homogeneous C^nX + Clifford interlayers",
    "controls": (2, 3, 4, 5),
    "blocks": 6,
    "seeds": 3,
    "seed_rule": "seed_index * 7 + 11",
    "h_probs": (0.0, 0.1, 0.3, 0.7, 1.0),
    "masks": "full enumeration 2**6 = 64 per instance",
    "exact_oracle": "symbolic_verifier.ReducedSymbolicVerifier",
    "tracker": "support_tracker.track_safe",
}


def seed_for(index: int) -> int:
    """The single seed rule; every script must call this, never inline a formula."""
    return int(index) * 7 + 11
