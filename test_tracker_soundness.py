#!/usr/bin/env python3
"""Permanent regression tests for the two tracker-v2 counterexamples.

1. ``H T H`` does **not** preserve diagonality of a Z-type residual:
   ``(H T H) Z (H T H)^dagger = (1/sqrt2)(Z - Y)``, max|offdiag| = 1/sqrt2.
   => the "Z -> X -> Z through an even number of H's is safe" rule is UNSOUND.
2. The CNOT X-spread direction: ``CNOT(c, t)`` sends ``X_c -> X_c X_t``
   (control spreads to target), not the other way round.  The earlier v2
   diagnostic had this backwards.

Run:  .venv/bin/python test_tracker_soundness.py
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
Z = np.diag([1, -1]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def test_hth_breaks_diagonality() -> None:
    """The minimal counterexample: the naive type model would accept this."""
    m = H @ T @ H
    c = m @ Z @ m.conj().T
    off = float(np.abs(c - np.diag(np.diag(c))).max())
    assert off > 1e-6, "H T H unexpectedly preserved diagonality"
    assert abs(off - 1 / np.sqrt(2)) < 1e-9, f"expected 1/sqrt(2), got {off}"
    # it is a two-Pauli mixture: each column has 2 non-zeros, whereas
    # (Pauli * diagonal) must have exactly one non-zero per column.
    counts = [int((np.abs(c[:, j]) > 1e-9).sum()) for j in range(c.shape[1])]
    assert counts == [2, 2], f"expected 2 non-zeros per column, got {counts}"
    print(f"  [ok  ] H T H breaks diagonality (max|offdiag|={off:.4f})")


def test_cnot_spreads_x_from_control_to_target() -> None:
    """CNOT(c,t): X_c -> X_c X_t ; X_t -> X_t.  Order matters."""
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex)  # |c t>, control first
    x0 = np.kron(X, np.eye(2))
    x1 = np.kron(np.eye(2), X)
    spread = cnot @ x0 @ cnot.conj().T
    assert np.allclose(spread, x0 @ x1), "control X must spread to the target"
    keep = cnot @ x1 @ cnot.conj().T
    assert np.allclose(keep, x1), "target X must stay on the target"
    assert not np.allclose(spread, x0), "the reversed (buggy) rule would pass"
    print("  [ok  ] CNOT spreads X control->target (reversed rule rejected)")


def test_known_unsound_corpus_rejected() -> None:
    """Mask-level corpus: every mask the *naive* type model accepts but the exact
    verifier rejects must be REJECTED by any sound tracker candidate.

    Pinned to the canonical setup (``experiment_setup``), so the corpus is
    reproducible by (h, controls, seed, mask).  Regenerate the count and assert.
    """
    setup = _load("ts_setup", "experiment_setup.py")
    symbolic = _load("ts_symbolic", "symbolic_verifier.py")
    tracker = _load("ts_tracker", "support_tracker.py")
    bailout = _load("ts_bailout", "tracker_bailout_check.py")
    corpus = 0
    for h_prob in (0.1, 0.3, 0.7):
        for num_controls in (2, 3, 4, 5):
            for seed_index in range(3):
                seed = setup.seed_for(seed_index)
                verifier = symbolic.ReducedSymbolicVerifier(num_controls, 6, seed, h_prob)
                schedule = tracker.homogeneous_schedule(num_controls, 6, seed, h_prob)
                for mask in range(1 << 6):
                    if bailout.naive_v2(schedule, mask) and not verifier.is_safe(mask):
                        corpus += 1
                        assert not bailout.sound_v2(schedule, mask), (
                            "a sound tracker accepted a known-unsound mask "
                            f"(h={h_prob}, controls={num_controls}, seed={seed}, mask={mask})"
                        )
    assert corpus > 0, "the known-unsound corpus came out empty - setup drift?"
    print(f"  [ok  ] known-unsound corpus = {corpus} masks; sound tracker rejects all")


if __name__ == "__main__":
    print("tracker soundness regression tests")
    try:
        test_hth_breaks_diagonality()
        test_cnot_spreads_x_from_control_to_target()
        test_known_unsound_corpus_rejected()
    except AssertionError as exc:
        print(f"  [FAIL] {exc}")
        raise SystemExit(1)
    print("  all tracker-v2 counterexample tests passed")
