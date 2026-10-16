#!/usr/bin/env python3
"""The CORRECT relative-phase legality criterion (calibrated 2026-10-16).

Convention
----------
The project's construction multiplies the correction on the **left**:

    U_candidate = C . U_ideal            (see ``block_diagonal``: B_rel = D . B_exact)

so the correction is ``C = U_candidate . U_ideal^dagger``.  The rewrite is legal
iff ``C`` is **diagonal with unit-modulus entries** -- this is necessary AND
sufficient for preserving the Z-basis measurement statistics of *any* input
state (basis, superposed, entangled):

  * sufficient: ``(C U)|i,j> = c_i U_{i,j}`` so ``|.|`` is unchanged;
  * necessary : a non-zero off-diagonal ``C_ij`` distorts the statistics of
    ``U|psi> = (|i> + |j>)/sqrt 2``.

Do NOT use ``U_ideal^dagger U_candidate`` (the right-multiplied convention): it
false-rejects left-multiplied constructions and false-accepts right-multiplied
ones, which are only safe for computational-basis inputs.  See
``calibrate_truth_criterion.py`` for the 5/5 physical calibration.

Usage
-----
    .venv/bin/python relative_phase_criterion.py --self-test
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent


def _as_operator(circuit_or_matrix: Any) -> np.ndarray:
    if isinstance(circuit_or_matrix, np.ndarray):
        return circuit_or_matrix
    from qiskit import qasm2
    from qiskit.quantum_info import Operator

    text = qasm2.dumps(circuit_or_matrix)
    return np.asarray(Operator(qasm2.loads(text)).data, dtype=complex)


def relative_phase_metrics(ideal: Any, candidate: Any,
                           ancillas: Optional[Sequence[int]] = None
                           ) -> Tuple[float, float]:
    """``(offdiagonal_max, ||diag|-1| max)`` of ``C = U_cand . U_ideal^dagger``.

    Legal iff both are below tolerance.  ``ancillas`` optionally restricts the
    comparison to the clean-ancilla (all-zero) subspace, which is the right
    thing to do when the ancillas are post-selected.
    """
    u_ideal = _as_operator(ideal)
    u_cand = _as_operator(candidate)
    if u_ideal.shape != u_cand.shape:
        raise ValueError("ideal and candidate must act on the same space")
    if ancillas:
        keep = [x for x in range(u_ideal.shape[0])
                if all(not (x >> int(a)) & 1 for a in ancillas)]
        u_ideal = u_ideal[np.ix_(keep, keep)]
        u_cand = u_cand[np.ix_(keep, keep)]
    correction = u_cand @ u_ideal.conj().T
    offdiag = float(np.abs(correction - np.diag(np.diag(correction))).max())
    unit = float(np.abs(np.abs(np.diag(correction)) - 1.0).max())
    return offdiag, unit


def relative_phase_ok(ideal: Any, candidate: Any,
                      ancillas: Optional[Sequence[int]] = None,
                      tol: float = 1e-9) -> Optional[bool]:
    """Legality verdict, or ``None`` when the space is too large for a dense check."""
    u_ideal = _as_operator(ideal)
    if u_ideal.shape[0] > 2 ** 11:
        return None
    offdiag, unit = relative_phase_metrics(ideal, candidate, ancillas)
    return bool(offdiag < tol and unit < tol)


def _self_test() -> int:
    """Calibration: positive AND negative examples (never accept everything)."""
    rng = np.random.default_rng(20261016)

    def rand_unitary(dim: int) -> np.ndarray:
        z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(z)
        return q

    dim = 4
    u_ideal = rand_unitary(dim)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, dim))
    cases = [
        ("left  D.U (our construction)", phase[:, None] * u_ideal, True),
        ("identity", u_ideal.copy(), True),
        ("left D.U, global phase", (np.exp(0.7j) * np.ones(dim))[:, None] * u_ideal, True),
        ("right U.D (only basis inputs)", u_ideal * phase[None, :], False),
        ("random unitary (BAD)", rand_unitary(dim), False),
    ]
    failures = 0
    for label, candidate, expected in cases:
        got = relative_phase_ok(u_ideal, candidate)
        mark = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print(f"  [{mark}] {label:<34} expected={expected} got={got}")
    if failures:
        print(f"self-test FAILED ({failures})")
        return 1
    print("  self-test passed: criterion matches the declared convention, "
          "including the negative controls")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="correct relative-phase criterion.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
