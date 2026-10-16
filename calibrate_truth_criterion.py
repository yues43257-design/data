#!/usr/bin/env python3
"""Physical-layer calibration of the legality criterion (do this BEFORE re-running).

The old ``relative_phase_metrics`` checks ``U_ideal^dagger U_cand`` diagonal, i.e.
it accepts the *right*-multiplied convention ``U_cand = U_ideal . D``.  The
project's own construction is *left*-multiplied (``B_relative = D . B_exact``).

Two candidate replacements are on the table:

    C1 : ``C = U_cand . U_ideal^dagger`` diagonal      <- proposed here
    C2 : entrywise ``|U_cand| == |U_ideal|``            <- rejected (too weak)

This script settles which one equals "Z-basis measurement statistics are
preserved", by brute force on random input states (basis, superposed, entangled)
-- no library convention is involved.  It includes known-BAD candidates so the
checker cannot degenerate into "accept everything".

Expected outcomes (to be confirmed, not assumed):

    U_cand = D_left  . U_ideal  -> safe for ALL inputs  (our construction)
    U_cand = U_ideal . D_right  -> safe ONLY for basis inputs
    U_cand = random unitary     -> never safe

Usage
-----
    .venv/bin/python calibrate_truth_criterion.py
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20261016)
N = 2
DIM = 2 ** N
TOL = 1e-9


def rand_unitary(dim: int) -> np.ndarray:
    z = RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))
    q, _ = np.linalg.qr(z)
    return q


def random_states(count: int) -> list:
    """Computational basis, random superpositions, and entangled 2-qubit states."""
    states = []
    for i in range(DIM):
        v = np.zeros(DIM, dtype=complex)
        v[i] = 1.0
        states.append(v)
    for _ in range(count):
        v = RNG.normal(size=DIM) + 1j * RNG.normal(size=DIM)
        states.append(v / np.linalg.norm(v))
    # entangled: |00> + |11> rotated by a random 2-qubit unitary, applied to |00>
    bell = np.zeros(DIM, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    states.append(bell)
    states.append(rand_unitary(DIM) @ bell)
    return states


def distribution(u: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.abs(u @ psi) ** 2


def physically_safe(u_ideal: np.ndarray, u_cand: np.ndarray, states) -> bool:
    return all(np.allclose(distribution(u_ideal, s), distribution(u_cand, s),
                           atol=1e-9) for s in states)


def c1_diagonal(u_ideal: np.ndarray, u_cand: np.ndarray) -> bool:
    c = u_cand @ u_ideal.conj().T
    return float(np.abs(c - np.diag(np.diag(c))).max()) < TOL


def c2_entrywise(u_ideal: np.ndarray, u_cand: np.ndarray) -> bool:
    return bool(np.allclose(np.abs(u_cand), np.abs(u_ideal), atol=TOL))


def old_criterion(u_ideal: np.ndarray, u_cand: np.ndarray) -> bool:
    c = u_ideal.conj().T @ u_cand
    off = float(np.abs(c - np.diag(np.diag(c))).max())
    unit = float(np.abs(np.abs(np.diag(c)) - 1.0).max())
    return bool(off < TOL and unit < TOL)


def main() -> int:
    states = random_states(24)
    u_ideal = rand_unitary(DIM)
    d = np.exp(1j * RNG.uniform(0, 2 * np.pi, DIM))
    cases = {
        "left  D.U  (our construction)": d[:, None] * u_ideal,
        "right U.D": u_ideal * d[None, :],
        "identity": u_ideal.copy(),
        "random unitary (BAD)": rand_unitary(DIM),
        "left D.U with D = phase only": (np.exp(1j * 0.7) * np.ones(DIM))[:, None] * u_ideal,
    }
    print("=" * 92)
    print(f"  criterion calibration on {len(states)} input states "
          f"(basis + superposed + entangled), {N} qubits")
    print("=" * 92)
    print(f"  {'candidate':<32}{'physically safe':>16}{'C1 diag':>9}"
          f"{'C2 entrywise':>14}{'old fn':>8}")
    rows = []
    for label, u_cand in cases.items():
        phys = physically_safe(u_ideal, u_cand, states)
        a, b, c = c1_diagonal(u_ideal, u_cand), c2_entrywise(u_ideal, u_cand), \
            old_criterion(u_ideal, u_cand)
        print(f"  {label:<32}{str(phys):>16}{str(a):>9}{str(b):>14}{str(c):>8}")
        rows.append((label, phys, a, b, c))
    print("=" * 92)
    c1_ok = all(r[1] == r[2] for r in rows)
    c2_ok = all(r[1] == r[3] for r in rows)
    old_ok = all(r[1] == r[4] for r in rows)
    print(f"  C1 (U_cand . U_ideal^dag diagonal) matches physics on all cases: {c1_ok}")
    print(f"  C2 (entrywise |U_cand| == |U_ideal|) matches physics:            {c2_ok}")
    print(f"  old relative_phase_metrics matches physics:                     {old_ok}")
    print("  (a checker that accepts EVERYTHING would also fail the random/BAD row)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
