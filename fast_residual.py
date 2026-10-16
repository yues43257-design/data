#!/usr/bin/env python3
"""Exact residual / support machinery for the reinforcement-4 corpus.

This computes exactly the same object as ``superset_check.py``:

    R = U_cand . U_ideal^dagger                      (a 2**n x 2**n matrix)
    true_support(R) = { q : R does not commute with X_q,
                            or diag(R) depends on bit q of the basis state }

and nothing about it is new mathematics -- the *definition* is unchanged.  What
changes is only the cost, because the dense version in ``superset_check.py``
costs ``O(8**n)`` per qubit (two full matrix products with a ``2**n x 2**n``
``X_q``) and therefore cannot reach ``N = 12``:

  1. ``X_q`` is the permutation matrix ``a -> a ^ 2**q``, so ``R @ X_q``
     permutes columns and ``X_q @ R`` permutes rows of the *same* matrix.  The
     commutation test becomes a comparison of two block-structured views;
     the diagonal test becomes ``diag(R)`` against ``diag(R)[idx]``.
  2. ``R = U_cand . U_ideal^dagger`` is accumulated by applying gates to one
     dense matrix (``O(4**n)`` per gate) instead of materialising both
     unitaries and calling BLAS once (``O(8**n)``).
  3. Gate application is in place with half-size temporaries: no ``2**n x 2**n``
     allocation per gate.

Everything is cross-validated against ``qiskit.quantum_info.Operator`` and
against the original ``O(8**n)`` definition when this file is run directly.

Scope: the generated instances use the gate set ``{h, t, tdg, cx}`` only.
Anything else raises ``NotImplementedError`` rather than being silently
approximated.

Numerical note (reported as part of the method, not hidden): the diagonal branch
is normalised by the LARGEST-magnitude diagonal entry, not by ``diag[0]``.  On
masks the tracker rejects ``R`` is not diagonal and ``diag[0]`` is frequently a
numerical zero (~1e-16), so ``diag / diag[0]`` was pure amplified noise.  On
every tracker-ACCEPTED mask ``|diag(R)| == 1``, so the two normalisations agree
exactly there; ``true_support`` returns ``None`` when the residual has no usable
diagonal at all (only possible for rejected masks).
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np

INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _one_qubit_tables(dtype):
    return {
        "h": (INV_SQRT2 * np.array([[1, 1], [1, -1]])).astype(dtype),
        "t": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]).astype(dtype),
        "tdg": np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]).astype(dtype),
        "s": np.array([[1, 0], [0, 1j]]).astype(dtype),
        "sdg": np.array([[1, 0], [0, -1j]]).astype(dtype),
        "x": np.array([[0, 1], [1, 0]]).astype(dtype),
        "z": np.array([[1, 0], [0, -1]]).astype(dtype),
    }


_ADJOINT_NAME = {"t": "tdg", "tdg": "t", "s": "sdg", "sdg": "s"}


def _gate_of(circuit, instruction, adjoint: bool):
    name = instruction.operation.name
    if adjoint:
        name = _ADJOINT_NAME.get(name, name)
    qubits = [circuit.find_bit(q).index for q in instruction.qubits]
    return name, qubits


def _left_1q(M, g, q, n):
    """``M <- (g on qubit q) . M`` in place, one half-size temporary."""
    R = M.reshape(1 << (n - q - 1), 2, 1 << q, M.shape[1])
    a = R[:, 0]
    b = R[:, 1]
    if g[0, 0] == 0.0:                       # X-like: pure swap
        tmp = b.copy()
        b[...] = a
        a[...] = tmp
        return M
    d = g[1, 0] * a + g[1, 1] * b            # temporary holds the s=1 half
    a *= g[0, 0]
    a += g[0, 1] * b
    R[:, 1] = d
    return M


def _right_1q(M, g, q, n):
    """``M <- M . (g on qubit q)`` in place, one half-size temporary."""
    C = M.reshape(M.shape[0], 1 << (n - q - 1), 2, 1 << q)
    a = C[:, :, 0]
    b = C[:, :, 1]
    if g[0, 0] == 0.0:
        tmp = b.copy()
        b[...] = a
        a[...] = tmp
        return M
    d = g[0, 1] * a + g[1, 1] * b
    a *= g[0, 0]
    a += b * g[1, 0]
    C[:, :, 1] = d
    return M


def _left_cx(M, c, t, n):
    """``M <- CX(c, t) . M``: a row permutation that swaps two row blocks."""
    cols = M.shape[1]
    V = np.moveaxis(M.reshape((2,) * n + (cols,)), (n - 1 - c, n - 1 - t), (0, 1))
    tmp = V[1, 0].copy()
    V[1, 0] = V[1, 1]
    V[1, 1] = tmp
    return M


def _right_cx(M, c, t, n):
    """``M <- M . CX(c, t)``: the same permutation on the column index."""
    rows = M.shape[0]
    V = np.moveaxis(M.reshape((rows,) + (2,) * n),
                    (1 + n - 1 - c, 1 + n - 1 - t), (0, 1))
    tmp = V[1, 0].copy()
    V[1, 0] = V[1, 1]
    V[1, 1] = tmp
    return M


def build_unitary(circuit, adjoint: bool = False, dtype=complex) -> np.ndarray:
    """Dense unitary by left-applying gates (``adjoint=True`` -> ``U^dagger``)."""
    n = circuit.num_qubits
    table = _one_qubit_tables(dtype)
    M = np.eye(1 << n, dtype=dtype)
    data = list(circuit.data)
    if adjoint:
        data = data[::-1]
    for instruction in data:
        name, qubits = _gate_of(circuit, instruction, adjoint)
        if name in table:
            _left_1q(M, table[name], qubits[0], n)
        elif name == "cx":
            _left_cx(M, qubits[0], qubits[1], n)
        else:                                # pragma: no cover - defensive
            raise NotImplementedError(f"gate {name!r} outside the dense kernel")
    return M


def residual(u_cand: np.ndarray, ideal_circuit) -> np.ndarray:
    """``R = U_cand . U_ideal^dagger`` (``u_cand`` is consumed in place)."""
    n = ideal_circuit.num_qubits
    table = _one_qubit_tables(u_cand.dtype)
    M = u_cand
    for instruction in ideal_circuit.data:   # U_ideal^dag = G_1^dag ... G_K^dag
        name, qubits = _gate_of(ideal_circuit, instruction, True)
        if name in table:
            _right_1q(M, table[name], qubits[0], n)
        elif name == "cx":
            _right_cx(M, qubits[0], qubits[1], n)
        else:                                # pragma: no cover - defensive
            raise NotImplementedError(f"gate {name!r} outside the dense kernel")
    return M


def off_diagonal(r: np.ndarray) -> float:
    """Largest absolute off-diagonal entry of the residual."""
    return float(np.abs(r - np.diag(np.diag(r))).max())


def commutes_with_x(r: np.ndarray, q: int, n: int, atol: float) -> bool:
    """Is ``R X_q == X_q R``?  (Two block comparisons, no 2**n x 2**n product.)

    With ``A_ab`` the four quadrant blocks split by bit ``q`` of the row and
    column index, ``R X_q`` is ``[[A01, A00], [A11, A10]]`` and ``X_q R`` is
    ``[[A10, A11], [A00, A01]]``, so commutation is equivalent to
    ``A01 == A10`` and ``A00 == A11``.
    """
    view = r.reshape((2,) * n + (2,) * n)          # [row bits][column bits]
    view = np.moveaxis(view, (n - 1 - q, 2 * n - 1 - q), (0, 1))
    if not np.allclose(view[0, 1], view[1, 0], atol=atol, rtol=0.0):
        return False
    if not np.allclose(view[0, 0], view[1, 1], atol=atol, rtol=0.0):
        return False
    return True


def true_support(r: np.ndarray, n: int, atol: float = 1e-9) -> Optional[Set[int]]:
    """Qubits the residual genuinely depends on; ``None`` if undefined."""
    diag = np.diag(r)
    pivot = int(np.argmax(np.abs(diag)))
    if abs(diag[pivot]) < 1e-9:
        return None                          # residual has no usable diagonal
    d = diag / diag[pivot]
    flat = np.arange(1 << n)
    support: Set[int] = set()
    for q in range(n):
        if not commutes_with_x(r, q, n, atol):
            support.add(q)
            continue
        idx = flat ^ (1 << q)
        if bool(np.any(np.abs(d - d[idx]) > atol)):
            support.add(q)
    return support


def _self_test() -> int:
    """Compare against qiskit Operator and the original O(8**n) definition."""
    import importlib.machinery
    import importlib.util
    import random
    import sys
    from pathlib import Path

    from qiskit.quantum_info import Operator

    root = Path(__file__).resolve().parent
    loader = importlib.machinery.SourceFileLoader(
        "fr_test_ult", str(root / "ultimate_scalability_test.py"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)

    rng = random.Random(0)
    worst_residual = 0.0
    checked = compared = excluded = 0
    bad_exclusions = []
    for _ in range(10):
        width = rng.randint(3, 6)
        blocks = rng.randint(1, 4)
        seed = rng.randint(0, 10 ** 6)
        inst = module.WideInstance(width=width, blocks=blocks, seed=seed,
                                   h_prob=0.7, ancilla_pool=2)
        ideal = inst.circuit(0)
        u_ideal = np.asarray(Operator(ideal).data, dtype=complex)
        n = inst.total
        for mask in range(1 << blocks):
            qc = inst.circuit(mask)
            u_cand = np.asarray(Operator(qc).data, dtype=complex)
            ref_r = u_cand @ u_ideal.conj().T
            mine_r = residual(build_unitary(qc), ideal)
            worst_residual = max(worst_residual, float(np.abs(ref_r - mine_r).max()))
            # original definition, one dense 2**n x 2**n X_q per qubit
            ref_support = set()
            ref_diag = np.diag(ref_r) / np.diag(ref_r)[0]
            for q in range(n):
                x = np.kron(np.kron(np.eye(2 ** (n - q - 1)),
                                    np.array([[0, 1], [1, 0]])), np.eye(2 ** q))
                if not np.allclose(ref_r @ x, x @ ref_r, atol=1e-9):
                    ref_support.add(q)
                elif np.any(np.abs(ref_diag - ref_diag[np.arange(1 << n) ^ (1 << q)]) > 1e-9):
                    ref_support.add(q)
            mine_support = true_support(mine_r, n)
            # The old routine divides by diag[0].  That is only meaningful when
            # diag[0] is not a numerical zero, i.e. when the residual really is
            # diagonal (which is exactly the tracker-accepted situation).  Where
            # it is not, the old answer is amplified noise, so we require the
            # divergence to sit on a clearly non-diagonal residual instead of
            # pretending to compare them.
            if abs(np.diag(ref_r)[0]) > 1e-6:
                compared += 1
                if mine_support is not None and mine_support != ref_support:
                    print(f"  support mismatch w={width} b={blocks} mask={mask}: "
                          f"{sorted(ref_support)} vs {sorted(mine_support)}")
                    bad_exclusions.append((width, blocks, mask))
            else:
                excluded += 1
                if off_diagonal(ref_r) < 1e-6:
                    bad_exclusions.append((width, blocks, mask, "diagonal yet excluded"))
            checked += 1
    print(f"  residuals checked          : {checked}")
    print(f"  max |fast - qiskit Operator|: {worst_residual:.3e}")
    print(f"  supports compared          : {compared}")
    print(f"  excluded (diag[0] ~ 0)     : {excluded}  "
          f"[all must be non-diagonal residuals]")
    print(f"  disagreements / bad excludes: {len(bad_exclusions)}")
    assert worst_residual < 1e-9, "fast residual disagrees with qiskit"
    assert not bad_exclusions, f"support test disagrees with the original: {bad_exclusions}"
    print("  fast_residual.py self-test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
