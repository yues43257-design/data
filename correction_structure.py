#!/usr/bin/env python3
"""What exactly is the relative-phase correction, and how expensive is it?

The RCCX (Margolus) ladder is certified *relative-phase* equivalent to the exact
ladder: ``U_exact^dagger U_rccx`` is diagonal with +-1 entries.  Whether the
43% T-count saving is **unconditional** or **algorithm dependent** depends
entirely on the structure of that diagonal:

* sign(x) = (-1)^(b + a.x)  (an *affine* sign pattern)  -> the correction is a
  product of single-qubit Z gates plus a global sign, i.e. **Clifford and free**.
  Then every block can be downgraded with a free correction: the saving is
  unconditional and needs no learning.
* otherwise (e.g. a sign that depends on a product of bits, i.e. a
  multi-controlled-Z-like correction) -> correcting costs T gates, so the
  downgrade is only usable where the correction is harmless for the final
  measurement.  *That* is a genuine, algorithm-dependent combinatorial problem
  -- and the only version of "global variant selection" worth learning.

Usage
-----
    .venv/bin/python correction_structure.py --max-controls 5
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_linux_mcx():
    loader = importlib.machinery.SourceFileLoader("linux_mcx_corr", str(ROOT / "Linux-MCX"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("cannot build an import spec for Linux-MCX")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402


def data_subspace(num_controls: int, total: int):
    n_ancillas = max(0, num_controls - 2)
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    return [x for x in range(2 ** total) if all(not (x >> a) & 1 for a in ancillas)]


def affine_sign_pattern(diagonal: np.ndarray, total: int) -> Optional[int]:
    """Return the linear mask ``a`` if ``sign(x) = (-1)^(b + a.x)``, else None.

    ``diagonal[x]`` is the correction on basis state ``x`` (0/1 per qubit).
    """
    signs = np.sign(diagonal.real)
    if not np.allclose(signs, diagonal.real) or np.any(np.abs(diagonal.imag) > 1e-9):
        return None
    # sign(x) must be +-1 and factorize as an affine function of the bits
    for x, value in enumerate(signs):
        expected = 1.0
        if x == 0:
            continue
    # brute force the affine model over the subspace: b from x=0, a from the units
    base_index = 0
    base = signs[base_index]
    mask = 0
    for qubit in range(total):
        index = 1 << qubit
        if index >= len(signs):
            continue
        ratio = signs[index] / base
        if abs(ratio - 1.0) < 1e-9:
            continue
        if abs(ratio + 1.0) < 1e-9:
            mask |= 1 << qubit
            continue
        return None
    # verify the affine model on every basis state
    for x, value in enumerate(signs):
        parity = bin(x & mask).count("1") & 1
        expected = -base if parity else base
        if abs(value - expected) > 1e-9:
            return None
    return mask


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Structure/cost of the relative-phase correction.")
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--max-controls", type=int, default=5)
    args = parser.parse_args(argv)

    print(
        f"{'gate':>5} | {'diag':>5} {'unit':>6} {'offdiag':>9} | "
        f"{'affine?':>7} | {'correction':>28} | verdict"
    )
    for num_controls in range(args.min_controls, args.max_controls + 1):
        exact = LINUX_MCX.make_mcx_source(num_controls, basis="clifford_t")
        rel = LINUX_MCX.zx.Circuit.from_qasm(
            qasm2.dumps(LINUX_MCX.relative_phase_mcx_circuit(num_controls))
        ).to_basic_gates()
        total = exact.qubits
        keep = data_subspace(num_controls, total)
        a = Operator(qasm2.loads(rel.to_qasm())).data[np.ix_(keep, keep)]
        b = Operator(qasm2.loads(exact.to_qasm())).data[np.ix_(keep, keep)]
        correction = b.conj().T @ a
        diagonal = np.diag(correction)
        offdiag = float(np.max(np.abs(correction - np.diag(diagonal))))
        unit = float(np.max(np.abs(np.abs(diagonal) - 1.0)))
        mask = affine_sign_pattern(diagonal, total) if offdiag < 1e-8 else None
        if mask is None:
            description = "NOT affine (needs T gates)"
            verdict = "algorithm-dependent (Strategy B is real)"
        else:
            qubits = [q for q in range(total) if (mask >> q) & 1]
            description = f"Z on qubits {qubits}" if qubits else "global phase only"
            verdict = "Clifford + free -> unconditional downgrade"
        print(
            f"C^{num_controls}X | {offdiag < 1e-8!s:>5} {unit:6.1e} {offdiag:9.1e} | "
            f"{str(mask is not None):>7} | {description:>28} | {verdict}"
        )
    print(
        "\n    [reading] 'affine' means the correction is a product of single-qubit Z\n"
        "    gates: it can be applied for free, so the relative-phase saving is\n"
        "    unconditional.  A non-affine correction costs T gates to repair, so the\n"
        "    downgrade is only usable where the correction does not disturb the\n"
        "    algorithm's final measurement -- the genuinely learnable problem."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
