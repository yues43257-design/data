#!/usr/bin/env python3
"""Incremental / symbolic safety verifier for the strategy-B variant policy.

The simulation verifier answers "is this exact/relative mix safe?" by rebuilding
the whole circuit and computing its unitary -- O(circuit) per candidate, which is
what stops the greedy decoder from scaling to circuits with dozens of blocks.

This module answers the same question with linear algebra that is set up **once
per instance**.  Write the circuit (blocks ``B_0..B_{k-1}``, inter-block Clifford
layers ``L_1..L_{k-1}``) as a product and let ``C`` be the diagonal correction
that turns one exact MCX block into its relative-phase twin (``B_rel = C · B``).
Because the correction sits immediately after the block, pushing it to the
measurement conjugates it by the *downstream* circuit and, for a set ``S`` of
downgraded blocks,

    U'(S) = Omega_S · U,      Omega_S = C~_{i1} · C~_{i2} · ...   (ascending i)
    C~_i  = Q_i · C · Q_i^dagger,   Q_i = (unitaries after block i)

where ``U`` is the all-exact circuit.  Both ``C~_i`` and ``U`` are built from the
per-block / per-layer unitaries in O(k) small matrix products; there is no
enumeration and no per-candidate simulation.  A mask is safe exactly when the
post-selected Z-basis distributions of ``Omega_S U`` match those of ``U`` -- the
same criterion the simulator uses, but now checked with a couple of matrix
products and a moduli comparison.

Validation: ``python symbolic_verifier.py --self-test`` checks that this
criterion agrees with the full simulation on *every* mask of many instances.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


GEN = _load("variant_generator_symbolic", "generate_variant_dataset.py")


def _unitary(circuit) -> np.ndarray:
    return np.asarray(Operator(circuit).data, dtype=complex)


class SymbolicVerifier:
    """O(1)-per-block safety oracle for one strategy-B instance.

    Construction cost is a handful of ``2^n x 2^n`` matrix products for the whole
    instance; every later ``is_safe`` / ``greedy`` call is another couple of
    products, independent of how many gates the instance contains.
    """

    def __init__(self, num_controls: int, blocks: int, seed: int, h_prob: float,
                 tolerance: float = 1e-9):
        self.num_controls = int(num_controls)
        self.blocks = int(blocks)
        self.seed = int(seed)
        self.h_prob = float(h_prob)
        self.tolerance = float(tolerance)
        self.total = num_controls + 1 + max(0, num_controls - 2)
        self.dim = 2 ** self.total
        self.data_dim = 2 ** (num_controls + 1)

        layers = GEN.build_instance_layers(num_controls, blocks, seed, h_prob)
        identity = np.eye(self.dim, dtype=complex)
        block = _unitary(GEN.exact_block(num_controls))
        relative = _unitary(GEN.relative_block(num_controls))
        correction = relative @ block.conj().T          # B_rel = correction . B

        def layer(index: int) -> np.ndarray:
            circuit = layers[index] if 0 <= index < len(layers) else None
            return _unitary(circuit) if circuit is not None else identity

        # all-exact circuit unitary, built forward
        unitary = block.copy()
        for index in range(1, blocks):
            unitary = layer(index) @ unitary
            unitary = block @ unitary
        self.unitary = unitary

        # suffix unitaries Q_i (everything after block i), built backward
        suffix = identity.copy()
        suffixes: Dict[int, np.ndarray] = {blocks - 1: suffix}
        for i in range(blocks - 2, -1, -1):
            suffix = suffix @ block @ layer(i + 1)
            suffixes[i] = suffix.copy()
        self.corrections = [
            suffixes[i] @ correction @ suffixes[i].conj().T for i in range(blocks)
        ]

        # post-selected Z-basis distribution of the all-exact circuit
        _controls, ancillas, _target = GEN.qubit_layout(num_controls)
        self.rows = np.array(
            [z for z in range(self.dim) if all(not (z >> a) & 1 for a in ancillas)],
            dtype=int,
        )
        self.cols = self.rows.copy()   # data inputs = control+target bits, ancillas 0
        self.reference = self._distributions(unitary)
        self._cache: Dict[int, bool] = {0: True}

    # -- core ---------------------------------------------------------------

    def omega(self, mask: int) -> np.ndarray:
        """Omega_S = C~_{i} ... in ascending index order (leftmost = largest)."""
        operator = np.eye(self.dim, dtype=complex)
        for i in range(self.blocks):
            if (mask >> i) & 1:
                operator = self.corrections[i] @ operator
        return operator

    def _distributions(self, matrix: np.ndarray) -> np.ndarray:
        sub = matrix[np.ix_(self.rows, self.cols)]
        prob = np.abs(sub) ** 2
        prob /= np.maximum(prob.sum(axis=0, keepdims=True), 1e-300)
        return prob

    def deviation(self, mask: int) -> float:
        """Max total-variation distance from the all-exact distribution."""
        if mask == 0:
            return 0.0
        prob = self._distributions(self.omega(mask) @ self.unitary)
        return float(np.max(0.5 * np.sum(np.abs(prob - self.reference), axis=0)))

    def is_safe(self, mask: int) -> bool:
        mask = int(mask)
        cached = self._cache.get(mask)
        if cached is None:
            cached = self.deviation(mask) <= self.tolerance
            self._cache[mask] = cached
        return cached

    # -- decoders -----------------------------------------------------------

    def mask_from_probs(self, probs: Sequence[float], threshold: float = 0.5) -> int:
        mask = 0
        for index, prob in enumerate(probs[: self.blocks]):
            if prob >= threshold:
                mask |= 1 << index
        return mask

    def greedy(self, probs: Sequence[float], threshold: float = 0.0) -> int:
        """Accept downgrades in confidence order, each only if still safe."""
        order = sorted(
            (i for i in range(self.blocks) if probs[i] >= threshold),
            key=lambda i: -probs[i],
        )
        accepted = 0
        for block in order:
            candidate = accepted | (1 << block)
            if self.is_safe(candidate):
                accepted = candidate
        return accepted

    def t_count(self, mask: int) -> int:
        t_exact, t_relative = GEN.block_t_counts(self.num_controls)
        downgraded = bin(mask).count("1")
        return (self.blocks - downgraded) * t_exact + downgraded * t_relative

    def oracle(self) -> int:
        """Brute-force best usable T-count (only for small ``blocks``)."""
        best = self.blocks * GEN.block_t_counts(self.num_controls)[0]
        for mask in range(1 << self.blocks):
            value = self.t_count(mask)
            if value < best and self.is_safe(mask):
                best = value
        return best


def _rccx_ladder(num_controls: int):
    """``(a, b, c)`` of every RCCX in the relative-phase ladder, in gate order."""
    n_ancillas = max(0, num_controls - 2)
    controls = list(range(num_controls))
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    target = num_controls + n_ancillas
    if num_controls == 1:
        return []
    if num_controls == 2:
        return [(controls[0], controls[1], target)]
    ladder = [(controls[0], controls[1], ancillas[0])]
    for i in range(1, n_ancillas):
        ladder.append((ancillas[i - 1], controls[i + 1], ancillas[i]))
    ladder.append((ancillas[-1], controls[-1], target))
    for i in reversed(range(1, n_ancillas)):
        ladder.append((ancillas[i - 1], controls[i + 1], ancillas[i]))
    ladder.append((controls[0], controls[1], ancillas[0]))
    return ladder


def block_correction_data(num_controls: int) -> np.ndarray:
    """Diagonal ``D`` with ``B_relative = D . B_exact`` on the clean-ancilla data space.

    Each RCCX is a Toffoli followed by the *2-local diagonal* correction
    ``phi(a, b, c) = (-1)**(a*c) * (-i)**(a*b)`` (measured directly from the gate
    sequence).  The ladder is a permutation, so ``D`` is read off by a purely
    classical reversible simulation -- O(2**(nc+1) * #gates) time and no dense
    2**n matrices at all.
    """
    controls, ancillas, target = GEN.qubit_layout(num_controls)
    total = num_controls + 1 + len(ancillas)
    data_qubits = controls + [target]
    ladder = _rccx_ladder(num_controls)
    size = 2 ** len(data_qubits)
    diagonal = np.ones(size, dtype=complex)
    for y in range(size):
        bits = [0] * total
        for i, q in enumerate(data_qubits):
            bits[q] = (y >> i) & 1
        phase = 1.0 + 0.0j
        for a, b, c in ladder:
            bits[c] ^= bits[a] & bits[b]
            va, vb, vc = bits[a], bits[b], bits[c]
            if va and vb:
                phase *= -1j
            if va and vc:
                phase *= -1.0
        # ``B_relative = D . B_exact``, so the phase picked up by input ``y``
        # belongs at the *output* index ``B_exact(y)`` (the final ladder state).
        permuted = 0
        for i, q in enumerate(data_qubits):
            if bits[q]:
                permuted |= 1 << i
        diagonal[permuted] = phase
    return diagonal


class ReducedSymbolicVerifier:
    """Exact criterion, but in the ``2**(nc+1)``-dimensional *data* subspace.

    The full verifier works with ``2**n x 2**n`` matrices (``n = controls + 1 +
    ancillas``).  Every MCX block leaves the clean ancillas in ``|0>`` at its
    boundary and every inter-block layer touches only the data qubits, so the
    whole criterion actually lives on the data register -- controls plus the
    target -- of dimension ``d = 2**(nc + 1)``.  Working there removes the
    ancillas from the linear algebra completely: the per-product cost drops from
    ``2**(3n)`` to ``d**3 = 2**(3(nc+1))``, i.e. a factor ``2**(3a)``
    (``a`` = number of ancillas) less work, and a factor ``2**(2a)`` less memory.

    Everything is exact and validated against the full simulation; the block's
    data-space action is read off the real relative-phase circuit, and the
    inter-block layers are rebuilt as small data-register circuits.
    """

    def __init__(self, num_controls: int, blocks: int, seed: int, h_prob: float,
                 tolerance: float = 1e-9):
        self.num_controls = int(num_controls)
        self.blocks = int(blocks)
        self.seed = int(seed)
        self.h_prob = float(h_prob)
        self.tolerance = float(tolerance)
        controls, ancillas, target = GEN.qubit_layout(num_controls)
        self.total = num_controls + 1 + len(ancillas)
        data_qubits = controls + [target]
        self.data_qubits = data_qubits
        self.d = 2 ** len(data_qubits)
        self.target_bit = num_controls
        control_mask = (1 << num_controls) - 1

        keep = []
        for y in range(self.d):
            full = 0
            for i, q in enumerate(data_qubits):
                if (y >> i) & 1:
                    full |= 1 << q
            keep.append(full)
        self.keep = np.array(keep, dtype=int)

        # exact block, on the data register, is the MCX permutation
        self.block_exact = np.zeros((self.d, self.d), dtype=complex)
        for y in range(self.d):
            fired = 1 if (y & control_mask) == control_mask else 0
            self.block_exact[y ^ (fired << self.target_bit), y] = 1.0

        # relative block restricted to the clean-ancilla subspace, built
        # analytically (no dense 2**n matrices needed)
        diagonal = block_correction_data(num_controls)
        self.block_relative = np.diag(diagonal) @ self.block_exact
        self.correction = np.diag(diagonal)

        layers = GEN.build_instance_layers(num_controls, blocks, seed, h_prob)

        def layer_matrix(index: int) -> np.ndarray:
            circuit = layers[index] if 0 <= index < len(layers) else None
            if circuit is None:
                return np.eye(self.d, dtype=complex)
            return self._data_layer(circuit)

        unitary = self.block_exact.copy()
        for index in range(1, blocks):
            unitary = layer_matrix(index) @ unitary
            unitary = self.block_exact @ unitary
        self.unitary = unitary

        identity = np.eye(self.d, dtype=complex)
        suffix = identity.copy()
        suffixes: Dict[int, np.ndarray] = {blocks - 1: suffix}
        for i in range(blocks - 2, -1, -1):
            suffix = suffix @ self.block_exact @ layer_matrix(i + 1)
            suffixes[i] = suffix.copy()
        self.corrections = [
            suffixes[i] @ self.correction @ suffixes[i].conj().T for i in range(blocks)
        ]
        self.reference = np.abs(unitary) ** 2
        self._cache: Dict[int, bool] = {0: True}

    def _data_layer(self, circuit) -> np.ndarray:
        """Rebuild an inter-block layer as a small data-register circuit."""
        from qiskit import QuantumCircuit as _QC
        remap = {q: i for i, q in enumerate(self.data_qubits)}
        data_circuit = _QC(len(self.data_qubits))
        for instruction in circuit.data:
            name = instruction.operation.name
            qubits = [remap[circuit.find_bit(q).index] for q in instruction.qubits]
            if name == "cx":
                data_circuit.cx(qubits[0], qubits[1])
            elif name == "cz":
                data_circuit.cz(qubits[0], qubits[1])
            else:
                getattr(data_circuit, name)(qubits[0])
        return _unitary(data_circuit)

    def omega(self, mask: int) -> np.ndarray:
        operator = np.eye(self.d, dtype=complex)
        for i in range(self.blocks):
            if (mask >> i) & 1:
                operator = self.corrections[i] @ operator
        return operator

    def deviation(self, mask: int) -> float:
        """Max total-variation distance from the all-exact distribution.

        ``U`` restricted to the data register is unitary, so each column of
        ``|U|**2`` is already a normalised distribution and this is exactly the
        same quantity the simulator reports.
        """
        if mask == 0:
            return 0.0
        magnitudes = np.abs(self.omega(mask) @ self.unitary) ** 2
        return float(np.max(0.5 * np.sum(np.abs(magnitudes - self.reference), axis=0)))

    def deviation_table(self) -> "List[float]":
        """Deviation for *every* mask, sharing the partial products.

        ``Omega(mask)`` splits as ``Omega_high(hi) . Omega_low(lo)`` because every
        block index in the high half exceeds every index in the low half, so with
        ``W = Omega . U`` the whole table costs **one matrix product per mask**
        while only ``2 * 2**(blocks/2)`` partial products are held in memory (the
        naive recurrence would keep all ``2**blocks`` of them).  Returns a list
        indexed by mask.
        """
        size = 1 << self.blocks
        table = [0.0] * size
        reference = self.reference
        half = self.blocks // 2
        low_count = 1 << half
        high_count = 1 << (self.blocks - half)

        # low half (indices 0..half-1), already multiplied through by U
        low: "List[Optional[np.ndarray]]" = [None] * low_count
        low[0] = self.unitary
        for lo in range(1, low_count):
            high_bit = lo.bit_length() - 1
            low[lo] = self.corrections[high_bit] @ low[lo ^ (1 << high_bit)]

        # high half (indices half..blocks-1), pure product starting from identity
        high: "List[Optional[np.ndarray]]" = [None] * high_count
        high[0] = np.eye(self.d, dtype=complex)
        for hi in range(1, high_count):
            high_bit = hi.bit_length() - 1
            high[hi] = self.corrections[half + high_bit] @ high[hi ^ (1 << high_bit)]

        for hi in range(high_count):
            operator = high[hi]
            base = hi << half
            if hi == 0:
                for lo in range(low_count):
                    magnitudes = np.abs(low[lo]) ** 2
                    table[base | lo] = float(
                        np.max(0.5 * np.sum(np.abs(magnitudes - reference), axis=0))
                    )
            else:
                for lo in range(low_count):
                    product = operator @ low[lo]
                    magnitudes = np.abs(product) ** 2
                    table[base | lo] = float(
                        np.max(0.5 * np.sum(np.abs(magnitudes - reference), axis=0))
                    )
        return table

    def is_safe(self, mask: int) -> bool:
        mask = int(mask)
        cached = self._cache.get(mask)
        if cached is None:
            cached = self.deviation(mask) <= self.tolerance
            self._cache[mask] = cached
        return cached

    def mask_from_probs(self, probs: Sequence[float], threshold: float = 0.5) -> int:
        mask = 0
        for index, prob in enumerate(probs[: self.blocks]):
            if prob >= threshold:
                mask |= 1 << index
        return mask

    def greedy(self, probs: Sequence[float], threshold: float = 0.0) -> int:
        order = sorted(
            (i for i in range(self.blocks) if probs[i] >= threshold),
            key=lambda i: -probs[i],
        )
        accepted = 0
        for block in order:
            candidate = accepted | (1 << block)
            if self.is_safe(candidate):
                accepted = candidate
        return accepted


# ---------------------------------------------------------------------------
# self test: criterion must equal the full simulation on every mask
# ---------------------------------------------------------------------------


def _self_test(seed: int = 0) -> int:
    failures: List[str] = []
    cases = [(3, 4, 12345), (4, 3, 777), (2, 5, 999), (3, 3, 42), (4, 4, 2026)]
    total_checked = 0
    for num_controls, blocks, case_seed in cases:
        verifier = SymbolicVerifier(num_controls, blocks, case_seed, 0.7)
        reduced = ReducedSymbolicVerifier(num_controls, blocks, case_seed, 0.7)
        reference = GEN.build_instance(num_controls, blocks, 0, case_seed, 0.7)[0]
        reference_dist = GEN.measurement_distributions(reference, num_controls)
        mismatches = reduced_mismatches = 0
        for mask in range(1 << blocks):
            circuit = GEN.build_instance(num_controls, blocks, mask, case_seed, 0.7)[0]
            dist = GEN.measurement_distributions(circuit, num_controls)
            simulated = max(
                GEN.total_variation(reference_dist[b], dist[b]) for b in reference_dist
            ) <= verifier.tolerance
            symbolic = verifier.is_safe(mask)
            reduced_safe = reduced.is_safe(mask)
            total_checked += 1
            if simulated != symbolic:
                mismatches += 1
            if simulated != reduced_safe:
                reduced_mismatches += 1
        ok = mismatches == 0 and reduced_mismatches == 0
        failures.append(ok)
        print(f"  [{'ok  ' if ok else 'FAIL'}] C^{num_controls}X x {blocks} (seed {case_seed}): "
              f"{1 << blocks} masks, full mismatches={mismatches}, "
              f"reduced mismatches={reduced_mismatches}")
    print()
    if not all(failures):
        print("symbolic self-test FAILED")
        return 1
    print(f"symbolic self-test passed ({total_checked} masks matched the simulation exactly)")
    return 0


def _benchmark(blocks: int = 30, num_controls: int = 4, seed: int = 7) -> int:
    """Compare the symbolic verifier with the simulation on a big circuit."""
    verifier = SymbolicVerifier(num_controls, blocks, seed, 0.7)
    probs = [0.9 - 0.01 * i for i in range(blocks)]

    start = time.perf_counter()
    mask_symbolic = verifier.greedy(probs)
    symbolic_seconds = time.perf_counter() - start

    reference = GEN.build_instance(num_controls, blocks, 0, seed, 0.7)[0]
    reference_dist = GEN.measurement_distributions(reference, num_controls)

    def simulated_safe(mask: int) -> bool:
        circuit = GEN.build_instance(num_controls, blocks, mask, seed, 0.7)[0]
        dist = GEN.measurement_distributions(circuit, num_controls)
        return max(GEN.total_variation(reference_dist[b], dist[b]) for b in reference_dist) <= 1e-9

    start = time.perf_counter()
    accepted = 0
    for block in sorted(range(blocks), key=lambda i: -probs[i]):
        candidate = accepted | (1 << block)
        if simulated_safe(candidate):
            accepted = candidate
    simulated_seconds = time.perf_counter() - start

    print(f"  blocks={blocks}: symbolic greedy {symbolic_seconds * 1e3:.1f} ms "
          f"(mask matches simulation: {mask_symbolic == accepted}) | "
          f"simulation greedy {simulated_seconds:.2f} s "
          f"({simulated_seconds / max(symbolic_seconds, 1e-9):.0f}x slower)")
    return 0 if mask_symbolic == accepted else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Incremental symbolic verifier.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--blocks", type=int, default=30)
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.benchmark:
        return _benchmark(args.blocks)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
