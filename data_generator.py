#!/usr/bin/env python3
"""
Local-ZX data generator for GNN training on quantum circuits.

What this script produces
-------------------------
A dataset of *local ZX phase diagrams* derived from random 3-8 qubit
circuits, together with the circuit's phase polynomial (parity -> phase) and a
T-count label.  Three circuit families are covered:

``mcx``
    Multi-controlled X gates C^nX with n = 1..6 controls, random control
    subsets, random control polarities and random target placement
    (synthesised with Qiskit and transpiled into {H, CX, Rz} form, the same
    convention that ``Linux-MCX`` uses).

``clifford_t``
    Random Clifford+T circuits with 5..30 T gates (H, S, X, Z, CNOT, CZ
    dressing allowed).

``phase_poly``
    Random phase polynomials that are *reverse engineered* into a ZX diagram
    (CNOT parity ladder + Z phase gadget per term).

Every sample is simplified with ``pyzx`` (``clifford_simp`` followed by
``full_reduce``), the phase polynomial is extracted, a T-count label is
computed, and the result is stored as a ``torch_geometric.data.Data`` object:

``x``            node features: spider type, phase, degree, qubit support, ...
``edge_index``   COO edge index (both directions), ``edge_type`` simple/Hadamard
``y_parity``     (num_terms, max_qubits) int8 parity bitmasks, zero padded
``y_phase``      (num_terms,) int64 phase class of each term: ``k`` means
                 ``k * pi/16`` modulo 32 classes (the 32-way GNN head)
``y_block``      (num_terms,) int64 index of the phase-polynomial block
``y_tcount``     scalar int64 T-count label
``y_depth``      scalar float T-depth after one Steiner routing pass
``y_count``      scalar float number of phase-polynomial terms
``metadata``     circuit family, qubit count, optimality level, ... (dict)

Phase granularity is the 32 point grid of ``[0, 2*pi)`` (``pi/16`` per class),
matching ``PHASE_CLASSES = tuple(Fraction(k, 16) for k in range(32))`` in
``Linux-MCX``.  ``pi/8`` phases are therefore representable exactly (class 2).
The T-count model itself is defined in units of ``pi/4``; the finer labels are
projected back onto that grid for TODD, the exhaustive solver and the
``rounded_upper_bound`` bookkeeping, so T-count labels did not change.

How the T-count label is defined
--------------------------------
The label is the T-count of the *phase polynomial* attached to each sample
(Clifford gates are free; a parity term whose coefficient in units of pi/4 is
odd costs one T gate).  This is the model in which TODD [Heyfron & Campbell
2019] and t-par [Amy, Maslov & Mosca 2014] are defined.

Two backends are used:

* PyZX' TODD implementation: ``pyzx.optimize.todd_simp`` on each
  Hadamard-separated phase block, plus ``phase_block_optimize`` for the whole
  Clifford+T circuit (which also handles Clifford corrections).  A local
  TOpt/TODD binary can be plugged in with ``--topt-cmd``.
* an optional exact search (``--exact-search``) that runs Dijkstra's algorithm
  over residual phase functions.  It explores the full model (every parity
  term, every coefficient 1..7, even coefficients free) and therefore *proves*
  optimality whenever it terminates inside its node budget.

``--self-test`` checks that both agree on random small phase polynomials.

Optimality levels written into ``metadata['optimality_level']``:

``exact_optimal``
    The exact search terminated inside its budget and reproduced the reported
    value, so the label is a proven optimum for the phase polynomial model.
``todd_best_known``
    The label is the best value found by TODD (per block or whole circuit) or
    by summing per-block optima.  No optimality proof.
``rounded_upper_bound``
    The circuit contains rotations that are not multiples of pi/4 (e.g. the
    Qiskit C^nX synthesis).  Phases are rounded up to the next multiple of
    pi/4 and the exact optimum *of the rounded phase function* is reported, so
    the value is an upper bound for the original circuit.
``upper_bound``
    Fallback (naive per-term count only).

Usage
-----
    # quick sanity check of the whole pipeline
    python data_generator.py self-test

    # small run
    python data_generator.py generate --samples 200 --workers 8 --out-dir ./dataset

    # the full 50k set (train/val/test, 80/10/10)
    python data_generator.py generate --samples 50000 --workers 16 --out-dir ./dataset

    # use a local TOpt / TODD binary inside the TODD step
    python data_generator.py generate --samples 50000 --topt-cmd /path/to/TOpt

``data.y_depth`` is produced by a Steiner routing pass over the extracted
phase polynomial (``routed_t_depth``); set ``GenConfig(use_steiner_depth=False)``
to fall back to scheduling the direct CNOT-ladder synthesis, which is roughly
an order of magnitude faster and gave identical T-depths on the default
families.

Output files are ``train.pt``, ``val.pt`` and ``test.pt`` (plain lists of
``Data`` objects).  With a small ``--shard-size`` a split is written as
``train_000.pt`` ... instead, and ``python data_generator.py merge`` joins the
shards back into single files.

Code map (the framework, top to bottom)
---------------------------------------
1. ``GenConfig`` / ``SampleSpec`` -- every knob, plus the (seed, family, split)
   plan of one sample.
2. helpers -- deterministic seeds, phase units, character signs, phase vectors.
3. phase polynomial extraction -- parity tracking inside H-free blocks.
4. Steiner routing + T-depth scheduling (``_t_parallel_schedule``,
   ``steiner_route_phase_layers``, ``routed_t_depth``).
5. circuit generators -- ``gen_mcx``, ``gen_clifford_t``, ``gen_phase_poly``
   and the ``GENERATORS`` registry (add a family = add a function + an entry).
6. simplification and labelling -- ``simplify_circuit`` (clifford_simp +
   full_reduce) and ``label_tcount`` (TODD / exact search / external oracle).
7. featurisation -- ``graph_to_data`` turns the simplified ZX diagram into a
   PyG ``Data`` object.
8. workers and IO -- ``plan_specs``, ``build_sample``, ``run_generation``,
   ``merge_shards``.
9. ``_self_test`` + CLI.

Public API
----------
``GenConfig``                 all generation knobs
``run_generation(cfg)``       generate and write train/val/test
``build_sample(spec, cfg)``   one sample from the plan
``data_from_qasm(qasm, cfg)`` your own circuit, QASM input
``data_from_graph(g, cfg)``   your own circuit, ZX diagram input
``load_split(path)``          ``torch.load(..., weights_only=False)`` for PyG
``merge_shards(out_dir)``     join ``<split>_NNN.pt`` shards

    from data_generator import GenConfig, load_split, data_from_qasm
    train = load_split("dataset/train.pt")
    print(train[0].x.shape, int(train[0].y_tcount), train[0].metadata["optimality_level"])
    single = data_from_qasm(open("my_circuit.qasm").read())
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyzx as zx
from pyzx.circuit.gates import CNOT, ParityPhase, ZPhase, Gate

try:  # pragma: no cover - routing helpers ship with pyzx
    from pyzx.routing.architecture import FULLY_CONNECTED, create_architecture
    from pyzx.routing.cnot_mapper import ElimMode
    from pyzx.routing.phase_poly import (
        PhasePoly,
        RootHeuristic,
        RoutingMethod,
        SplitHeuristic,
        route_phase_poly,
    )

    _HAS_ROUTING = True
except Exception:  # pragma: no cover - keep the generator usable without them
    _HAS_ROUTING = False

try:  # pragma: no cover - optional dependency
    import torch
    from torch_geometric.data import Data

    _HAS_PYG = True
    _PYG_ERROR = ""
except Exception as exc:  # pragma: no cover
    _HAS_PYG = False
    _PYG_ERROR = repr(exc)
    torch = None  # type: ignore
    Data = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from qiskit import QuantumCircuit, transpile

    try:
        from qiskit import qasm2
    except ImportError:  # older Qiskit
        qasm2 = None  # type: ignore
    from qiskit.circuit.library import MCXGate

    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    _HAS_QISKIT = False


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

VERSION = "0.1.0"

FAMILY_MCX = "mcx"
FAMILY_CLIFFORD_T = "clifford_t"
FAMILY_PHASE_POLY = "phase_poly"
ALL_FAMILIES = (FAMILY_MCX, FAMILY_CLIFFORD_T, FAMILY_PHASE_POLY)
DEFAULT_FAMILY_WEIGHTS = {FAMILY_MCX: 0.35, FAMILY_CLIFFORD_T: 0.35, FAMILY_PHASE_POLY: 0.30}

LEVEL_EXACT = "exact_optimal"
LEVEL_TODD = "todd_best_known"
LEVEL_ROUNDED = "rounded_upper_bound"
LEVEL_UPPER = "upper_bound"
LEVEL_EXTERNAL = "external_best_known"
LEVELS = (LEVEL_EXACT, LEVEL_TODD, LEVEL_EXTERNAL, LEVEL_ROUNDED, LEVEL_UPPER)

PHASE_MODES = ("exact", "ceil", "round")

# Phase label granularity.  ``phase_to_units`` returns *class indices* on the
# 32 point grid of [0, 2*pi), i.e. one class is pi/16, taken modulo 32.  This is
# exactly the phase library used by the Linux-MCX GNN head
# (``PHASE_CLASSES = tuple(Fraction(k, 16) for k in range(32))``), so
# ``data.y_phase`` can be fed straight into a 32-way classifier.
PHASE_CLASSES = 32
CLASSES_PER_PI = PHASE_CLASSES // 2             # 16 classes per pi
CLASSES_PER_QUARTER_PI = CLASSES_PER_PI // 4    # pi/4 spans four classes

# Z-phases that count as a T/T-dagger gate in the scheduling pass
# (copied from the Linux-MCX scheduler).
T_PHASES = (Fraction(1, 4), Fraction(7, 4))

# node feature layout (kept in one place so downstream GNNs can rely on it)
NODE_FEATURES = (
    "is_z_spider",
    "is_x_spider",
    "is_boundary",
    "is_ground",
    "phase_over_pi",
    "phase_sin",
    "phase_cos",
    "phase_is_non_clifford",
    "degree_over_n",
    "support_size_over_n",
    "qubit_over_n",
)

# ``metadata`` is a plain dict attribute; PyG recurses into dicts when batching,
# so every sample must expose exactly the same key set (missing keys break
# ``Batch.from_data_list``).  Family specific entries are padded with ``None``.
METADATA_KEYS = (
    "sample_id",
    "seed",
    "family",
    "n_qubits",
    "input_format",
    "split",
    "generator_version",
    "optimality_level",
    "label_method",
    "y_tcount",
    "num_nodes",
    "num_edges",
    "num_terms",
    "n_blocks",
    "rounded_phases",
    "phase_source",
    "t_depth",
    "unsupported_gates",
    "tcount_n_blocks",
    "tcount_n_terms",
    "tcount_rounded",
    "tcount_naive",
    "tcount_todd_global",
    "tcount_todd_sum_of_blocks",
    "tcount_exact",
    "tcount_external",
    "exact_search_proved",
    "tcount_candidates_json",
    "gen_info_json",
    "graph_stats_json",
    "gen_n_controls",
    "gen_target",
    "gen_variant",
    "gen_n_t_requested",
    "gen_h_free",
    "gen_requested_terms",
    "gen_realised_terms",
    "qasm",
)


# ---------------------------------------------------------------------------
# configuration / containers
# ---------------------------------------------------------------------------


@dataclass
class GenConfig:
    """All knobs of the generator (picklable, handed to every worker)."""

    samples: int = 50_000
    families: Tuple[str, ...] = ALL_FAMILIES
    family_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FAMILY_WEIGHTS))
    min_qubits: int = 3
    max_qubits: int = 8
    seed: int = 20240914
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    # --- family: multi-controlled X
    mcx_min_controls: int = 1
    mcx_max_controls: int = 6
    mcx_padding_prob: float = 0.4          # probability of using extra idle qubits
    mcx_random_polarity_prob: float = 0.5  # probability of inverting some controls
    mcx_relative_phase_prob: float = 0.25  # n=2: use the 4-T Margolus variant
    mcx_opt_level: int = 1

    # --- family: random Clifford + T
    t_min: int = 5
    t_max: int = 30
    clifford_t_hfree_prob: float = 0.25    # H-free circuits are single-block phase polynomials
    clifford_t_dressing_prob: float = 0.35

    # --- family: random phase polynomial
    pp_min_terms: int = 1
    pp_max_terms: int = 6
    pp_min_parity_size: int = 2
    pp_max_parity_size: int = 6
    pp_native_zx_prob: float = 0.5         # report the sample as a native ZX diagram

    # --- simplification / labelling
    run_clifford_simp: bool = True
    run_full_reduce: bool = True
    phase_mode: str = "ceil"               # how to map non pi/4 phases onto pi/4 units
    use_todd: bool = True
    use_todd_blockwise: bool = True
    todd_max_gates: int = 400              # skip the (possibly slow) global TODD above this
    topt_cmd: Optional[str] = None         # external TOpt binary for pyzx' TODD step
    exact_search: bool = False             # prove optimality on small instances
    exact_max_qubits: int = 3
    exact_max_cost: int = 6
    exact_max_nodes: int = 20_000
    external_tcount_cmd: Optional[str] = None   # TOpt / t-par style CLI oracle
    external_tcount_exact: bool = False         # does that tool return proven optima?

    # --- T-depth label (Steiner routing + parallel T scheduling)
    use_steiner_depth: bool = True
    depth_route_attempts: int = 1             # extra randomized Steiner attempts
    depth_route_min_terms: int = 4            # tiny blocks are scheduled directly
    depth_route_max_terms: int = 64            # skip routing for very large blocks

    # --- output / runtime
    out_dir: str = "dataset"
    workers: int = 0                       # 0 -> min(cpu_count, 8)
    chunksize: int = 16
    shard_size: int = 100_000
    store_qasm: bool = True
    retries: int = 3
    max_nodes: int = 4000
    progress_every: int = 500

    def normalized_workers(self) -> int:
        if self.workers and self.workers > 0:
            return self.workers
        return max(1, min(os.cpu_count() or 1, 8))


@dataclass(frozen=True)
class SampleSpec:
    """Deterministic description of one sample (seed, family, split)."""

    index: int
    seed: int
    family: str
    split: str


@dataclass
class GeneratedSample:
    """Raw output of a family generator."""

    circuit: zx.Circuit
    n_qubits: int
    family: str
    input_format: str
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TCountLabel:
    value: int
    level: str
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def derive_seed(base_seed: int, index: int) -> int:
    """Stable per-sample seed: independent of worker count and result order."""
    ss = np.random.SeedSequence([int(base_seed) & 0xFFFFFFFF, int(index) & 0xFFFFFFFF])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def largest_remainder(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Turn fractional weights into integer counts summing to ``total``."""
    total_w = float(sum(weights.values())) or 1.0
    raw = {k: total * float(w) / total_w for k, w in weights.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    missing = total - sum(out.values())
    order = sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)
    for i in range(missing):
        out[order[i % len(order)]] += 1
    return out


def phase_to_units(phase: Any, mode: str = "ceil", tol: float = 1e-6) -> Tuple[int, bool]:
    """Map a phase (in units of pi) onto one of the 32 phase classes.

    The class index ``k`` encodes the phase ``k * pi/16`` modulo ``32`` classes
    covering ``[0, 2*pi)``, which is the label space of the GNN head
    (``Fraction(k, 16)`` for ``k`` in ``0..31``).  ``pi/8`` phases land on even
    classes, ``pi/4`` phases on multiples of four.

    Returns ``(class_index, exact_pi4)``.  The second element keeps the
    original meaning used by the T-count machinery: it is True only when the
    phase is an exact multiple of pi/4, i.e. Clifford+T representable, so that
    the T-count label stays a *proof* rather than a rounded upper bound.
    """
    if mode not in PHASE_MODES:
        raise ValueError(f"unknown phase mode {mode!r}")
    frac = Fraction(phase) % 2
    scaled = float(frac) * CLASSES_PER_PI  # class index within [0, 32)
    nearest = int(round(scaled))
    if abs(scaled - nearest) <= tol:
        return nearest % PHASE_CLASSES, nearest % CLASSES_PER_QUARTER_PI == 0
    if mode == "ceil":
        units = int(math.ceil(scaled - 1e-9))
    else:
        units = int(round(scaled))
    return units % PHASE_CLASSES, False


def pi4_units_of_phase(phase: Any, mode: str = "ceil", tol: float = 1e-6) -> Tuple[int, bool]:
    """Legacy pi/4 view of a phase (units of pi/4 modulo 8) plus exactness.

    The T-count model, TODD and the exhaustive solver all work in units of
    pi/4, so the finer 32-class labels are converted back here.  This is the
    arithmetic the generator used before the phase grid was refined, which
    keeps the T-count labels and optimality levels unchanged.
    """
    if mode not in PHASE_MODES:
        raise ValueError(f"unknown phase mode {mode!r}")
    frac = Fraction(phase) % 2
    scaled = float(frac) * 4.0  # units of pi/4 within [0, 8)
    nearest = int(round(scaled))
    if abs(scaled - nearest) <= tol:
        return nearest % 8, True
    if mode == "ceil":
        units = int(math.ceil(scaled - 1e-9))
    else:
        units = int(round(scaled))
    return units % 8, False


def phase_terms_to_pi4(terms: Dict[int, int], mode: str = "ceil") -> Dict[int, int]:
    """Convert 32-class phase terms ``{parity: class}`` to pi/4 units mod 8."""
    converted: Dict[int, int] = defaultdict(int)
    for parity, class_index in terms.items():
        class_index %= PHASE_CLASSES
        if class_index % CLASSES_PER_QUARTER_PI == 0:
            units = (class_index // CLASSES_PER_QUARTER_PI) % 8
        else:
            scaled = class_index / CLASSES_PER_QUARTER_PI  # pi/4 units
            if mode == "ceil":
                units = int(math.ceil(scaled - 1e-9))
            else:
                units = int(round(scaled))
            units %= 8
        converted[parity] = (converted[parity] + units) % 8
    return {p: c for p, c in converted.items() if c % 8}


def gate_phase_units(gate: Gate) -> Tuple[int, bool]:
    """Phase class (pi/16 grid, mod 32) of a single-qubit phase gate.

    The boolean is True when the gate phase is also a multiple of pi/4.
    """
    name = gate.name
    if name == "Z":
        return 16, True          # pi   = 16 * pi/16
    if name == "S":
        return 8, True           # pi/2 =  8 * pi/16
    if name == "T":
        return 4, True           # pi/4 =  4 * pi/16
    if isinstance(gate, ZPhase):
        return phase_to_units(gate.phase, "exact")
    return 0, True


def tcount_of_gates(gates: Iterable[Gate], mode: str = "ceil") -> int:
    """Number of T/T-dagger gates needed to write a gate list verbatim."""
    total = 0
    for gate in gates:
        name = gate.name
        if name in ("CNOT", "CZ", "H", "HAD", "SWAP", "NOT", "Z", "S"):
            continue
        if name == "T":
            total += 1
            continue
        phase = getattr(gate, "phase", None)
        if isinstance(gate, ZPhase) or isinstance(gate, ParityPhase) or phase is not None:
            units, exact = pi4_units_of_phase(phase, mode)
            total += (units % 2) if exact else max(1, int(units))
    return int(total)


_SIGN_CACHE: Dict[int, List[Tuple[int, ...]]] = {}


def character_signs(n_qubits: int) -> List[Tuple[int, ...]]:
    """``signs[p][x] = (-1)^popcount(p & x)`` for every parity ``p``."""
    cached = _SIGN_CACHE.get(n_qubits)
    if cached is not None:
        return cached
    size = 1 << n_qubits
    idx = np.arange(size, dtype=np.int64)
    table = [tuple((1 - 2 * (np.bitwise_count(idx & p) & 1)).tolist()) for p in range(size)]
    _SIGN_CACHE[n_qubits] = table
    return table


def phase_function_from_terms(terms: Dict[int, int], n_qubits: int) -> np.ndarray:
    """Z8 valued phase function ``r(x) = sum_p c_p (-1)^(p.x)`` (values 0..7)."""
    size = 1 << n_qubits
    idx = np.arange(size, dtype=np.int64)
    values = np.zeros(size, dtype=np.int64)
    for parity, units in terms.items():
        units %= 8
        if parity == 0 or units == 0:
            continue
        values += units * (1 - 2 * (np.bitwise_count(idx & parity) & 1))
    return values % 8


def exact_tcount_search(
    terms: Dict[int, int],
    n_qubits: int,
    max_cost: int = 6,
    max_nodes: int = 20_000,
) -> Tuple[Optional[int], bool]:
    """Exact minimum T-count of a phase polynomial, or ``(None, False)``.

    The phase polynomial is realised by adding parity terms until the residual
    phase function vanishes.  A term whose coefficient (in units of pi/4) is
    odd costs one T gate, an even coefficient is free.  This is Dijkstra's
    algorithm over residual phase functions: states are popped in
    non-decreasing cost, so the first time the zero residual is popped its cost
    is optimal.  Returns ``(value, proved)`` where ``proved`` is False when the
    node budget ran out.
    """
    import heapq

    if n_qubits > 8:
        return None, False
    size = 1 << n_qubits
    start = tuple(int(v) for v in phase_function_from_terms(terms, n_qubits))
    if not any(start):
        return 0, True
    table = character_signs(n_qubits)
    moves = [(table[p], c, c % 2) for p in range(1, size) for c in range(1, 8)]

    best: Optional[int] = None
    distances: Dict[Tuple[int, ...], int] = {start: 0}
    heap: List[Tuple[int, Tuple[int, ...]]] = [(0, start)]
    nodes = 0
    while heap:
        cost, residual = heapq.heappop(heap)
        if cost != distances.get(residual):
            continue
        nodes += 1
        if nodes > max_nodes:
            return best, False
        if not any(residual):
            return cost, True
        if cost >= max_cost:
            continue
        for signs, coefficient, step_cost in moves:
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            nxt = tuple((r - coefficient * s) % 8 for r, s in zip(residual, signs))
            if new_cost < distances.get(nxt, 1 << 30):
                distances[nxt] = new_cost
                heapq.heappush(heap, (new_cost, nxt))
    return best, False


# ---------------------------------------------------------------------------
# phase polynomial extraction
# ---------------------------------------------------------------------------


@dataclass
class PhasePolynomial:
    """Parity -> phase map, split into Hadamard separated blocks."""

    blocks: List[Dict[int, int]] = field(default_factory=list)
    n_qubits: int = 0
    all_exact: bool = True
    unsupported: List[str] = field(default_factory=list)
    source: str = "circuit"

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def n_terms(self) -> int:
        return sum(len(block) for block in self.blocks)

    def flat_terms(self) -> Dict[int, int]:
        """Block-oblivious merge (used for statistics / cross-checks)."""
        merged: Dict[int, int] = defaultdict(int)
        for block in self.blocks:
            for parity, units in block.items():
                merged[parity] = (merged[parity] + units) % PHASE_CLASSES
        return {p: c for p, c in merged.items() if c % PHASE_CLASSES}


def split_into_blocks(gates: Sequence[Gate]) -> List[List[Gate]]:
    """Split a circuit into maximal runs that contain no Hadamard gate."""
    blocks: List[List[Gate]] = []
    current: List[Gate] = []
    for gate in gates:
        if gate.name in ("H", "HAD"):
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(gate)
    if current:
        blocks.append(current)
    return blocks


def extract_phase_polynomial(
    circuit: zx.Circuit,
    phase_mode: str = "ceil",
    source: str = "circuit",
) -> PhasePolynomial:
    """Track parities through CNOT/CZ and collect Z phases per H-free block.

    Signs from X gates inside a block are tracked so that emitted phases are
    correct up to a global phase.  Gates outside the phase-polynomial fragment
    (XPhase/YPhase/...) are reported in ``unsupported``; callers then fall back
    to extracting from the un-simplified circuit.
    """
    n_qubits = circuit.qubits
    result = PhasePolynomial(n_qubits=n_qubits, source=source)
    parity = [1 << q for q in range(n_qubits)]
    signs = [1] * n_qubits
    current: Dict[int, int] = defaultdict(int)

    def flush() -> None:
        if current:
            cleaned = {
                p: c % PHASE_CLASSES for p, c in current.items() if c % PHASE_CLASSES
            }
            if cleaned:
                result.blocks.append(cleaned)
        current.clear()

    for gate in circuit.gates:
        name = gate.name
        if name in ("H", "HAD"):
            flush()
        elif name == "CNOT":
            parity[gate.target] ^= parity[gate.control]
        elif name == "CZ":
            pc, pt = parity[gate.control], parity[gate.target]
            half_pi = CLASSES_PER_PI // 2  # pi/2 in phase classes
            for par in (pc, pt, pc ^ pt):
                current[par] = (current[par] + half_pi) % PHASE_CLASSES
        elif name == "SWAP":
            a, b = gate.control, gate.target
            parity[a], parity[b] = parity[b], parity[a]
            signs[a], signs[b] = signs[b], signs[a]
        elif name == "NOT":
            signs[gate.target] *= -1
        elif name in ("Z", "S", "T") or isinstance(gate, ZPhase):
            units, exact = gate_phase_units(gate)
            if not exact:
                units, _ = phase_to_units(gate.phase, phase_mode)
            result.all_exact &= exact
            if units % PHASE_CLASSES:
                target = parity[gate.target]
                current[target] = (
                    current[target] + units * signs[gate.target]
                ) % PHASE_CLASSES
        else:
            result.unsupported.append(name)
    flush()
    return result


def terms_to_gate_block(terms: Dict[int, int], n_qubits: int) -> List[Gate]:
    """Synthesise a block as CNOT parity ladder + ZPhase gates.

    ``terms`` carries phases in units of ``pi/4`` modulo 8: this is the
    T-count model view used by TODD and the exhaustive solver.
    """
    gates: List[Gate] = []
    for parity, units in sorted(terms.items()):
        bits = [q for q in range(n_qubits) if parity >> q & 1]
        if not bits:
            continue
        target = bits[-1]
        for q in bits[:-1]:
            gates.append(CNOT(q, target))
        units %= 8
        if units:
            signed = units if units <= 4 else units - 8
            gates.append(ZPhase(target, Fraction(signed, 4)))
        for q in reversed(bits[:-1]):
            gates.append(CNOT(q, target))
    return gates


def class_terms_to_gate_block(terms: Dict[int, int], n_qubits: int) -> List[Gate]:
    """Synthesise a block from 32-class phases (pi/16 grid) as CNOT + ZPhase.

    Unlike :func:`terms_to_gate_block` this keeps the fine phase resolution, so
    the Steiner routing / T-depth pass sees exactly the phases written to
    ``data.y_phase``.
    """
    gates: List[Gate] = []
    for parity, units in sorted(terms.items()):
        bits = [q for q in range(n_qubits) if parity >> q & 1]
        if not bits:
            continue
        target = bits[-1]
        for q in bits[:-1]:
            gates.append(CNOT(q, target))
        units %= PHASE_CLASSES
        if units:
            signed = units if units <= PHASE_CLASSES // 2 else units - PHASE_CLASSES
            gates.append(ZPhase(target, Fraction(signed, CLASSES_PER_PI)))
        for q in reversed(bits[:-1]):
            gates.append(CNOT(q, target))
    return gates


def todd_tcount_of_terms(terms: Dict[int, int], n_qubits: int) -> Optional[int]:
    """Run PyZX' TODD implementation (optionally delegating to TOpt) on a block."""
    gates = terms_to_gate_block(terms, n_qubits)
    if not gates:
        return 0
    try:
        optimised, _ = zx.optimize.todd_simp(gates, n_qubits, quiet=True)
    except Exception:
        return None
    return tcount_of_gates(optimised, "ceil")


def is_clifford_t(circuit: zx.Circuit) -> bool:
    """True when every gate is Clifford or a multiple of pi/4 rotation."""
    for gate in circuit.gates:
        name = gate.name
        if name in ("H", "HAD", "S", "Z", "T", "CNOT", "CZ", "NOT", "SWAP"):
            continue
        if isinstance(gate, ZPhase):
            _, exact = pi4_units_of_phase(gate.phase, "exact")
            if exact:
                continue
        return False
    return True


def todd_global_tcount(circuit: zx.Circuit, max_gates: int) -> Optional[int]:
    """Whole-circuit TODD + parity-network resynthesis (best known value)."""
    if len(circuit.gates) > max_gates or not is_clifford_t(circuit):
        return None
    try:
        optimised = zx.optimize.phase_block_optimize(circuit, quiet=True)
    except Exception:
        return None
    return tcount_of_gates(optimised.gates, "ceil")


def external_tcount(circuit: zx.Circuit, cmd: str, timeout: float = 60.0) -> Optional[int]:
    """Ask an external T-count tool (TOpt, t-par, ...) for a T-count label.

    Contract: the QASM of the *simplified* circuit is written to the tool's
    stdin and the first integer on stdout is taken as the T-count.  Any error
    (tool missing, non-zero exit, timeout, no integer) degrades to ``None`` so
    that generation keeps running with the built-in backends.
    """
    if not cmd:
        return None
    try:
        proc = subprocess.run(
            cmd.split(),
            input=circuit.to_qasm(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    match = re.search(r"-?\d+", proc.stdout or "")
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# Steiner phase routing + T-depth scheduling
#
# Ported verbatim from the MCX script so that ``y_depth`` uses exactly the
# scheduling/routing convention the GNN was trained with.
# ---------------------------------------------------------------------------


def _circuit_block(n_qubits: int, gates: Iterable[Gate]) -> zx.Circuit:
    circuit = zx.Circuit(n_qubits)
    circuit.gates = list(gates)
    return circuit


def _is_t_gate(gate: Gate) -> bool:
    return isinstance(gate, ZPhase) and (Fraction(gate.phase) % 2 in T_PHASES)


def _gate_qubits(gate: Gate) -> Tuple[int, ...]:
    if isinstance(gate, CNOT):
        return (gate.control, gate.target)
    if isinstance(gate, ZPhase):
        return (gate.target,)
    name = gate.name
    if name in ("H", "HAD", "NOT"):
        return (gate.target,)
    if name in ("CZ", "SWAP"):
        return (gate.control, gate.target)
    return ()


def _t_parallel_schedule(circuit: zx.Circuit) -> Tuple[int, List[Tuple[int, Gate]]]:
    """Assign each T/T-dagger gate to an earliest parallel layer.

    Only T gates consume layers; commuting gates on different qubits share a
    layer.  This layer assignment is the scheduler whose result is reported as
    T-depth and is later fed into the GNN objective.
    """
    last_depth = [0] * circuit.qubits
    layers: List[Tuple[int, Gate]] = []
    for gate in circuit.gates:
        qubits = _gate_qubits(gate)
        if not qubits:
            continue
        start = max(last_depth[q] for q in qubits)
        if _is_t_gate(gate):
            start += 1
            layers.append((start, gate))
        for q in qubits:
            last_depth[q] = start
    return (max((layer for layer, _ in layers), default=0), layers)


def compute_t_depth(circuit: zx.Circuit) -> int:
    """Scheduled T-depth of a routed circuit."""
    return _t_parallel_schedule(circuit)[0]


def _count_cnots(circuit: zx.Circuit) -> int:
    return sum(1 for gate in circuit.gates if isinstance(gate, CNOT))


def _best_steiner_route(phase_poly: "PhasePoly", block_seed: int, attempts: int = 6) -> zx.Circuit:
    """Randomized Steiner-tree phase-polynomial routing.

    For each phase layer we run several Steiner-mode Gray/Meijer attempts on a
    fully connected architecture and keep the candidate that minimizes the
    FTQC objective ``(T-depth, CX-count)``.
    """
    n_qubits = phase_poly.out_par[0].n_qubits()
    architecture = create_architecture(FULLY_CONNECTED, n_qubits=n_qubits)
    best: Optional[zx.Circuit] = None
    best_score: Optional[Tuple[int, int]] = None

    for method in (RoutingMethod.GRAY_MEIJER, RoutingMethod.MEIJER):
        for attempt in range(attempts):
            np.random.seed(20_260_909 + block_seed * 100 + attempt)
            try:
                candidate = route_phase_poly(
                    phase_poly,
                    architecture,
                    method=method,
                    mode=ElimMode.STEINER_MODE,
                    root_heuristic=RootHeuristic.RECURSIVE,
                    split_heuristic=SplitHeuristic.RANDOM,
                )
            except Exception:
                continue
            metrics = (compute_t_depth(candidate), _count_cnots(candidate))
            if best_score is None or metrics < best_score:
                best = candidate
                best_score = metrics

    if best is None:
        # Last-resort Meijer+Steiner fallback if the randomized attempts all fail.
        best = route_phase_poly(
            phase_poly,
            architecture,
            method=RoutingMethod.MEIJER,
            mode=ElimMode.STEINER_MODE,
        )
    return best


def steiner_route_phase_layers(
    circuit: zx.Circuit,
    attempts: int = 6,
    min_terms: int = 1,
    max_terms: Optional[int] = None,
) -> zx.Circuit:
    """Route CNOT+ZPhase phase layers with the low-CNOT Steiner heuristic.

    Hadamard boundaries are preserved; each maximal CNOT+ZPhase block is
    replaced by :func:`_best_steiner_route`.  Blocks with fewer than
    ``min_terms`` or more than ``max_terms`` phase terms are kept as they are
    (their routing round trip costs more than it can save).
    """
    n_qubits = circuit.qubits
    out = zx.Circuit(n_qubits)
    current: List[Gate] = []
    block_id = 0

    def flush() -> None:
        nonlocal current, block_id
        if not current:
            return
        block = _circuit_block(n_qubits, current)
        n_terms = sum(1 for gate in current if isinstance(gate, ZPhase))
        eligible = (
            _HAS_ROUTING
            and all(isinstance(g, (CNOT, ZPhase)) for g in current)
            and n_terms >= min_terms
            and (max_terms is None or n_terms <= max_terms)
        )
        if eligible:
            try:
                phase_poly = PhasePoly.fromCircuit(block)
                routed = _best_steiner_route(phase_poly, block_id, attempts=attempts)
                block_id += 1
            except Exception:
                routed = block
        else:
            routed = block
        for gate in routed.gates:
            out.add_gate(gate)
        current = []

    for gate in circuit.gates:
        if gate.name in ("H", "HAD"):
            flush()
            out.add_gate(gate)
        else:
            current.append(gate)
    flush()
    return out


def phase_polynomial_circuit(polynomial: PhasePolynomial) -> zx.Circuit:
    """Synthesise the extracted 32-class phase polynomial as CNOT + ZPhase.

    All blocks are concatenated in order; the scheduler recovers the original
    Hadamard-boundary dependencies from the qubit usage of each block.
    """
    circuit = zx.Circuit(polynomial.n_qubits)
    for block in polynomial.blocks:
        for gate in class_terms_to_gate_block(block, polynomial.n_qubits):
            circuit.add_gate(gate)
    return circuit


def routed_t_depth(polynomial: PhasePolynomial, cfg: GenConfig) -> float:
    """Steiner-route the extracted phase polynomial and schedule its T-depth.

    The extracted phase polynomial is re-synthesised as CNOT + ZPhase and passed
    through one ``steiner_route_phase_layers`` pass (randomized Steiner routing,
    skipped for trivial/oversized blocks), then scheduled with
    ``_t_parallel_schedule``.  With ``GenConfig.use_steiner_depth = False`` the
    same scheduler is applied to the direct CNOT-ladder synthesis instead
    (much faster, slightly higher T-depth).
    """
    circuit = phase_polynomial_circuit(polynomial)
    if not circuit.gates:
        return 0.0
    if cfg.use_steiner_depth and _HAS_ROUTING:
        try:
            circuit = steiner_route_phase_layers(
                circuit,
                attempts=cfg.depth_route_attempts,
                min_terms=cfg.depth_route_min_terms,
                max_terms=cfg.depth_route_max_terms,
            )
        except Exception:
            pass
    return float(compute_t_depth(circuit))


# ---------------------------------------------------------------------------
# circuit generators
# ---------------------------------------------------------------------------


def _dump_qasm(circuit: "QuantumCircuit") -> str:
    if qasm2 is not None:
        return qasm2.dumps(circuit)
    return circuit.qasm()  # pragma: no cover - very old Qiskit


def _qiskit_mcx_circuit(
    n_qubits: int,
    controls: Sequence[int],
    target: int,
    polarities: Sequence[bool],
    opt_level: int,
) -> zx.Circuit:
    """C^nX on a random qubit subset, returned as a PyZX {H, CX, Rz} circuit."""
    if not _HAS_QISKIT:
        raise RuntimeError("qiskit is required for the mcx family")
    qc = QuantumCircuit(n_qubits)
    for qubit, polarity in zip(controls, polarities):
        if not polarity:
            qc.x(qubit)
    qc.append(MCXGate(len(controls)), list(controls) + [target])
    for qubit, polarity in zip(controls, polarities):
        if not polarity:
            qc.x(qubit)
    transpiled = transpile(qc, basis_gates=["h", "cx", "rz"], optimization_level=opt_level)
    return zx.Circuit.from_qasm(_dump_qasm(transpiled))


def _relative_phase_toffoli(n_qubits: int, c0: int, c1: int, target: int) -> zx.Circuit:
    """4-T Margolus / RCCX relative phase Toffoli (exact Clifford+T)."""
    if not _HAS_QISKIT:
        raise RuntimeError("qiskit is required for the mcx family")
    qc = QuantumCircuit(n_qubits)
    qc.h(target)
    qc.t(target)
    qc.cx(c1, target)
    qc.tdg(target)
    qc.cx(c0, target)
    qc.t(target)
    qc.cx(c1, target)
    qc.tdg(target)
    qc.h(target)
    return zx.Circuit.from_qasm(_dump_qasm(qc)).to_basic_gates()


def gen_mcx(rng: random.Random, cfg: GenConfig) -> GeneratedSample:
    """Random multi-controlled X gate C^nX, n = 1..6."""
    max_controls = min(cfg.mcx_max_controls, max(cfg.mcx_min_controls, cfg.max_qubits - 1))
    n_controls = rng.randint(cfg.mcx_min_controls, max_controls)
    required = n_controls + 1
    if required > cfg.max_qubits:
        n_controls = cfg.max_qubits - 1
        required = n_controls + 1

    n_qubits = max(required, cfg.min_qubits)
    if n_qubits < cfg.max_qubits and rng.random() < cfg.mcx_padding_prob:
        n_qubits = rng.randint(n_qubits, cfg.max_qubits)

    qubits = rng.sample(range(n_qubits), required)
    controls = qubits[:n_controls]
    target = qubits[n_controls]
    polarities = [True] * n_controls
    if rng.random() < cfg.mcx_random_polarity_prob:
        for i in range(n_controls):
            if rng.random() < 0.5:
                polarities[i] = False

    use_relative_phase = n_controls == 2 and rng.random() < cfg.mcx_relative_phase_prob
    if use_relative_phase:
        circuit = _relative_phase_toffoli(n_qubits, controls[0], controls[1], target)
        variant = "relative_phase_toffoli"
    else:
        circuit = _qiskit_mcx_circuit(n_qubits, controls, target, polarities, cfg.mcx_opt_level)
        variant = "qiskit_hcx_rz"
    return GeneratedSample(
        circuit=circuit,
        n_qubits=n_qubits,
        family=FAMILY_MCX,
        input_format="qasm",
        info={
            "n_controls": n_controls,
            "controls": list(controls),
            "target": target,
            "polarities": [bool(p) for p in polarities],
            "variant": variant,
        },
    )


def _random_gate(
    rng: random.Random,
    n_qubits: int,
    weights: Dict[str, float],
) -> Tuple[str, Tuple[int, ...]]:
    names = list(weights)
    probs = [weights[name] for name in names]
    name = rng.choices(names, weights=probs, k=1)[0]
    if name in ("H", "S", "T", "X", "Z"):
        return name, (rng.randrange(n_qubits),)
    q0 = rng.randrange(n_qubits)
    q1 = rng.randrange(n_qubits - 1)
    if q1 >= q0:
        q1 += 1
    return name, (q0, q1)


CLIFFORD_DRESSING_WEIGHTS = {"H": 1.0, "S": 1.0, "CNOT": 1.0, "X": 0.4, "Z": 0.4}


def _add_random_gate(circuit: zx.Circuit, name: str, qubits: Sequence[int]) -> None:
    if name == "X":
        circuit.add_gate("NOT", *qubits)
    else:
        circuit.add_gate(name, *qubits)


def gen_clifford_t(rng: random.Random, cfg: GenConfig) -> GeneratedSample:
    """Random Clifford+T circuit with ``t_min..t_max`` T gates."""
    n_qubits = rng.randint(cfg.min_qubits, cfg.max_qubits)
    n_t = rng.randint(cfg.t_min, cfg.t_max)
    h_free = rng.random() < cfg.clifford_t_hfree_prob

    weights = {"CNOT": 3.0, "S": 1.0, "X": 0.6, "Z": 0.6, "T": 1.6}
    if h_free:
        weights["CZ"] = 0.8
    else:
        weights["H"] = 2.0

    circuit = zx.Circuit(n_qubits)
    if rng.random() < cfg.clifford_t_dressing_prob:
        for _ in range(rng.randint(1, 3 * n_qubits)):
            name, qubits = _random_gate(rng, n_qubits, CLIFFORD_DRESSING_WEIGHTS)
            _add_random_gate(circuit, name, qubits)

    placed = 0
    while placed < n_t:
        name, qubits = _random_gate(rng, n_qubits, weights)
        if name == "T":
            circuit.add_gate("T", *qubits)
            placed += 1
        else:
            _add_random_gate(circuit, name, qubits)

    if rng.random() < cfg.clifford_t_dressing_prob:
        for _ in range(rng.randint(1, 3 * n_qubits)):
            name, qubits = _random_gate(rng, n_qubits, CLIFFORD_DRESSING_WEIGHTS)
            _add_random_gate(circuit, name, qubits)

    return GeneratedSample(
        circuit=circuit,
        n_qubits=n_qubits,
        family=FAMILY_CLIFFORD_T,
        input_format="qasm",
        info={"n_t_requested": n_t, "h_free": h_free},
    )


def gen_phase_poly(rng: random.Random, cfg: GenConfig) -> GeneratedSample:
    """Random phase polynomial, reverse engineered into a ZX diagram."""
    n_qubits = rng.randint(cfg.min_qubits, cfg.max_qubits)
    n_terms = rng.randint(cfg.pp_min_terms, cfg.pp_max_terms)
    max_parity_size = min(cfg.pp_max_parity_size, n_qubits)
    min_parity_size = min(cfg.pp_min_parity_size, max_parity_size)

    terms: Dict[int, int] = {}
    for _ in range(n_terms):
        size = rng.randint(min_parity_size, max_parity_size)
        bits = rng.sample(range(n_qubits), size)
        parity = 0
        for bit in bits:
            parity |= 1 << bit
        units = rng.choice([1, 2, 3, 4, 5, 6, 7])
        terms[parity] = (terms.get(parity, 0) + units) % 8
    terms = {p: c for p, c in terms.items() if c % 8}
    if not terms:
        size = min(2, n_qubits)
        parity = 0
        for bit in rng.sample(range(n_qubits), size):
            parity |= 1 << bit
        terms[parity] = 1

    circuit = zx.Circuit(n_qubits)
    circuit.gates = terms_to_gate_block(terms, n_qubits)
    input_format = "zx_graph" if rng.random() < cfg.pp_native_zx_prob else "qasm"
    return GeneratedSample(
        circuit=circuit,
        n_qubits=n_qubits,
        family=FAMILY_PHASE_POLY,
        input_format=input_format,
        info={"requested_terms": n_terms, "realised_terms": len(terms)},
    )


GENERATORS: Dict[str, Callable[[random.Random, GenConfig], GeneratedSample]] = {
    FAMILY_MCX: gen_mcx,
    FAMILY_CLIFFORD_T: gen_clifford_t,
    FAMILY_PHASE_POLY: gen_phase_poly,
}


# ---------------------------------------------------------------------------
# simplification and labelling
# ---------------------------------------------------------------------------


def simplify_circuit(
    circuit: zx.Circuit,
    cfg: GenConfig,
) -> Tuple[zx.Graph, zx.Circuit, Dict[str, int]]:
    """``clifford_simp`` + ``full_reduce`` and re-extraction to a circuit."""
    graph = circuit.to_graph()
    stats: Dict[str, int] = {
        "nodes_before": int(graph.num_vertices()),
        "edges_before": int(graph.num_edges()),
    }
    if cfg.run_clifford_simp:
        zx.simplify.clifford_simp(graph)
        stats["nodes_after_clifford_simp"] = int(graph.num_vertices())
    if cfg.run_full_reduce:
        try:
            zx.simplify.full_reduce(graph)
        except Exception:
            pass
    stats["nodes_after"] = int(graph.num_vertices())
    stats["edges_after"] = int(graph.num_edges())

    simplified = circuit
    try:
        extracted = zx.extract_circuit(graph.copy())
        if extracted.qubits >= circuit.qubits:
            simplified = extracted
        stats["extracted_gates"] = int(len(simplified.gates))
    except Exception:
        stats["extracted_gates"] = int(len(circuit.gates))
    return graph, simplified, stats


def label_tcount(
    original: zx.Circuit,
    simplified: zx.Circuit,
    cfg: GenConfig,
) -> Tuple[TCountLabel, PhasePolynomial]:
    """Compute the T-count label plus the phase polynomial target."""
    if cfg.topt_cmd:
        zx.settings.topt_command = cfg.topt_cmd.split()

    polynomial = extract_phase_polynomial(simplified, cfg.phase_mode, source="simplified")
    if polynomial.unsupported or not polynomial.blocks:
        fallback = extract_phase_polynomial(original, cfg.phase_mode, source="original")
        if not fallback.unsupported and fallback.blocks:
            polynomial = fallback

    details: Dict[str, Any] = {
        "n_blocks": int(polynomial.n_blocks),
        "n_terms": int(polynomial.n_terms),
        "rounded": bool(not polynomial.all_exact),
        "tcount_naive": int(tcount_of_gates(simplified.gates, cfg.phase_mode)),
        "tcount_todd_global": None,
        "tcount_todd_sum_of_blocks": None,
        "tcount_exact": None,
        "tcount_external": None,
        "exact_search_proved": False,
    }

    # The phase polynomial now carries 32-class (pi/16) labels; the T-count
    # model itself is defined in units of pi/4, so convert once here.  This is
    # the same arithmetic the generator used before the finer phase grid.
    pi4_blocks = [phase_terms_to_pi4(block, cfg.phase_mode) for block in polynomial.blocks]

    # --- TODD on every phase block separately
    if cfg.use_todd and cfg.use_todd_blockwise:
        total = 0
        ok = True
        for block in pi4_blocks:
            value = todd_tcount_of_terms(block, polynomial.n_qubits)
            if value is None:
                ok = False
                break
            total += int(value)
        details["tcount_todd_sum_of_blocks"] = total if ok else None

    # --- TODD, whole circuit (phase blocks + parity network + Clifford fix-ups)
    if cfg.use_todd:
        details["tcount_todd_global"] = todd_global_tcount(simplified, cfg.todd_max_gates)

    # --- optional exact search (proves optimality for small instances)
    if cfg.exact_search and polynomial.n_qubits <= cfg.exact_max_qubits:
        total = 0
        proved = True
        for block in pi4_blocks:
            value, ok = exact_tcount_search(
                block, polynomial.n_qubits, cfg.exact_max_cost, cfg.exact_max_nodes
            )
            if not ok or value is None:
                proved = False
                break
            total += int(value)
        if proved:
            details["tcount_exact"] = total
            details["exact_search_proved"] = True

    # --- optional external oracle (TOpt / t-par style binary)
    if cfg.external_tcount_cmd:
        details["tcount_external"] = external_tcount(simplified, cfg.external_tcount_cmd)

    candidates: List[Tuple[int, str]] = []
    for key, name in (
        ("tcount_exact", "exact_search"),
        ("tcount_external", "external_tool"),
        ("tcount_todd_sum_of_blocks", "todd_per_block"),
        ("tcount_todd_global", "todd_global"),
        ("tcount_naive", "naive_gate_count"),
    ):
        value = details.get(key)
        if value is not None:
            candidates.append((int(value), name))
    if not candidates:
        candidates.append((int(details["tcount_naive"]), "naive_gate_count"))
    value, method = min(candidates, key=lambda item: item[0])

    if not polynomial.all_exact:
        level = LEVEL_ROUNDED
    elif details["exact_search_proved"] and value == details.get("tcount_exact"):
        level = LEVEL_EXACT
    elif cfg.external_tcount_exact and value == details.get("tcount_external"):
        level = LEVEL_EXACT
    elif details.get("tcount_external") is not None and value == details.get("tcount_external"):
        level = LEVEL_EXTERNAL
    elif details.get("tcount_todd_global") is not None or details.get(
        "tcount_todd_sum_of_blocks"
    ) is not None:
        level = LEVEL_TODD
    else:
        level = LEVEL_UPPER

    details["candidates"] = {name: int(val) for val, name in candidates}
    label = TCountLabel(value=int(value), level=level, method=method, details=details)
    return label, polynomial


# ---------------------------------------------------------------------------
# ZX graph -> PyG Data
# ---------------------------------------------------------------------------


def _vertex_support(graph: zx.Graph, verts: Sequence[int], n_qubits: int) -> Dict[int, int]:
    """Component-wise boundary qubit support of every vertex (bitmask)."""
    neighbours = {v: set(graph.neighbors(v)) for v in verts}
    support: Dict[int, int] = {}
    seen: set = set()
    for start in verts:
        if start in seen:
            continue
        stack = [start]
        component: List[int] = []
        seen.add(start)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for nb in neighbours[vertex]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        mask = 0
        for vertex in component:
            if int(graph.type(vertex)) == int(zx.VertexType.BOUNDARY):
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        if mask == 0:
            for vertex in component:
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        for vertex in component:
            support[vertex] = mask
    return support


def graph_to_data(
    graph: zx.Graph,
    polynomial: PhasePolynomial,
    label: TCountLabel,
    generated: GeneratedSample,
    cfg: GenConfig,
    simplify_stats: Dict[str, int],
    spec: SampleSpec,
    t_depth: float = 0.0,
) -> "Data":
    """Turn a simplified ZX graph (+ labels) into a PyG ``Data`` object."""
    if not _HAS_PYG:
        raise RuntimeError(f"torch_geometric is required for Data objects: {_PYG_ERROR}")

    n_qubits = generated.n_qubits
    verts = sorted(graph.vertices())
    index_of = {vertex: i for i, vertex in enumerate(verts)}
    support = _vertex_support(graph, verts, n_qubits)

    features = np.zeros((len(verts), len(NODE_FEATURES)), dtype=np.float32)
    support_mask = np.zeros((len(verts), cfg.max_qubits), dtype=np.int8)
    qubit_index = np.full((len(verts),), -1, dtype=np.int64)

    for vertex in verts:
        row = index_of[vertex]
        vtype = int(graph.type(vertex))
        phase = graph.phase(vertex)
        phase_value = float(phase) if phase is not None else 0.0
        degree = int(graph.vertex_degree(vertex))
        mask = support.get(vertex, 0)
        size = int(np.bitwise_count(np.int64(mask))) if mask else 0
        qubit = graph.qubit(vertex)
        is_non_clifford = abs((phase_value * 4.0) % 1.0) > 1e-9
        features[row] = (
            1.0 if vtype == int(zx.VertexType.Z) else 0.0,
            1.0 if vtype == int(zx.VertexType.X) else 0.0,
            1.0 if vtype == int(zx.VertexType.BOUNDARY) else 0.0,
            1.0 if graph.is_ground(vertex) else 0.0,
            (phase_value % 2.0) / 2.0,
            math.sin(math.pi * phase_value),
            math.cos(math.pi * phase_value),
            1.0 if is_non_clifford else 0.0,
            degree / max(n_qubits, 1),
            size / max(n_qubits, 1),
            (float(qubit) / max(n_qubits - 1, 1)) if qubit is not None else -1.0,
        )
        for q in range(min(n_qubits, cfg.max_qubits)):
            support_mask[row, q] = 1 if mask >> q & 1 else 0
        if qubit is not None:
            qubit_index[row] = int(qubit)

    edges: List[Tuple[int, int]] = []
    edge_types: List[int] = []
    for edge in graph.edges():
        a, b = graph.edge_st(edge)
        if a not in index_of or b not in index_of:
            continue
        etype = 1 if int(graph.edge_type(edge)) == int(zx.EdgeType.HADAMARD) else 0
        i, j = index_of[a], index_of[b]
        edges.append((i, j))
        edge_types.append(etype)
        edges.append((j, i))
        edge_types.append(etype)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    parities: List[List[int]] = []
    phases: List[int] = []
    blocks: List[int] = []
    for block_id, block in enumerate(polynomial.blocks):
        for parity, units in sorted(block.items()):
            parities.append([(parity >> q) & 1 for q in range(cfg.max_qubits)])
            phases.append(int(units) % PHASE_CLASSES)
            blocks.append(block_id)
    if parities:
        y_parity = torch.tensor(parities, dtype=torch.int8)
        y_phase = torch.tensor(phases, dtype=torch.long)
        y_block = torch.tensor(blocks, dtype=torch.long)
    else:
        y_parity = torch.zeros((0, cfg.max_qubits), dtype=torch.int8)
        y_phase = torch.zeros((0,), dtype=torch.long)
        y_block = torch.zeros((0,), dtype=torch.long)

    qasm = ""
    if cfg.store_qasm:
        try:
            qasm = generated.circuit.to_qasm()
        except Exception:
            qasm = ""

    def _optional_int(value: Any) -> int:
        return -1 if value is None else int(value)

    metadata: Dict[str, Any] = {
        "sample_id": int(spec.index),
        "seed": int(spec.seed),
        "family": generated.family,
        "n_qubits": int(n_qubits),
        "input_format": generated.input_format,
        "split": spec.split,
        "generator_version": VERSION,
        "optimality_level": label.level,
        "label_method": label.method,
        "y_tcount": int(label.value),
        "num_nodes": int(len(verts)),
        "num_edges": int(len(edges) // 2),
        "num_terms": int(len(phases)),
        "n_blocks": int(polynomial.n_blocks),
        "rounded_phases": bool(not polynomial.all_exact),
        "phase_source": polynomial.source,
        "t_depth": float(t_depth),
        "unsupported_gates": int(len(polynomial.unsupported)),
        "tcount_n_blocks": int(label.details.get("n_blocks", 0)),
        "tcount_n_terms": int(label.details.get("n_terms", 0)),
        "tcount_rounded": bool(label.details.get("rounded", False)),
        "tcount_naive": int(label.details.get("tcount_naive", 0)),
        "tcount_todd_global": _optional_int(label.details.get("tcount_todd_global")),
        "tcount_todd_sum_of_blocks": _optional_int(
            label.details.get("tcount_todd_sum_of_blocks")
        ),
        "tcount_exact": _optional_int(label.details.get("tcount_exact")),
        "tcount_external": _optional_int(label.details.get("tcount_external")),
        "exact_search_proved": bool(label.details.get("exact_search_proved", False)),
        "tcount_candidates_json": json.dumps(label.details.get("candidates", {}), sort_keys=True),
        "gen_info_json": json.dumps(generated.info, sort_keys=True, default=str),
        "graph_stats_json": json.dumps(simplify_stats, sort_keys=True),
        "gen_n_controls": int(generated.info.get("n_controls", -1)),
        "gen_target": int(generated.info.get("target", -1)),
        "gen_variant": str(generated.info.get("variant", "")),
        "gen_n_t_requested": int(generated.info.get("n_t_requested", -1)),
        "gen_h_free": bool(generated.info.get("h_free", False)),
        "gen_requested_terms": int(generated.info.get("requested_terms", -1)),
        "gen_realised_terms": int(generated.info.get("realised_terms", -1)),
        "qasm": qasm,
    }
    # uniform key set across samples: required for PyG batching of dict attributes
    metadata = {key: metadata.get(key) for key in METADATA_KEYS}

    data = Data(x=torch.from_numpy(features), edge_index=edge_index)
    data.edge_type = torch.tensor(edge_types, dtype=torch.long)
    data.x_support = torch.from_numpy(support_mask)
    data.x_qubit = torch.from_numpy(qubit_index)
    data.y_parity = y_parity
    data.y_phase = y_phase
    data.y_block = y_block
    data.y_tcount = torch.tensor(int(label.value), dtype=torch.long)
    data.num_terms = int(len(phases))
    data.n_qubits = int(n_qubits)
    data.family = generated.family
    data.optimality_level = label.level
    data.metadata = metadata
    data.y_depth = torch.tensor([float(t_depth)], dtype=torch.float)
    data.y_count = torch.tensor([data.num_terms], dtype=torch.float)
    return data


# ---------------------------------------------------------------------------
# one sample + worker plumbing
# ---------------------------------------------------------------------------

_WORKER_CFG: Optional[GenConfig] = None


def build_sample(spec: SampleSpec, cfg: GenConfig) -> "Data":
    """Generate, simplify, label and featurise a single sample."""
    last_error: Optional[Exception] = None
    for attempt in range(max(1, cfg.retries)):
        rng = random.Random(spec.seed + 7919 * attempt)
        try:
            generated = GENERATORS[spec.family](rng, cfg)
            if generated.circuit.qubits > cfg.max_qubits:
                raise ValueError("generated circuit exceeds max_qubits")
            data = _finish_sample(generated, cfg, spec)
            if data.x.shape[0] > cfg.max_nodes:
                raise ValueError(f"graph with {data.x.shape[0]} nodes exceeds max_nodes")
            return data
        except Exception as exc:  # retry with a different seed offset
            last_error = exc
    raise RuntimeError(f"sample {spec.index} ({spec.family}) failed: {last_error!r}")


def _init_worker(cfg: GenConfig) -> None:  # pragma: no cover - runs in children
    global _WORKER_CFG
    _WORKER_CFG = cfg
    salt = (cfg.seed + os.getpid()) % (2**32)
    np.random.seed(salt)
    random.seed(salt)
    if cfg.topt_cmd:
        zx.settings.topt_command = cfg.topt_cmd.split()


def serialize_data(data: "Data") -> bytes:
    """Serialise a ``Data`` object to bytes.

    Sending tensors through a ``multiprocessing`` queue makes torch share their
    storage over a file-descriptor socket, which is unavailable in locked-down
    sandboxes.  Serialising to bytes inside the worker keeps the IPC payload
    plain and portable.
    """
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def deserialize_data(payload: bytes) -> "Data":
    return torch.load(io.BytesIO(payload), weights_only=False)


def _finish_sample(
    generated: GeneratedSample,
    cfg: GenConfig,
    spec: SampleSpec,
) -> "Data":
    """Simplify, label (T-count + routed T-depth) and featurise a circuit."""
    graph, simplified, stats = simplify_circuit(generated.circuit, cfg)
    if graph.num_vertices() == 0:
        raise ValueError("degenerate ZX diagram (no vertices)")
    label, polynomial = label_tcount(generated.circuit, simplified, cfg)
    # one Steiner routing pass over the extracted phase polynomial -> T-depth
    t_depth = routed_t_depth(polynomial, cfg)
    return graph_to_data(
        graph, polynomial, label, generated, cfg, stats, spec, t_depth=t_depth
    )


def data_from_qasm(
    qasm: str,
    cfg: Optional[GenConfig] = None,
    spec: Optional[SampleSpec] = None,
    family: str = "input_qasm",
) -> "Data":
    """Build one sample from an existing QASM string.

    This is the "QASM input" entry point: the circuit is simplified with
    ``clifford_simp``/``full_reduce``, its phase polynomial and T-count label
    are computed with the configured backends, and the simplified ZX diagram is
    featurised into a ``Data`` object.
    """
    cfg = cfg or GenConfig()
    spec = spec or SampleSpec(index=-1, seed=0, family=family, split="input")
    circuit = zx.Circuit.from_qasm(qasm)
    if circuit.qubits > cfg.max_qubits:
        raise ValueError(
            f"circuit uses {circuit.qubits} qubits but max_qubits={cfg.max_qubits}; "
            "increase GenConfig.max_qubits so that parity columns are not truncated"
        )
    generated = GeneratedSample(
        circuit=circuit,
        n_qubits=circuit.qubits,
        family=family,
        input_format="qasm",
        info={},
    )
    return _finish_sample(generated, cfg, spec)


def data_from_graph(
    graph: zx.Graph,
    n_qubits: Optional[int] = None,
    cfg: Optional[GenConfig] = None,
    spec: Optional[SampleSpec] = None,
    family: str = "input_zx_graph",
) -> "Data":
    """Build one sample from an existing ZX diagram (the "ZX graph input" path).

    The diagram is copied before simplification, and a circuit is extracted from
    it so that the same phase-polynomial / T-count machinery applies.
    """
    cfg = cfg or GenConfig()
    spec = spec or SampleSpec(index=-1, seed=0, family=family, split="input")
    working = graph.copy()
    if n_qubits is None:
        qubits = [int(q) for q in (working.qubit(v) for v in working.vertices()) if q is not None]
        n_qubits = (max(qubits) + 1) if qubits else working.qubit_count()
    if int(n_qubits) > cfg.max_qubits:
        raise ValueError(
            f"diagram uses {n_qubits} qubits but max_qubits={cfg.max_qubits}; "
            "increase GenConfig.max_qubits so that parity columns are not truncated"
        )
    helper = working.copy()
    for step in (zx.simplify.full_reduce, zx.simplify.to_graph_like):
        try:
            step(helper)
        except Exception:
            pass
    try:
        circuit = zx.extract_circuit(helper)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "could not extract a circuit from the supplied ZX diagram; run "
            "pyzx.simplify.full_reduce on it first"
        ) from exc
    generated = GeneratedSample(
        circuit=circuit,
        n_qubits=int(max(n_qubits, circuit.qubits)),
        family=family,
        input_format="zx_graph",
        info={},
    )
    return _finish_sample(generated, cfg, spec)


def load_split(path: str) -> List["Data"]:
    """Load a generated split file (``weights_only=False`` is required for PyG)."""
    return torch.load(path, weights_only=False)


def _worker_build(spec: SampleSpec) -> Tuple[SampleSpec, Optional[bytes], Optional[str]]:
    assert _WORKER_CFG is not None, "worker not initialised"
    try:
        return spec, serialize_data(build_sample(spec, _WORKER_CFG)), None
    except Exception as exc:  # pragma: no cover - surfaced in the report
        return spec, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# task planning / splits
# ---------------------------------------------------------------------------


def effective_families(cfg: GenConfig) -> Tuple[str, ...]:
    """Selected families, minus those whose dependencies are missing."""
    families = tuple(f for f in cfg.families if f in GENERATORS)
    if not _HAS_QISKIT and FAMILY_MCX in families:
        families = tuple(f for f in families if f != FAMILY_MCX)
    return families


def plan_specs(cfg: GenConfig) -> List[SampleSpec]:
    """Deterministic (seed, family, split) assignment, stratified by family."""
    families = list(effective_families(cfg))
    if not families:
        raise ValueError("no valid families selected")
    weights = {f: float(cfg.family_weights.get(f, 1.0)) for f in families}
    counts = largest_remainder(weights, cfg.samples)

    rng = random.Random(cfg.seed)
    specs: List[SampleSpec] = []
    next_index = 0
    ratios = cfg.split_ratios
    for family in families:
        total = counts[family]
        n_train = int(round(total * ratios[0]))
        n_val = int(round(total * ratios[1]))
        n_test = max(0, total - n_train - n_val)
        split_list = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        if len(split_list) < total:
            split_list += ["train"] * (total - len(split_list))
        split_list = split_list[:total]
        rng.shuffle(split_list)
        for split in split_list:
            specs.append(
                SampleSpec(
                    index=next_index,
                    seed=derive_seed(cfg.seed, next_index),
                    family=family,
                    split=split,
                )
            )
            next_index += 1
    rng.shuffle(specs)
    return [
        SampleSpec(index=i, seed=s.seed, family=s.family, split=s.split)
        for i, s in enumerate(specs)
    ]


def _shard_path(shard_dir: Path, split: str, shard: int) -> Path:
    return shard_dir / f"{split}_{shard:03d}.pt"


def run_generation(cfg: GenConfig, verbose: bool = True) -> Dict[str, Any]:
    """Generate the dataset with a process pool and write train/val/test files."""
    if not _HAS_PYG:
        raise RuntimeError(f"torch / torch_geometric are required: {_PYG_ERROR}")

    if not _HAS_QISKIT and FAMILY_MCX in cfg.families:
        print("warning: qiskit is not installed -> dropping the 'mcx' family", file=sys.stderr)
        cfg = replace(cfg, families=effective_families(cfg))
        cfg.family_weights = {f: cfg.family_weights.get(f, 1.0) for f in cfg.families}

    specs = plan_specs(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    workers = cfg.normalized_workers()
    buffers: Dict[str, List["Data"]] = {split: [] for split in ("train", "val", "test")}
    shard_counts: Dict[str, int] = defaultdict(int)
    failures: List[str] = []
    level_hist: Counter = Counter()
    family_hist: Counter = Counter()
    started = time.time()
    done = 0

    def flush(split: str, force: bool = False) -> None:
        while len(buffers[split]) >= cfg.shard_size or (force and buffers[split]):
            chunk = buffers[split][: cfg.shard_size]
            del buffers[split][: cfg.shard_size]
            torch.save(chunk, _shard_path(shard_dir, split, shard_counts[split]))
            shard_counts[split] += 1

    def handle(spec: SampleSpec, payload: Optional[bytes], error: Optional[str]) -> None:
        nonlocal done
        done += 1
        if payload is None:
            failures.append(f"[{spec.family}] {error}")
        else:
            data = deserialize_data(payload)
            buffers[spec.split].append(data)
            flush(spec.split)
            level_hist[data.optimality_level] += 1
            family_hist[data.family] += 1
        if verbose and (done % cfg.progress_every == 0 or done == len(specs)):
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"  [{done}/{len(specs)}] {done / elapsed:6.1f} samples/s  "
                f"failures={len(failures)}  elapsed={elapsed:6.1f}s",
                flush=True,
            )

    if workers == 1:
        # single process: skip the pool entirely (faster and easier to debug)
        _init_worker(cfg)
        results: Iterable[Tuple[SampleSpec, Optional[bytes], Optional[str]]] = (
            _worker_build(spec) for spec in specs
        )
        for spec, payload, error in results:
            handle(spec, payload, error)
    else:
        pool = mp.Pool(processes=workers, initializer=_init_worker, initargs=(cfg,))
        try:
            for spec, payload, error in pool.imap_unordered(
                _worker_build, specs, chunksize=cfg.chunksize
            ):
                handle(spec, payload, error)
        finally:
            pool.close()
            pool.join()

    for split in ("train", "val", "test"):
        flush(split, force=True)

    written: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        n_shards = shard_counts[split]
        written[split] = []
        if n_shards == 1:
            target = out_dir / f"{split}.pt"
            shutil.move(str(_shard_path(shard_dir, split, 0)), str(target))
            written[split].append(str(target))
        elif n_shards > 1:
            written[split] = [str(_shard_path(shard_dir, split, i)) for i in range(n_shards)]
    if shard_dir.exists() and not any(shard_dir.iterdir()):
        shard_dir.rmdir()

    report = {
        "samples_requested": cfg.samples,
        "samples_written": int(sum(family_hist.values())),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "families": dict(family_hist),
        "optimality_levels": dict(level_hist),
        "files": written,
        "seconds": round(time.time() - started, 2),
        "workers": workers,
        "config": asdict(cfg),
        "versions": {
            "generator": VERSION,
            "pyzx": getattr(zx, "__version__", "?"),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", "?"),
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if verbose:
        preview = {k: report[k] for k in ("samples_written", "failures", "families", "optimality_levels", "seconds")}
        print(json.dumps(preview, indent=2))
    return report


def merge_shards(out_dir: str, splits: Sequence[str] = ("train", "val", "test")) -> Dict[str, str]:
    """Concatenate ``<split>_NNN.pt`` shards into single ``<split>.pt`` files."""
    base = Path(out_dir)
    merged: Dict[str, str] = {}
    for split in splits:
        shards = sorted((base / "_shards").glob(f"{split}_[0-9][0-9][0-9].pt"))
        shards += sorted(base.glob(f"{split}_[0-9][0-9][0-9].pt"))
        if not shards:
            continue
        bucket: List[Any] = []
        for shard in shards:
            bucket.extend(torch.load(shard, weights_only=False))
        target = base / f"{split}.pt"
        torch.save(bucket, target)
        merged[split] = str(target)
        for shard in shards:
            shard.unlink()
    return merged


# ---------------------------------------------------------------------------
# self tests
# ---------------------------------------------------------------------------


def _self_test(seed: int = 0) -> int:
    """Verify generators, phase extraction and T-count labels."""
    failures: List[str] = []

    def check(name: str, condition: bool, extra: str = "") -> None:
        print(f"  [{'ok  ' if condition else 'FAIL'}] {name} {extra}")
        if not condition:
            failures.append(name)

    cfg = GenConfig(
        samples=8,
        seed=seed,
        workers=1,
        out_dir="/tmp/_zx_gen_selftest",
        exact_search=True,
    )
    rng = random.Random(seed)

    # -- 1. multi-controlled X circuits match the Qiskit reference -----------
    if _HAS_QISKIT and _HAS_PYG:
        from qiskit import qasm2 as qasm2_mod
        from qiskit.quantum_info import Operator

        for n_controls in (1, 2, 3):
            reference_qc = QuantumCircuit(n_controls + 1)
            reference_qc.append(MCXGate(n_controls), list(range(n_controls + 1)))
            reference = Operator(reference_qc).data
            sample = _qiskit_mcx_circuit(
                n_controls + 1, list(range(n_controls)), n_controls, [True] * n_controls, 1
            )
            # re-parse with Qiskit so both matrices use the same qubit convention
            matrix = Operator(qasm2_mod.loads(sample.to_qasm())).data
            scale = complex(np.vdot(reference, matrix) / np.vdot(reference, reference))
            ok = np.allclose(matrix, reference * scale, atol=1e-8) and abs(abs(scale) - 1.0) < 1e-6
            check(f"C^{n_controls}X matches the Qiskit reference (up to global phase)", ok)
    else:
        print("  [skip] qiskit/torch_geometric missing: mcx checks skipped")

    # -- 2. parity extraction agrees with pyzx' PhasePoly --------------------
    from pyzx.routing.phase_poly import PhasePoly

    agreement = True
    compared = 0
    for _ in range(15):
        n = rng.randint(2, 5)
        circuit = zx.Circuit(n)
        for _ in range(rng.randint(1, 4)):
            bits = rng.sample(range(n), rng.randint(1, n))
            target = bits[-1]
            for q in bits[:-1]:
                circuit.add_gate("CNOT", q, target)
            circuit.add_gate("ZPhase", target, Fraction(rng.choice([1, 2, 3, 5, 7]), 4))
        mine = extract_phase_polynomial(circuit, "exact").flat_terms()
        try:
            reference = PhasePoly.fromCircuit(circuit)
        except Exception:
            continue
        ref_terms: Dict[int, int] = {}
        for parity, phase in reference.zphases.items():
            bits = [int(b) for b in parity.parity]
            mask = 0
            for q, bit in enumerate(bits):
                if bit:
                    mask |= 1 << q
            units, _ = phase_to_units(phase, "exact")  # now a 32-class index
            ref_terms[mask] = (ref_terms.get(mask, 0) + units) % PHASE_CLASSES
        ref_terms = {p: c for p, c in ref_terms.items() if c % PHASE_CLASSES}
        compared += 1
        if mine != ref_terms:
            agreement = False
            print(f"      mismatch: {mine} != {ref_terms}")
    check("parity extraction matches pyzx PhasePoly", agreement, f"(compared={compared})")

    # -- 2b. SWAP tracking and X-sign tracking -------------------------------
    # SWAP is expanded into three CNOTs and handed to pyzx' own extractor, so
    # this is checked against independent machinery.
    swap_ok = True
    compared = 0
    for _ in range(10):
        n = rng.randint(2, 4)
        with_swap = zx.Circuit(n)
        expanded = zx.Circuit(n)
        for _ in range(rng.randint(1, 3)):
            bits = rng.sample(range(n), rng.randint(1, n))
            target = bits[-1]
            for q in bits[:-1]:
                with_swap.add_gate("CNOT", q, target)
                expanded.add_gate("CNOT", q, target)
            phase = Fraction(rng.choice([1, 3, 5, 7]), 4)
            with_swap.add_gate("ZPhase", target, phase)
            expanded.add_gate("ZPhase", target, phase)
        a, b = rng.sample(range(n), 2)
        with_swap.add_gate("SWAP", a, b)
        for control, target in ((a, b), (b, a), (a, b)):
            expanded.add_gate("CNOT", control, target)
        mine = extract_phase_polynomial(with_swap, "exact").flat_terms()
        try:
            reference = PhasePoly.fromCircuit(expanded)
        except Exception:
            continue
        expected: Dict[int, int] = {}
        for parity, phase in reference.zphases.items():
            mask = 0
            for q, bit in enumerate(int(x) for x in parity.parity):
                if bit:
                    mask |= 1 << q
            units, _ = phase_to_units(phase, "exact")
            expected[mask] = (expected.get(mask, 0) + units) % PHASE_CLASSES
        expected = {p: c for p, c in expected.items() if c % PHASE_CLASSES}
        compared += 1
        if mine != expected:
            swap_ok = False
            print(f"      swap mismatch: {mine} != {expected}")
    check("SWAP tracking matches SWAP-as-3-CNOTs", swap_ok, f"(compared={compared})")

    sign_ok = True
    try:
        import numpy as _np

        lhs = zx.Circuit(1)
        lhs.add_gate("NOT", 0)
        lhs.add_gate("T", 0)
        rhs = zx.Circuit(1)
        rhs.add_gate("ZPhase", 0, Fraction(-1, 4))
        rhs.add_gate("NOT", 0)
        reference = lhs.to_matrix()
        candidate = rhs.to_matrix()
        scale = complex(_np.vdot(reference, candidate) / _np.vdot(reference, reference))
        sign_ok = _np.allclose(candidate, reference * scale, atol=1e-9)
    except Exception as exc:  # pragma: no cover
        sign_ok = False
        print("      sign check error:", exc)
    check("X-gate sign tracking matches the T/T-dagger identity", sign_ok)

    # -- 3. exact search equals TODD on single blocks ------------------------
    agreement = True
    compared = 0
    for _ in range(15):
        n = rng.randint(2, 4)
        terms: Dict[int, int] = {}
        for _ in range(rng.randint(1, 5)):
            parity = rng.randrange(1, 1 << n)
            terms[parity] = (terms.get(parity, 0) + rng.choice([1, 2, 3, 4, 5, 6, 7])) % 8
        terms = {p: c for p, c in terms.items() if c % 8}
        if not terms:
            continue
        expected, proved = exact_tcount_search(terms, n, max_cost=5, max_nodes=40_000)
        todd = todd_tcount_of_terms(terms, n)
        if not proved or expected is None:
            continue  # budget exhausted: not a counterexample
        compared += 1
        if todd is not None and todd != expected:
            agreement = False
            print(f"      mismatch: exact={expected} todd={todd} terms={terms}")
    check("exact search equals TODD", agreement, f"(compared={compared})")

    # -- 4. end to end sample construction ----------------------------------
    if _HAS_PYG:
        specs = plan_specs(cfg)
        built: Counter = Counter()
        levels: Counter = Counter()
        for spec in specs:
            data = build_sample(spec, cfg)
            built[data.family] += 1
            levels[data.optimality_level] += 1
            assert data.x.shape[1] == len(NODE_FEATURES)
            assert data.edge_index.shape[0] == 2
            assert data.y_parity.shape[1] == cfg.max_qubits
            assert data.y_parity.shape[0] == data.y_phase.shape[0] == data.y_block.shape[0]
            assert data.optimality_level in LEVELS
            assert cfg.min_qubits <= int(data.n_qubits) <= cfg.max_qubits
            assert data.edge_type.shape[0] == data.edge_index.shape[1]
            assert data.x.shape[0] == data.x_support.shape[0] == data.x_qubit.shape[0]
            assert tuple(data.metadata) == METADATA_KEYS, "metadata key set is not uniform"
            assert data.y_depth.shape == (1,) and data.y_depth.dtype == torch.float
            assert data.y_count.shape == (1,) and data.y_count.dtype == torch.float
            assert float(data.y_depth) >= 0.0
            assert float(data.y_count) == float(data.num_terms)
            assert 0 <= int(data.y_phase.max() if data.y_phase.numel() else 0) < PHASE_CLASSES
        check("end-to-end Data construction", sum(built.values()) == len(specs), str(dict(built)))
        check("optimality levels are populated", len(levels) > 0, str(dict(levels)))
        check("metadata key set is batching-safe", all(tuple(d.metadata) == METADATA_KEYS for d in map(lambda s: build_sample(s, cfg), specs[:3])))
    else:
        print("  [skip] torch_geometric missing: end-to-end check skipped")

    # -- 5. splits ----------------------------------------------------------
    counts = Counter(spec.split for spec in plan_specs(GenConfig(samples=1000, seed=seed)))
    fraction = counts["train"] / max(sum(counts.values()), 1)
    check("split ratios ~80/10/10", 0.78 <= fraction <= 0.82, str(dict(counts)))

    # -- 6. QASM / ZX graph round trips reproduce the same label -------------
    if _HAS_PYG:
        sample = build_sample(plan_specs(cfg)[0], cfg)
        qasm = sample.metadata.get("qasm")
        if qasm:
            from_qasm = data_from_qasm(qasm, cfg)
            check(
                "QASM round trip reproduces the T-count label",
                int(from_qasm.y_tcount) == int(sample.y_tcount),
                f"({int(from_qasm.y_tcount)} vs {int(sample.y_tcount)})",
            )
            graph = zx.Circuit.from_qasm(qasm).to_graph()
            from_graph = data_from_graph(graph, n_qubits=sample.n_qubits, cfg=cfg)
            check(
                "ZX graph round trip reproduces the T-count label",
                int(from_graph.y_tcount) == int(sample.y_tcount),
                f"({int(from_graph.y_tcount)} vs {int(sample.y_tcount)})",
            )
        else:
            print("  [skip] qasm storage disabled")

    print()
    if failures:
        print(f"self-test FAILED ({len(failures)}): {failures}")
        return 1
    print("self-test passed")
    return 0


def _smoke(seed: int = 0, out_dir: str = "dataset_smoke") -> int:
    cfg = GenConfig(samples=24, seed=seed, workers=1, out_dir=out_dir, shard_size=100)
    report = run_generation(cfg, verbose=True)
    return 0 if report["failures"] == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--seed", type=int, default=20240914)
    parser.add_argument("--workers", type=int, default=0, help="0 = min(cpu_count, 8)")
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--families", default="mcx,clifford_t,phase_poly")
    parser.add_argument("--min-qubits", type=int, default=3)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--phase-mode", choices=PHASE_MODES, default="ceil")
    parser.add_argument("--todd-max-gates", type=int, default=400)
    parser.add_argument("--no-todd", action="store_true")
    parser.add_argument("--topt-cmd", default=None, help="external TOpt binary used inside pyzx' TODD step")
    parser.add_argument(
        "--exact-search",
        action="store_true",
        help="run the exhaustive (Dijkstra) T-count solver for small instances to prove optimality",
    )
    parser.add_argument("--exact-max-qubits", type=int, default=3)
    parser.add_argument("--exact-max-cost", type=int, default=6)
    parser.add_argument("--exact-max-nodes", type=int, default=20_000)
    parser.add_argument(
        "--external-tcount-cmd",
        default=None,
        help="external T-count oracle (TOpt, t-par, ...): QASM on stdin, integer T-count on stdout",
    )
    parser.add_argument(
        "--external-tcount-exact",
        action="store_true",
        help="treat the external oracle as returning proven optima (level exact_optimal)",
    )
    parser.add_argument("--no-qasm", action="store_true", help="do not store the QASM source in metadata")
    parser.add_argument("--progress-every", type=int, default=500)


def _config_from_args(args: argparse.Namespace) -> GenConfig:
    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    weights = {f: DEFAULT_FAMILY_WEIGHTS.get(f, 1.0) for f in families}
    return GenConfig(
        samples=args.samples,
        families=families,
        family_weights=weights,
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        seed=args.seed,
        out_dir=args.out_dir,
        workers=args.workers,
        chunksize=args.chunksize,
        shard_size=args.shard_size,
        phase_mode=args.phase_mode,
        use_todd=not args.no_todd,
        todd_max_gates=args.todd_max_gates,
        topt_cmd=args.topt_cmd,
        exact_search=args.exact_search,
        exact_max_qubits=args.exact_max_qubits,
        exact_max_cost=args.exact_max_cost,
        exact_max_nodes=args.exact_max_nodes,
        external_tcount_cmd=args.external_tcount_cmd,
        external_tcount_exact=args.external_tcount_exact,
        store_qasm=not args.no_qasm,
        progress_every=args.progress_every,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local ZX-graph data generator for GNN training on quantum circuits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="generate the dataset (default)")
    _add_config_args(gen)

    smoke = sub.add_parser("smoke", help="tiny end-to-end run (24 samples)")
    smoke.add_argument("--out-dir", default="dataset_smoke")
    smoke.add_argument("--seed", type=int, default=0)

    merge = sub.add_parser("merge", help="merge <split>_NNN.pt shards back into <split>.pt")
    merge.add_argument("--out-dir", default="dataset")

    test = sub.add_parser("self-test", help="run correctness checks")
    test.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command in (None, "generate"):
        cfg = _config_from_args(args if args.command else gen.parse_args([]))
        run_generation(cfg)
        return 0
    if args.command == "smoke":
        return _smoke(args.seed, args.out_dir)
    if args.command == "merge":
        print(json.dumps(merge_shards(args.out_dir), indent=2))
        return 0
    if args.command == "self-test":
        return _self_test(args.seed)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
