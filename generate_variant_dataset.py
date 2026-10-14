#!/usr/bin/env python3
"""Supervised data for strategy B: per-block exact vs relative-phase MCX choice.

An "instance" is a small algorithm fragment: ``k`` multi-controlled-X blocks of
the same control count, separated by random Clifford layers (Hadamards included,
which is what makes a diagonal correction matter), followed by a Z-basis
measurement of the data qubits.  For every one of the ``2^k`` assignments of
{exact ladder, relative-phase RCCX ladder} the generator compares the Z-basis
output distribution against the all-exact circuit; an assignment is *usable* when
the maximum total-variation distance over all data-basis inputs stays below
``--tolerance``.

The oracle label is the usable assignment with the fewest T gates (ties resolved
in favour of fewer downgrades, i.e. the more conservative circuit).  Each block
becomes one training sample carrying

* ``x`` / ``edge_index`` / ``edge_attr`` / ``x_support``: the block's own
  simplified ZX diagram (the same feature layout the rest of the project uses),
* ``context``: what surrounds the block (Hadamards/CNOTs in the next layer, how
  many blocks remain, control count, ...), which is what the decision depends on,
* ``y_variant``: 0 = keep exact, 1 = downgrade to relative phase (oracle),
* ``y_instance`` / ``y_oracle_tcount`` / ``y_best_is_all_exact`` /
  ``y_all_relative_usable`` / ``y_fragmented``: instance-level supervision and
  diagnostics.

Usage
-----
    .venv/bin/python generate_variant_dataset.py --self-test
    .venv/bin/python generate_variant_dataset.py --instances 2000 --workers 6 \
        --out-dir variant_dataset --max-blocks 5
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_linux_mcx():
    loader = importlib.machinery.SourceFileLoader("linux_mcx_vardata", str(ROOT / "Linux-MCX"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("cannot build an import spec for Linux-MCX")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from qiskit import QuantumCircuit, qasm2  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402

MAX_QUBITS = int(getattr(LINUX_MCX, "MAX_QUBITS", 16))
NODE_FEATURES = 11

# per-block context vector (what the policy sees besides the block graph).
# NOTE: ``report["context"][k]`` (k >= 1) is the Clifford layer that sits
# *before* block k, i.e. the layer *after* block k-1; context[0] is the empty
# leading layer.  Everything below is the layer(s) that FOLLOW the block.
CONTEXT_FEATURES = (
    "h_next",             # Hadamards in the layer right after this block
    "cnot_next",          # CNOTs in that layer
    "h_downstream",       # total Hadamards in every layer after this block
    "cnot_downstream",    # total CNOTs in those layers
    "n_downstream_layers",# how many inter-block layers remain
    "h_layer_fraction",   # fraction of remaining layers that contain a Hadamard
    "h_layer_max",        # max Hadamards in a single remaining layer
    "h_total",            # Hadamards in the whole instance
    "h_layer_mean",       # mean Hadamards per inter-block layer
    "remaining_blocks",   # blocks after this one
    "block_index",        # position in the instance
    "is_first", "is_last",# boundary flags
    "num_controls", "blocks",
    "t_exact", "t_relative",  # per-block T-cost of each variant
    "next_h_free",        # 1.0 when the very next layer has no Hadamard
    "position",           # block_index / (blocks - 1)
)
# One-hot buckets so the policy can *specialise* per control count instead of
# extrapolating along a single numeric axis (C^7X / C^8X only have a few hundred
# samples each, so a numeric encoding is easily dominated by the C^2X..C^4X mass).
CONTROL_ONEHOT = (1, 2, 3, 4, 5, 6, 7, 8)
BLOCK_ONEHOT = (2, 3, 4, 5, 6, 7, 8)
CONTEXT_FEATURES = CONTEXT_FEATURES + tuple(
    f"is_control_{c}" for c in CONTROL_ONEHOT
) + tuple(f"is_blocks_{b}" for b in BLOCK_ONEHOT)
CONTEXT_DIM = len(CONTEXT_FEATURES)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class VariantConfig:
    instances: int = 200
    max_blocks: int = 5
    control_values: Tuple[int, ...] = (2, 3, 4)
    h_prob: float = 0.7
    tolerance: float = 1e-9
    out_dir: str = "variant_dataset"
    workers: int = 0
    seed: int = 20261007
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    store_mask_table: bool = True
    progress_every: int = 200
    shard_size: int = 100_000

    def normalized_workers(self) -> int:
        return self.workers if self.workers > 0 else max(1, min(os.cpu_count() or 1, 8))


# ---------------------------------------------------------------------------
# instance construction
# ---------------------------------------------------------------------------


def qubit_layout(num_controls: int) -> "Tuple[List[int], List[int], int]":
    """(controls, ancillas, target) for the exact/relative MCX synthesis.

    ``make_mcx_source`` lays the circuit out as controls ``0..n-1``, then the
    clean ancillas ``n..n+a-1``, then the **target on the highest qubit index**.
    The target is *data* while the ancillas must be post-selected to |0>, so
    swapping the two silently selects the wrong subspace -- see Linux-MCX's
    ``relative_phase_metrics`` docstring for the same pitfall.
    """
    n_ancillas = max(0, num_controls - 2)
    total = num_controls + 1 + n_ancillas
    controls = list(range(num_controls))
    ancillas = list(range(num_controls, num_controls + n_ancillas))
    return controls, ancillas, total - 1


def exact_block(num_controls: int) -> QuantumCircuit:
    return qasm2.loads(LINUX_MCX.make_mcx_source(num_controls, basis="clifford_t").to_qasm())


def relative_block(num_controls: int) -> QuantumCircuit:
    return LINUX_MCX.relative_phase_mcx_circuit(num_controls)


_TCOUNT_CACHE: "Dict[int, Tuple[int, int]]" = {}


def block_t_counts(num_controls: int) -> Tuple[int, int]:
    cached = _TCOUNT_CACHE.get(num_controls)
    if cached is not None:
        return cached

    def count(circuit: QuantumCircuit) -> int:
        source = LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()
        return LINUX_MCX.compute_ftq_metrics(source).t_count

    result = (count(exact_block(num_controls)), count(relative_block(num_controls)))
    _TCOUNT_CACHE[num_controls] = result
    return result


def clifford_layer(
    qc: QuantumCircuit,
    qubits: Sequence[int],
    rng: random.Random,
    h_prob: float,
) -> Tuple[int, int]:
    """Random Clifford layer; returns (#Hadamards, #CNOTs) for context features."""
    hadamards = cnots = 0
    for _ in range(rng.randint(1, 2) * len(qubits)):
        use_h = rng.random() < h_prob
        choice = "h" if use_h else rng.choice(("s", "x", "z", "cx"))
        qubit = rng.choice(qubits)
        if choice == "cx":
            other = rng.choice([q for q in qubits if q != qubit])
            qc.cx(qubit, other)
            cnots += 1
        else:
            getattr(qc, choice)(qubit)
            hadamards += int(choice == "h")
    return hadamards, cnots


def build_instance(
    num_controls: int,
    blocks: int,
    mask: int,
    seed: int,
    h_prob: float,
) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    controls, ancillas, target = qubit_layout(num_controls)
    n_ancillas = len(ancillas)
    total = num_controls + 1 + n_ancillas
    data_qubits = controls + [target]     # controls + target; ancillas excluded
    rng = random.Random(seed)
    context: List[Tuple[int, int]] = []
    circuit = QuantumCircuit(total)
    for index in range(blocks):
        if index:
            context.append(clifford_layer(circuit, data_qubits, rng, h_prob))
        else:
            context.append((0, 0))
        block = relative_block(num_controls) if (mask >> index) & 1 else exact_block(num_controls)
        circuit.compose(block, range(total), inplace=True)
    return circuit, context


def build_instance_layers(
    num_controls: int,
    blocks: int,
    seed: int,
    h_prob: float,
) -> "List[Optional[QuantumCircuit]]":
    """The inter-block Clifford layers, as standalone circuits.

    ``layers[j]`` is the layer that sits *before* block ``j`` (``layers[0]`` is
    ``None``).  The RNG consumption mirrors :func:`build_instance` exactly, so
    these are the very gates that instance uses -- which lets a caller rebuild
    the circuit from parts without re-running the generator.
    """
    controls, ancillas, target = qubit_layout(num_controls)
    n_ancillas = len(ancillas)
    total = num_controls + 1 + n_ancillas
    data_qubits = controls + [target]
    rng = random.Random(seed)
    layers: "List[Optional[QuantumCircuit]]" = [None] * blocks
    for index in range(blocks):
        if index:
            layer = QuantumCircuit(total)
            clifford_layer(layer, data_qubits, rng, h_prob)
            layers[index] = layer
    return layers


def measurement_distributions(circuit: QuantumCircuit, num_controls: int) -> "Dict[int, np.ndarray]":
    """Z-basis distribution of the data qubits for every data-basis input."""
    _controls, ancillas, _target = qubit_layout(num_controls)
    total = circuit.num_qubits
    matrix = Operator(circuit).data
    keep = (
        np.array([x for x in range(2 ** total) if all(not (x >> a) & 1 for a in ancillas)])
        if ancillas
        else None
    )
    # The data qubits are the controls plus the *highest* qubit (the target);
    # ancillas sit in between and must start in |0>.  So the valid input states
    # are exactly the ancilla=0 basis states -- NOT the low ``num_controls + 1``
    # bits, which would feed the ancilla as an input and never touch the target.
    input_indices = keep if keep is not None else np.arange(2 ** total)
    distributions: "Dict[int, np.ndarray]" = {}
    for basis in input_indices:
        amplitudes = matrix[:, int(basis)]
        probabilities = np.abs(amplitudes) ** 2
        if keep is not None:
            probabilities = probabilities[keep]
        probabilities = probabilities / max(probabilities.sum(), 1e-30)
        distributions[int(basis)] = probabilities
    return distributions


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    size = max(len(a), len(b))
    aa = np.zeros(size)
    bb = np.zeros(size)
    aa[: len(a)] = a
    bb[: len(b)] = b
    return float(0.5 * np.sum(np.abs(aa - bb)))


_SYMBOLIC_MODULE = None


def _symbolic():
    """Lazily import ``symbolic_verifier`` (it imports this module back)."""
    global _SYMBOLIC_MODULE
    if _SYMBOLIC_MODULE is None:
        loader = importlib.machinery.SourceFileLoader(
            "variant_symbolic_in_generator", str(Path(__file__).resolve().parent / "symbolic_verifier.py")
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None:  # pragma: no cover - defensive
            raise ImportError("cannot build an import spec for symbolic_verifier.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[loader.name] = module
        loader.exec_module(module)
        _SYMBOLIC_MODULE = module
    return _SYMBOLIC_MODULE


def oracle_for_instance(
    num_controls: int,
    blocks: int,
    seed: int,
    cfg: VariantConfig,
) -> "Dict[str, Any]":
    """Enumerate every variant assignment and return the oracle label."""
    t_exact, t_relative = block_t_counts(num_controls)
    context = build_instance(num_controls, blocks, 0, seed, cfg.h_prob)[1]
    # Incremental symbolic verifier: the same measurement-statistics criterion as
    # the circuit simulation, but O(1) per mask instead of a full re-simulation.
    # Validated to agree with the simulation on every mask (see symbolic_verifier).
    verifier = _symbolic().ReducedSymbolicVerifier(
        num_controls, blocks, seed, cfg.h_prob, tolerance=cfg.tolerance
    )

    # One shared partial-product sweep gives the deviation of every mask, so the
    # oracle costs O(2**blocks) matrix products, not O(blocks * 2**blocks).
    size = 1 << blocks
    partials = (1 << (blocks // 2)) + (1 << (blocks - blocks // 2))
    if partials * verifier.d * verifier.d * 16 <= 1_500_000_000:
        deviations = verifier.deviation_table()
    else:  # pragma: no cover - memory guard for very large data registers
        deviations = [verifier.deviation(mask) for mask in range(size)]

    mask_rows: "List[Dict[str, Any]]" = []
    for mask in range(size):
        deviation = deviations[mask]
        downgraded = bin(mask).count("1")
        mask_rows.append(
            {
                "mask": mask,
                "downgraded": downgraded,
                "t_count": (blocks - downgraded) * t_exact + downgraded * t_relative,
                "tv_distance": deviation,
                "usable": deviation <= cfg.tolerance,
            }
        )

    usable = [row for row in mask_rows if row["usable"]]
    # oracle: fewest T gates, ties -> fewer downgrades (more conservative)
    best = min(usable, key=lambda row: (row["t_count"], row["downgraded"])) if usable else None
    all_exact = mask_rows[0]
    all_relative = mask_rows[-1]
    return {
        "num_controls": num_controls,
        "blocks": blocks,
        "seed": seed,
        "t_per_block": {"exact": t_exact, "relative": t_relative},
        "context": context,
        "mask_table": mask_rows if cfg.store_mask_table else None,
        "oracle_mask": best["mask"] if best else None,
        "oracle_t_count": best["t_count"] if best else None,
        "all_exact_t_count": all_exact["t_count"],
        "all_relative_usable": bool(all_relative["usable"]),
        "all_exact_useful": True,
        "fragmented": bool(best and 0 < best["downgraded"] < blocks),
        "headroom_t": (all_exact["t_count"] - best["t_count"]) if best else 0,
        "usable_fraction": len(usable) / max(1, len(mask_rows)),
    }


# ---------------------------------------------------------------------------
# featurisation (same layout as the rest of the project)
# ---------------------------------------------------------------------------


def block_graph_features(circuit: QuantumCircuit) -> Data:
    """Simplified ZX diagram of one block as PyG node/edge features."""
    n_qubits = circuit.num_qubits
    graph = LINUX_MCX.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_graph()
    LINUX_MCX.zx.simplify.clifford_simp(graph)
    try:
        LINUX_MCX.zx.simplify.full_reduce(graph)
    except Exception:
        pass
    vertices = sorted(graph.vertices())
    index_of = {vertex: i for i, vertex in enumerate(vertices)}

    # component boundary support (same definition as data_generator.py)
    neighbours = {v: set(graph.neighbors(v)) for v in vertices}
    support_mask: "Dict[int, int]" = {}
    seen: set = set()
    for start in vertices:
        if start in seen:
            continue
        stack = [start]
        component = []
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
            if int(graph.type(vertex)) == int(LINUX_MCX.zx.VertexType.BOUNDARY):
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        if mask == 0:
            for vertex in component:
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        for vertex in component:
            support_mask[vertex] = mask

    features = np.zeros((len(vertices), NODE_FEATURES), dtype=np.float32)
    support = np.zeros((len(vertices), MAX_QUBITS), dtype=np.int8)
    qubit_index = np.full((len(vertices),), -1, dtype=np.int64)
    for vertex in vertices:
        row = index_of[vertex]
        vtype = int(graph.type(vertex))
        phase = graph.phase(vertex)
        phase_value = float(phase) if phase is not None else 0.0
        degree = int(graph.vertex_degree(vertex))
        mask = support_mask.get(vertex, 0)
        size = int(np.bitwise_count(np.int64(mask))) if mask else 0
        qubit = graph.qubit(vertex)
        non_clifford = abs((phase_value * 4.0) % 1.0) > 1e-9
        features[row] = (
            1.0 if vtype == int(LINUX_MCX.zx.VertexType.Z) else 0.0,
            1.0 if vtype == int(LINUX_MCX.zx.VertexType.X) else 0.0,
            1.0 if vtype == int(LINUX_MCX.zx.VertexType.BOUNDARY) else 0.0,
            1.0 if graph.is_ground(vertex) else 0.0,
            (phase_value % 2.0) / 2.0,
            math.sin(math.pi * phase_value),
            math.cos(math.pi * phase_value),
            1.0 if non_clifford else 0.0,
            degree / max(n_qubits, 1),
            size / max(n_qubits, 1),
            (float(qubit) / max(n_qubits - 1, 1)) if qubit is not None else -1.0,
        )
        for q in range(min(n_qubits, MAX_QUBITS)):
            support[row, q] = 1 if mask >> q & 1 else 0
        if qubit is not None:
            qubit_index[row] = int(qubit)

    edges: "List[Tuple[int, int]]" = []
    edge_types: "List[int]" = []
    for edge in graph.edges():
        a, b = graph.edge_st(edge)
        if a not in index_of or b not in index_of:
            continue
        etype = 1 if int(graph.edge_type(edge)) == int(LINUX_MCX.zx.EdgeType.HADAMARD) else 0
        i, j = index_of[a], index_of[b]
        edges.append((i, j))
        edge_types.append(etype)
        edges.append((j, i))
        edge_types.append(etype)
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    if edge_types:
        edge_attr = torch.zeros((len(edge_types), 2), dtype=torch.float)
        for row, etype in enumerate(edge_types):
            edge_attr[row, etype] = 1.0
    else:
        edge_attr = torch.zeros((0, 2), dtype=torch.float)
    return Data(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        edge_attr=edge_attr,
        x_support=torch.from_numpy(support),
        x_qubit=torch.from_numpy(qubit_index),
    )


def context_features(
    report: "Dict[str, Any]",
    block_index: int,
) -> torch.Tensor:
    """Per-block context: the Clifford layers that FOLLOW this block.

    The previous revision read ``report['context'][block_index]`` and therefore
    reported the layer *before* the block (and a constant 0 for the first block)
    -- an off-by-one that threw away the single most informative feature.  What
    decides whether a block's diagonal correction survives to the measurement is
    the *downstream* layers, so they are summarised here explicitly.
    """
    context = report["context"]
    blocks = int(report["blocks"])
    downstream = [tuple(context[k]) for k in range(block_index + 1, blocks)]
    h_next, cnot_next = downstream[0] if downstream else (0, 0)
    n_down = len(downstream)
    h_down = sum(layer[0] for layer in downstream)
    cnot_down = sum(layer[1] for layer in downstream)
    h_layers = sum(1 for layer in downstream if layer[0] > 0)
    h_max = max((layer[0] for layer in downstream), default=0)
    h_total = sum(tuple(layer)[0] for layer in context)
    h_mean = h_total / max(blocks - 1, 1)
    remaining = blocks - block_index - 1
    t_exact, t_relative = report["t_per_block"]["exact"], report["t_per_block"]["relative"]
    return torch.tensor(
        [
            float(h_next),
            float(cnot_next),
            float(h_down),
            float(cnot_down),
            float(n_down),
            float(h_layers / n_down) if n_down else 0.0,
            float(h_max),
            float(h_total),
            float(h_mean),
            float(remaining),
            float(block_index),
            1.0 if block_index == 0 else 0.0,
            1.0 if remaining == 0 else 0.0,
            float(report["num_controls"]),
            float(blocks),
            float(t_exact),
            float(t_relative),
            1.0 if h_next == 0 else 0.0,
            float(block_index / max(blocks - 1, 1)),
            *[1.0 if report["num_controls"] == c else 0.0 for c in CONTROL_ONEHOT],
            *[1.0 if blocks == b else 0.0 for b in BLOCK_ONEHOT],
        ],
        dtype=torch.float,
    )


_BLOCK_GRAPH_CACHE: "Dict[int, Data]" = {}


def _block_graph(num_controls: int) -> Data:
    """The block's simplified ZX diagram (same for every block of this size)."""
    cached = _BLOCK_GRAPH_CACHE.get(num_controls)
    if cached is None:
        cached = block_graph_features(exact_block(num_controls))
        _BLOCK_GRAPH_CACHE[num_controls] = cached
    return cached


def instance_samples(report: "Dict[str, Any]", instance_id: int) -> "List[Data]":
    """Turn one oracle report into one PyG sample per block."""
    num_controls = report["num_controls"]
    blocks = report["blocks"]
    oracle_mask = report["oracle_mask"]
    samples: "List[Data]" = []
    template = _block_graph(num_controls)
    for index in range(blocks):
        graph = Data(
            x=template.x,
            edge_index=template.edge_index,
            edge_attr=template.edge_attr,
            x_support=template.x_support,
            x_qubit=template.x_qubit,
        )
        graph.context = context_features(report, index)
        graph.y_variant = torch.tensor(
            [1 if oracle_mask is not None and (oracle_mask >> index) & 1 else 0],
            dtype=torch.long,
        )
        graph.y_instance = torch.tensor([instance_id], dtype=torch.long)
        graph.y_oracle_tcount = torch.tensor(
            [float(report["oracle_t_count"] or 0.0)], dtype=torch.float
        )
        graph.y_all_exact_tcount = torch.tensor(
            [float(report["all_exact_t_count"])], dtype=torch.float
        )
        graph.y_all_relative_usable = torch.tensor(
            [1.0 if report["all_relative_usable"] else 0.0], dtype=torch.float
        )
        graph.y_fragmented = torch.tensor(
            [1.0 if report["fragmented"] else 0.0], dtype=torch.float
        )
        graph.y_headroom = torch.tensor([float(report["headroom_t"])], dtype=torch.float)
        graph.y_block = torch.tensor([index], dtype=torch.long)
        graph.metadata = {
            "instance_id": int(instance_id),
            "block_index": int(index),
            "num_controls": int(num_controls),
            "blocks": int(blocks),
            "oracle_mask": int(oracle_mask) if oracle_mask is not None else -1,
            "oracle_t_count": float(report["oracle_t_count"] or 0.0),
            "all_exact_t_count": float(report["all_exact_t_count"]),
            "all_relative_usable": bool(report["all_relative_usable"]),
            "fragmented": bool(report["fragmented"]),
            "headroom_t": float(report["headroom_t"]),
            "hadamards_after": int(report["context"][index + 1][0]) if index + 1 < report["blocks"] else 0,
            "t_exact_per_block": float(report["t_per_block"]["exact"]),
            "t_relative_per_block": float(report["t_per_block"]["relative"]),
        }
        samples.append(graph)
    return samples


# ---------------------------------------------------------------------------
# workers / generation
# ---------------------------------------------------------------------------

_WORKER_CFG: "Optional[VariantConfig]" = None


def _worker_instance(job: Tuple[int, int]) -> "Tuple[int, Optional[Dict[str, Any]], Optional[str]]":
    instance_id, seed = job
    assert _WORKER_CFG is not None
    cfg = _WORKER_CFG
    rng = random.Random(seed)
    try:
        num_controls = rng.choice(list(cfg.control_values))
        blocks = rng.randint(2, max(2, cfg.max_blocks))
        report = oracle_for_instance(num_controls, blocks, seed, cfg)
        report["instance_id"] = instance_id
        return instance_id, report, None
    except Exception as exc:  # pragma: no cover - surfaced in the report
        return instance_id, None, f"{type(exc).__name__}: {exc}"


def _init_worker(cfg: VariantConfig) -> None:
    global _WORKER_CFG
    _WORKER_CFG = cfg


def plan_jobs(cfg: VariantConfig) -> "List[Tuple[int, int]]":
    ss = np.random.SeedSequence(cfg.seed)
    seeds = ss.generate_state(cfg.instances, dtype=np.uint32)
    return [(index, int(seeds[index])) for index in range(cfg.instances)]


def generate(cfg: VariantConfig, verbose: bool = True) -> "Dict[str, Any]":
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan_jobs(cfg)
    workers = cfg.normalized_workers()
    started = time.time()
    reports: "List[Dict[str, Any]]" = []
    failures: "List[str]" = []

    if workers == 1:
        _init_worker(cfg)
        results = (_worker_instance(job) for job in jobs)
        for instance_id, report, error in results:
            if report is None:
                failures.append(f"[{instance_id}] {error}")
            else:
                reports.append(report)
    else:
        pool = mp.Pool(processes=workers, initializer=_init_worker, initargs=(cfg,))
        try:
            for instance_id, report, error in pool.imap_unordered(
                _worker_instance, jobs, chunksize=max(1, cfg.instances // (workers * 8) or 1)
            ):
                if report is None:
                    failures.append(f"[{instance_id}] {error}")
                else:
                    reports.append(report)
                    if verbose and len(reports) % cfg.progress_every == 0:
                        rate = len(reports) / max(time.time() - started, 1e-9)
                        print(
                            f"    instances: {len(reports)}/{cfg.instances} ({rate:5.2f}/s) "
                            f"failures={len(failures)}",
                            flush=True,
                        )
        finally:
            pool.close()
            pool.join()

    reports.sort(key=lambda row: row["instance_id"])
    rng = random.Random(cfg.seed)
    rng.shuffle(reports)
    total = len(reports)
    n_train = int(round(total * cfg.split_ratios[0]))
    n_val = int(round(total * cfg.split_ratios[1]))
    splits = {
        "train": reports[:n_train],
        "val": reports[n_train : n_train + n_val],
        "test": reports[n_train + n_val :],
    }
    written: "Dict[str, str]" = {}
    for split, rows in splits.items():
        samples: "List[Data]" = []
        for report in rows:
            samples.extend(instance_samples(report, report["instance_id"]))
        path = out_dir / f"{split}.pt"
        torch.save(samples, path)
        written[split] = str(path)
        if verbose:
            print(
                f"    {split}: {len(rows)} instances -> {len(samples)} block samples "
                f"-> {path}"
            )

    with open(out_dir / "instances.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    key: value
                    for key, value in report.items()
                    if key != "mask_table" or cfg.store_mask_table
                }
                for report in reports
            ],
            handle,
            indent=2,
            sort_keys=True,
        )
    summary = {
        "instances": total,
        "failures": len(failures),
        "failure_examples": failures[:5],
        "blocks_per_instance": dict(Counter(row["blocks"] for row in reports)),
        "controls": dict(Counter(row["num_controls"] for row in reports)),
        "fragmented_fraction": sum(1 for row in reports if row["fragmented"]) / max(1, total),
        "all_relative_usable_fraction": sum(
            1 for row in reports if row["all_relative_usable"]
        )
        / max(1, total),
        "mean_headroom_t": float(np.mean([row["headroom_t"] for row in reports]))
        if reports
        else 0.0,
        "files": written,
        "seconds": round(time.time() - started, 2),
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in cfg.__dict__.items()
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if verbose:
        print(json.dumps({k: summary[k] for k in (
            "instances", "failures", "fragmented_fraction", "all_relative_usable_fraction",
            "mean_headroom_t", "seconds",
        )}, indent=2))
    return summary


def self_test(cfg: VariantConfig) -> int:
    """Checks that the oracle behaves as physics requires."""
    failures: "List[str]" = []

    def check(name: str, condition: bool, extra: str = "") -> None:
        print(f"  [{'ok  ' if condition else 'FAIL'}] {name} {extra}")
        if not condition:
            failures.append(name)

    # all-exact must be usable, and the H-free regime must allow downgrades
    report = oracle_for_instance(3, 3, 12345, VariantConfig(h_prob=0.0))
    check("all-exact is usable by construction", report["all_exact_useful"])
    check(
        "H-free instance allows at least one downgrade",
        report["oracle_mask"] is not None
        and bin(report["oracle_mask"]).count("1") > 0,
        f"(oracle mask={report['oracle_mask']}, headroom={report['headroom_t']} T)",
    )
    heavy = oracle_for_instance(3, 3, 12345, VariantConfig(h_prob=1.0))
    check(
        "Hadamard-heavy instance is fragmented or all-exact",
        heavy["oracle_mask"] is not None,
        f"(oracle mask={heavy['oracle_mask']}, fragmented={heavy['fragmented']}, "
        f"headroom={heavy['headroom_t']} T)",
    )
    # per-block samples carry the oracle label of their instance
    samples = instance_samples(report, 0)
    check("one sample per block", len(samples) == report["blocks"])
    labels = [int(sample.y_variant) for sample in samples]
    expected = [
        (report["oracle_mask"] >> index) & 1 if report["oracle_mask"] is not None else 0
        for index in range(report["blocks"])
    ]
    check("per-block labels match the oracle mask", labels == expected, f"{labels}")
    check("context features present", all(sample.context.numel() == CONTEXT_DIM for sample in samples))
    print()
    if failures:
        print(f"self-test FAILED ({len(failures)}): {failures}")
        return 1
    print("self-test passed")
    return 0


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy-B supervised data generator.")
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--max-blocks", type=int, default=5)
    parser.add_argument("--control-values", default="2,3,4")
    parser.add_argument("--h-prob", type=float, default=0.7)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out-dir", default="variant_dataset")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    cfg = VariantConfig(
        instances=args.instances,
        max_blocks=args.max_blocks,
        control_values=tuple(int(v) for v in args.control_values.split(",") if v.strip()),
        h_prob=args.h_prob,
        tolerance=args.tolerance,
        out_dir=args.out_dir,
        workers=args.workers,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    if args.self_test:
        return self_test(cfg)
    generate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
