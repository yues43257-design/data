#!/usr/bin/env python3
"""Large-scale MIXED-control-count stress test for the strategy-B policy.

The dataset and the earlier stress tests used *homogeneous* instances: every
block had the same control count, so the whole circuit lived on one native
``C^nX`` layout.  This module lifts that restriction.

Layout
------
All blocks share one data register:

    controls  0 .. n_max-1          (data)
    target    n_max                 (data)
    ancillas  n_max+1 .. total-1    (clean, and **reused by every block**)

A block with ``n`` controls is the native ``C^nX`` ladder relabelled into that
register (its own ancillas map onto the shared pool, its target onto ``n_max``),
so a circuit can mix ``C^2X`` ... ``C^8X`` blocks while the whole instance still
lives on ``n_max+1`` data qubits.  That keeps the *reduced* verifier applicable:
it works in ``d = 2**(n_max+1)`` and the safety criterion is again the Z-basis
measurement distribution of the data register.

Verification
------------
``MixedVerifier`` generalises the reduced verifier to per-block corrections:
with ``C~_i = Q_i D_i Q_i^dagger`` (``Q_i`` = unitaries after block ``i``,
``D_i`` = that block's diagonal correction) the perturbed circuit is
``Omega_S . U`` with ``Omega_S = prod_{i in S, ascending} C~_i``, and a mask is
safe iff the post-selected distributions of ``Omega_S U`` and ``U`` agree.
``--self-test`` checks this against the full circuit simulation on every mask.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parent
CONTROL_CHOICES = (2, 3, 4, 5, 6, 7, 8)


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


GEN = _load("variant_generator_mixed", "generate_variant_dataset.py")
SYM = _load("variant_symbolic_mixed", "symbolic_verifier.py")
TRAINER = _load("variant_trainer_mixed", "train_variant_policy.py")


# ---------------------------------------------------------------------------
# mixed layout / circuit construction
# ---------------------------------------------------------------------------


def mixed_layout(control_counts: Sequence[int]):
    """(n_max, total qubits, data qubits, ancilla qubits) for a mixed instance."""
    n_max = max(control_counts)
    n_ancillas = max(0, n_max - 2)
    total = n_max + 1 + n_ancillas
    data_qubits = list(range(n_max + 1))          # controls 0..n_max-1, target n_max
    ancillas = list(range(n_max + 1, total))      # shared pool, reused by every block
    return n_max, total, data_qubits, ancillas


def block_remap(num_controls: int, n_max: int) -> Dict[int, int]:
    """Native ``C^nX`` qubits -> the shared mixed register."""
    n_ancillas = max(0, num_controls - 2)
    mapping = {q: q for q in range(num_controls)}          # controls
    for i in range(n_ancillas):
        mapping[num_controls + i] = n_max + 1 + i          # ancillas
    mapping[num_controls + n_ancillas] = n_max             # target
    return mapping


def _apply_gate(target: QuantumCircuit, name: str, qubits: Sequence[int]) -> None:
    if name == "cx":
        target.cx(qubits[0], qubits[1])
    elif name == "cz":
        target.cz(qubits[0], qubits[1])
    else:
        getattr(target, name)(qubits[0])


def remap_circuit(circuit: QuantumCircuit, mapping: Dict[int, int], total: int) -> QuantumCircuit:
    out = QuantumCircuit(total)
    for instruction in circuit.data:
        qubits = [mapping[circuit.find_bit(q).index] for q in instruction.qubits]
        _apply_gate(out, instruction.operation.name, qubits)
    return out


def _data_matrix(circuit: QuantumCircuit, width: int) -> np.ndarray:
    """Matrix of a gate list that only touches qubits ``0..width-1``."""
    out = QuantumCircuit(width)
    for instruction in circuit.data:
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        _apply_gate(out, instruction.operation.name, qubits)
    return np.asarray(Operator(out).data, dtype=complex)


def build_mixed_circuit(control_counts: Sequence[int], seed: int, h_prob: float,
                        mask: int = 0):
    """Circuit, per-layer ``(Hadamards, CNOTs)`` stats, and the layer circuits."""
    counts = [int(c) for c in control_counts]
    n_max, total, data_qubits, _ = mixed_layout(counts)
    rng = random.Random(int(seed))
    circuit = QuantumCircuit(total)
    stats: List[Tuple[int, int]] = []
    layers: List[Optional[QuantumCircuit]] = [None] * len(counts)
    for index, n in enumerate(counts):
        if index:
            layer = QuantumCircuit(total)
            stats.append(GEN.clifford_layer(layer, data_qubits, rng, h_prob))
            layers[index] = layer
            circuit.compose(layer, range(total), inplace=True)
        else:
            stats.append((0, 0))
        block = GEN.relative_block(n) if (mask >> index) & 1 else GEN.exact_block(n)
        circuit.compose(
            remap_circuit(block, block_remap(n, n_max), total), range(total), inplace=True
        )
    return circuit, stats, layers


def block_diagonal(num_controls: int, n_max: int) -> np.ndarray:
    """Diagonal ``D`` with ``B_relative = D . B_exact`` on the mixed data register.

    Each RCCX is a Toffoli followed by the 2-local phase
    ``phi(a,b,c) = (-1)**(a*c) * (-i)**(a*b)``; the ladder is a permutation, so a
    classical reversible sweep reads ``D`` off in ``O(2**(n_max+1) * #gates)``.
    """
    n_ancillas = max(0, num_controls - 2)
    total = n_max + 1 + max(0, n_max - 2)
    size = 2 ** (n_max + 1)
    mapping = block_remap(num_controls, n_max)
    ladder = [
        (mapping[a], mapping[b], mapping[c]) for a, b, c in SYM._rccx_ladder(num_controls)
    ]
    diagonal = np.ones(size, dtype=complex)
    for y in range(size):
        bits = [0] * total
        for q in range(n_max + 1):
            bits[q] = (y >> q) & 1
        phase = 1.0 + 0.0j
        for a, b, c in ladder:
            bits[c] ^= bits[a] & bits[b]
            va, vb, vc = bits[a], bits[b], bits[c]
            if va and vb:
                phase *= -1j
            if va and vc:
                phase *= -1.0
        out = 0
        for q in range(n_max + 1):
            if bits[q]:
                out |= 1 << q
        diagonal[out] = phase
    return diagonal


# ---------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------


class MixedVerifier:
    """Per-block reduced verifier for a heterogeneous control-count instance."""

    def __init__(self, control_counts: Sequence[int], seed: int, h_prob: float,
                 tolerance: float = 1e-9):
        self.counts = [int(c) for c in control_counts]
        self.k = len(self.counts)
        n_max, total, data_qubits, _ = mixed_layout(self.counts)
        self.n_max = n_max
        self.total = total
        self.data_qubits = data_qubits
        self.d = 2 ** (n_max + 1)
        self.tolerance = float(tolerance)

        self.perms: List[np.ndarray] = []
        self.diags: List[np.ndarray] = []
        for n in self.counts:
            low = (1 << n) - 1
            self.perms.append(
                np.array(
                    [y ^ ((1 if (y & low) == low else 0) << n_max) for y in range(self.d)],
                    dtype=int,
                )
            )
            self.diags.append(block_diagonal(n, n_max))

        _, _, layer_circuits = build_mixed_circuit(self.counts, seed, h_prob, 0)
        self.layer_matrices: List[Optional[np.ndarray]] = [None] * self.k
        for j in range(1, self.k):
            self.layer_matrices[j] = _data_matrix(layer_circuits[j], n_max + 1)

        eye = np.eye(self.d, dtype=complex)
        unitary = eye[self.perms[0]]
        for j in range(1, self.k):
            unitary = self.layer_matrices[j] @ unitary
            unitary = unitary[self.perms[j]]
        self.unitary = unitary

        suffix = eye.copy()
        suffixes: Dict[int, np.ndarray] = {self.k - 1: suffix}
        for i in range(self.k - 2, -1, -1):
            suffix = suffix[:, self.perms[i + 1]] @ self.layer_matrices[i + 1]
            suffixes[i] = suffix.copy()
        self.corrections = [
            suffixes[i] @ np.diag(self.diags[i]) @ suffixes[i].conj().T
            for i in range(self.k)
        ]
        self.reference = np.abs(unitary) ** 2
        self._cache: Dict[int, bool] = {0: True}
        self._tcounts = [GEN.block_t_counts(n) for n in self.counts]

    # -- criterion ----------------------------------------------------------

    def omega(self, mask: int) -> np.ndarray:
        operator = np.eye(self.d, dtype=complex)
        for i in range(self.k):
            if (mask >> i) & 1:
                operator = self.corrections[i] @ operator
        return operator

    def deviation(self, mask: int) -> float:
        if mask == 0:
            return 0.0
        magnitudes = np.abs(self.omega(mask) @ self.unitary) ** 2
        return float(np.max(0.5 * np.sum(np.abs(magnitudes - self.reference), axis=0)))

    def is_safe(self, mask: int) -> bool:
        mask = int(mask)
        cached = self._cache.get(mask)
        if cached is None:
            cached = self.deviation(mask) <= self.tolerance
            self._cache[mask] = cached
        return cached

    # -- decoding / cost ----------------------------------------------------

    def mask_from_probs(self, probs: Sequence[float], threshold: float = 0.5) -> int:
        mask = 0
        for index, prob in enumerate(probs[: self.k]):
            if prob >= threshold:
                mask |= 1 << index
        return mask

    def greedy(self, probs: Sequence[float], threshold: float = 0.0) -> int:
        order = sorted(
            (i for i in range(self.k) if probs[i] >= threshold), key=lambda i: -probs[i]
        )
        accepted = 0
        for block in order:
            candidate = accepted | (1 << block)
            if self.is_safe(candidate):
                accepted = candidate
        return accepted

    def t_count(self, mask: int) -> int:
        total = 0
        for i, (t_exact, t_relative) in enumerate(self._tcounts):
            total += t_relative if (mask >> i) & 1 else t_exact
        return int(total)

    def all_exact_t(self) -> int:
        return self.t_count(0)

    def all_relative_t(self) -> int:
        return self.t_count((1 << self.k) - 1)


# ---------------------------------------------------------------------------
# policy inputs
# ---------------------------------------------------------------------------


def _policy_data(sample: Data) -> Data:
    return Data(
        x=torch.cat([sample.x.float(), sample.x_support.float()], dim=-1),
        edge_index=sample.edge_index,
        edge_attr=sample.edge_attr,
        context=sample.context.view(1, -1).float(),
    )


class MixedInstance:
    """A mixed instance: policy inputs (one per block) + verifier + costs."""

    def __init__(self, control_counts: Sequence[int], seed: int, h_prob: float,
                 tolerance: float = 1e-9):
        self.counts = [int(c) for c in control_counts]
        self.seed = int(seed)
        self.verifier = MixedVerifier(self.counts, seed, h_prob, tolerance)
        _, stats, _ = build_mixed_circuit(self.counts, seed, h_prob, 0)
        self.layer_stats = stats
        self.blocks_data: List[Data] = []
        for index, n in enumerate(self.counts):
            t_exact, t_relative = GEN.block_t_counts(n)
            report = {
                "context": stats,
                "blocks": len(self.counts),
                "num_controls": n,
                "t_per_block": {"exact": t_exact, "relative": t_relative},
            }
            graph = GEN._block_graph(n)
            sample = Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                x_support=graph.x_support,
            )
            sample.context = GEN.context_features(report, index)
            self.blocks_data.append(_policy_data(sample))

    def probs(self, model, device) -> List[float]:
        batch = Batch.from_data_list(self.blocks_data).to(device)
        with torch.no_grad():
            return torch.sigmoid(model(batch)).tolist()


# ---------------------------------------------------------------------------
# stress test
# ---------------------------------------------------------------------------


def _control_overview(counts: Sequence[int]) -> str:
    histogram = Counter(counts)
    return " ".join(f"C{n}:{histogram[n]}" for n in sorted(histogram))


def depth_budget(blocks: int) -> int:
    """Depth-conditioned downgrade budget ``k_max = clamp(round(4/sqrt(blocks)), 1, blocks)``.

    The report's ``k_max = rho * blocks`` idea with ``rho`` fitted to the data; the
    square-root shape keeps shallow circuits at 2-3 blocks (where the one-shot is
    already safe) and drops deep circuits to a single block.  Measured on held-out
    circuits: see ``reports/strategy_b_experiments.md`` §4.
    """
    return max(1, min(int(blocks), int(round(4.0 / (int(blocks) ** 0.5)))))


def top_mask(probs: Sequence[float], budget: int) -> int:
    """Mask selecting the ``budget`` most confident blocks."""
    order = sorted(range(len(probs)), key=lambda i: -probs[i])[: max(0, int(budget))]
    mask = 0
    for block in order:
        mask |= 1 << block
    return mask


def repair_mask(verifier: "MixedVerifier", probs: Sequence[float], mask: int):
    """Salvage an unsafe proposal by dropping its least-confident downgrades.

    A raw threshold mask is often unsafe (the block scores ignore the *cumulative*
    effect), and plain fallback then throws away every saving.  Instead, remove
    the accepted downgrades one at a time, least confident first, and stop as soon
    as the mask is safe again.  ``mask = 0`` is always safe, so this terminates and
    never returns an unsafe mask.  Returns ``(mask, drops)``.
    """
    if verifier.is_safe(mask):
        return int(mask), 0
    n_blocks = int(getattr(verifier, "blocks", None) or getattr(verifier, "k"))
    drops = 0
    for block in sorted(
        (i for i in range(n_blocks) if (mask >> i) & 1), key=lambda i: probs[i]
    ):
        mask &= ~(1 << block)
        drops += 1
        if verifier.is_safe(mask):
            return int(mask), drops
    return 0, drops


def run_stress(model, device, num_tests: int = 10, control_choices: Sequence[int] = CONTROL_CHOICES,
               min_blocks: int = 30, max_blocks: int = 60, seed: int = 20261012,
               h_prob: float = 0.7, threshold: float = 0.5, log=print) -> Dict[str, Any]:
    """Run the mixed large-circuit battle report and return the summary."""
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for test in range(1, num_tests + 1):
        blocks = int(rng.integers(min_blocks, max_blocks + 1))
        counts = [int(rng.choice(control_choices)) for _ in range(blocks)]
        instance_seed = int(rng.integers(1, 2 ** 31 - 1))
        started = time.perf_counter()
        instance = MixedInstance(counts, instance_seed, h_prob)
        probs = instance.probs(model, device)
        verifier = instance.verifier

        exact_t = verifier.all_exact_t()
        relative_mask = (1 << blocks) - 1
        relative_t = verifier.all_relative_t()
        relative_safe = verifier.is_safe(relative_mask)

        greedy_mask = verifier.greedy(probs)
        greedy_safe = verifier.is_safe(greedy_mask)
        greedy_t = verifier.t_count(greedy_mask)

        one_shot_mask = verifier.mask_from_probs(probs, threshold)
        one_shot_safe = verifier.is_safe(one_shot_mask)
        one_shot_raw_t = verifier.t_count(one_shot_mask)
        # old behaviour: give up entirely when the raw mask is unsafe
        one_shot_fallback_t = one_shot_raw_t if one_shot_safe else exact_t
        # new behaviour: drop the least-confident downgrades until it is safe
        repaired_mask, one_shot_drops = repair_mask(verifier, probs, one_shot_mask)
        one_shot_t = verifier.t_count(repaired_mask)
        one_shot_downgrades = bin(repaired_mask).count("1")

        # depth-conditioned ablation: hard cap the downgrade count by k_max(blocks)
        budget = depth_budget(blocks)
        depth_mask = top_mask(probs, budget)
        depth_safe = verifier.is_safe(depth_mask)
        depth_repaired, depth_drops = repair_mask(verifier, probs, depth_mask)
        depth_t = verifier.t_count(depth_repaired)

        row = {
            "test": test,
            "blocks": blocks,
            "counts": counts,
            "seed": instance_seed,
            "exact_t": exact_t,
            "relative_t": relative_t,
            "relative_safe": relative_safe,
            "greedy_t": greedy_t,
            "greedy_safe": greedy_safe,
            "greedy_downgrades": bin(greedy_mask).count("1"),
            "greedy_mask": greedy_mask,
            "greedy_dist_from_end": sorted(
                blocks - 1 - i for i in range(blocks) if (greedy_mask >> i) & 1
            ),
            "one_shot_t": one_shot_t,
            "one_shot_raw_t": one_shot_raw_t,
            "one_shot_fallback_t": one_shot_fallback_t,
            "one_shot_safe": one_shot_safe,
            "one_shot_drops": one_shot_drops,
            "one_shot_raw_downgrades": bin(one_shot_mask).count("1"),
            "one_shot_downgrades": one_shot_downgrades,
            "depth_t": depth_t,
            "depth_safe": depth_safe,
            "depth_budget": budget,
            "depth_drops": depth_drops,
            "depth_downgrades": bin(depth_repaired).count("1"),
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)

        head = f"  混合压力测试 {test:2d}/{num_tests}"
        log("")
        log("═" * 78)
        log(f"{head}   区块数 = {blocks}   实例种子 = {instance_seed}")
        log(f"  控制数分布   {_control_overview(counts)}")
        log(f"  控制数序列   前12 {counts[:12]}   …   后6 {counts[-6:]}")
        log("─" * 78)
        log(f"  {'策略':<34}{'T-count':>10}{'省下 T':>10}{'安全性':>12}")
        log(f"  {'all-exact (全精确基线)':<34}{exact_t:>10}{'—':>10}{'—':>12}")
        log(
            f"  {'all-relative (全相对, 非法)':<34}{relative_t:>10}"
            f"{exact_t - relative_t:>10}{('安全' if relative_safe else '不安全'):>12}"
        )
        log(
            f"  {'验证器贪心 (greedy+safe)':<34}{greedy_t:>10}{exact_t - greedy_t:>10}"
            f"{('安全' if greedy_safe else '不安全'):>12}"
        )
        one_shot_state = "安全" if one_shot_drops == 0 else f"修复-{one_shot_drops}"
        log(
            f"  {'RL 一次性 (threshold+repair)':<34}{one_shot_t:>10}{exact_t - one_shot_t:>10}"
            f"{one_shot_state:>12}"
        )
        log(
            f"  {'  └ 原始阈值掩码 (未验证)':<34}{one_shot_raw_t:>10}"
            f"{exact_t - one_shot_raw_t:>10}{('安全' if one_shot_safe else '不安全'):>12}"
        )
        log(
            f"  {'RL 一次性 (深度条件化上限+修复)':<34}{depth_t:>10}{exact_t - depth_t:>10}"
            f"{('安全' if depth_safe else f'修复-{depth_drops}'):>12}"
        )
        log(
            f"  {'  └ 上限 k_max(blocks)':<34}{'':>10}{'':>10}{f'k_max={budget}':>12}"
        )
        log("─" * 78)
        log(
            f"  降级区块数  贪心 {row['greedy_downgrades']}/{blocks} | "
            f"一次性 {row['one_shot_downgrades']}/{blocks}"
            f"（修复丢弃 {row['one_shot_drops']} 个，原始阈值 {row['one_shot_raw_downgrades']}/{blocks}）"
            f"  |  耗时 {row['seconds']:.1f}s"
        )
        log(
            f"  一次性收益  原始阈值 {exact_t - one_shot_raw_t} T"
            f"（{'安全' if one_shot_safe else '非法'}） → "
            f"纯回退 {exact_t - one_shot_fallback_t} T → "
            f"修复后 {exact_t - one_shot_t} T"
        )
        log(
            f"  贪心降级位置（距末端） {row['greedy_dist_from_end']}  "
            "← 末端区块没有下游，恒安全；其余只有当下游恰好保持修正对角时才可降级"
        )

    summary = {
        "tests": len(rows),
        "h_prob": float(h_prob),
        "min_blocks": int(min_blocks),
        "max_blocks": int(max_blocks),
        "control_choices": [int(c) for c in control_choices],
        "seed": int(seed),
        "mean_blocks": float(np.mean([r["blocks"] for r in rows])),
        "mean_exact_t": float(np.mean([r["exact_t"] for r in rows])),
        "mean_relative_t": float(np.mean([r["relative_t"] for r in rows])),
        "mean_greedy_t": float(np.mean([r["greedy_t"] for r in rows])),
        "mean_one_shot_t": float(np.mean([r["one_shot_t"] for r in rows])),
        "greedy_safe_rate": float(np.mean([r["greedy_safe"] for r in rows])),
        "one_shot_safe_rate": float(np.mean([r["one_shot_safe"] for r in rows])),
        "relative_safe_rate": float(np.mean([r["relative_safe"] for r in rows])),
        "greedy_saving": float(np.mean([r["exact_t"] - r["greedy_t"] for r in rows])),
        "one_shot_saving": float(np.mean([r["exact_t"] - r["one_shot_t"] for r in rows])),
        "one_shot_raw_saving": float(
            np.mean([r["exact_t"] - r["one_shot_raw_t"] for r in rows])
        ),
        "one_shot_fallback_saving": float(
            np.mean([r["exact_t"] - r["one_shot_fallback_t"] for r in rows])
        ),
        "one_shot_repair_rate": float(np.mean([r["one_shot_drops"] > 0 for r in rows])),
        "mean_depth_t": float(np.mean([r["depth_t"] for r in rows])),
        "depth_saving": float(np.mean([r["exact_t"] - r["depth_t"] for r in rows])),
        "depth_raw_safe_rate": float(np.mean([r["depth_safe"] for r in rows])),
        "depth_repair_rate": float(np.mean([r["depth_drops"] > 0 for r in rows])),
        "mean_one_shot_drops": float(np.mean([r["one_shot_drops"] for r in rows])),
    }
    best = summary["mean_exact_t"] - summary["mean_relative_t"]
    summary["greedy_share_of_legal_max"] = (
        summary["greedy_saving"] / best if best else float("nan")
    )
    summary["one_shot_share_of_legal_max"] = (
        summary["one_shot_saving"] / best if best else float("nan")
    )
    log("")
    log("═" * 78)
    log(f"  汇总（{len(rows)} 条混合大电路，平均 {summary['mean_blocks']:.0f} 区块）")
    log(
        f"  测试配置: h_prob = {h_prob:g} | 区块数 {min_blocks}..{max_blocks} | "
        f"控制数 {list(control_choices)} | seed {seed}"
    )
    log(
        f"  all-exact 均值 = {summary['mean_exact_t']:.1f} T   |   "
        f"all-relative 均值 = {summary['mean_relative_t']:.1f} T（安全率 "
        f"{summary['relative_safe_rate']:.0%}）"
    )
    log(
        f"  验证器贪心均值 = {summary['mean_greedy_t']:.1f} T "
        f"(省 {summary['greedy_saving']:.1f}, 安全率 {summary['greedy_safe_rate']:.0%})"
    )
    log(
        f"  RL 一次性均值  = {summary['mean_one_shot_t']:.1f} T "
        f"(省 {summary['one_shot_saving']:.1f}; 部署掩码恒安全，"
        f"原始阈值掩码安全率 {summary['one_shot_safe_rate']:.0%})"
    )
    log(
        f"    深度条件化上限（k_max=4/sqrt(blocks)）：原始安全率 "
        f"{summary['depth_raw_safe_rate']:.0%}，省 {summary['depth_saving']:.1f} T"
        f"（{summary['depth_repair_rate']:.0%} 需修复）"
    )
    log(
        f"    一次性修复对比：原始阈值省 {summary['one_shot_raw_saving']:.1f} T（非法） → "
        f"纯回退省 {summary['one_shot_fallback_saving']:.1f} T → 修复后省 "
        f"{summary['one_shot_saving']:.1f} T（{summary['one_shot_repair_rate']:.0%} 的电路需要修复，"
        f"平均丢弃 {summary['mean_one_shot_drops']:.1f} 个区块）"
    )
    log(
        f"  相对合法上界的捕获率: 贪心 {summary['greedy_share_of_legal_max']:.1%} | "
        f"一次性 {summary['one_shot_share_of_legal_max']:.1%}"
    )
    tail = [d for r in rows for d in r["greedy_dist_from_end"]]
    if tail:
        log(
            f"  贪心接受的降级区块距末端 {{min: {min(tail)}, 中位: {int(np.median(tail))}, "
            f"max: {max(tail)}}}（距末端 0 = 最后一个区块，无下游、恒安全）"
        )
    log("═" * 78)
    return {"summary": summary, "rows": rows}


def load_policy(checkpoint: Path, device):
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TRAINER.BlockVariantPolicy(
        node_dim=int(checkpoint.get("node_dim", 27)),
        context_dim=int(checkpoint.get("context_dim", GEN.CONTEXT_DIM)),
        hidden=int(checkpoint.get("hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# self test: the mixed criterion must equal the full simulation on every mask
# ---------------------------------------------------------------------------


def _simulated_safe(counts: Sequence[int], seed: int, h_prob: float, mask: int,
                    tolerance: float) -> bool:
    n_max, total, _, ancillas = mixed_layout(counts)
    keep = np.array([z for z in range(2 ** total) if all(not (z >> a) & 1 for a in ancillas)])
    candidate = np.asarray(Operator(build_mixed_circuit(counts, seed, h_prob, mask)[0]).data)
    reference = np.asarray(Operator(build_mixed_circuit(counts, seed, h_prob, 0)[0]).data)
    for basis in keep:
        p = np.abs(candidate[keep, basis]) ** 2
        q = np.abs(reference[keep, basis]) ** 2
        p = p / max(p.sum(), 1e-30)
        q = q / max(q.sum(), 1e-30)
        if 0.5 * np.sum(np.abs(p - q)) > tolerance:
            return False
    return True


def _self_test(seed: int = 0) -> int:
    cases = [
        [2, 3], [3, 2, 4], [2, 2, 3], [4, 2, 3, 2], [5, 2], [3, 3, 2, 4, 2],
    ]
    total_masks = 0
    mismatches = 0
    for index, counts in enumerate(cases):
        case_seed = 1000 + index
        verifier = MixedVerifier(counts, case_seed, 0.7)
        bad = 0
        for mask in range(1 << len(counts)):
            simulated = _simulated_safe(counts, case_seed, 0.7, mask, verifier.tolerance)
            total_masks += 1
            if simulated != verifier.is_safe(mask):
                bad += 1
        mismatches += bad
        print(f"  [{'ok  ' if bad == 0 else 'FAIL'}] mixed {counts}: "
              f"{1 << len(counts)} masks, mismatches={bad}")
    print()
    if mismatches:
        print(f"mixed self-test FAILED ({mismatches}/{total_masks})")
        return 1
    print(f"mixed self-test passed ({total_masks} masks matched the full simulation)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mixed control-count stress test.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--num-tests", type=int, default=10)
    parser.add_argument("--min-blocks", type=int, default=30)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--h-prob", type=float, default=0.7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    device = torch.device(args.device)
    model, _ = load_policy(Path(args.checkpoint), device)
    run_stress(
        model, device, num_tests=args.num_tests, min_blocks=args.min_blocks,
        max_blocks=args.max_blocks, seed=args.seed, h_prob=args.h_prob,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
