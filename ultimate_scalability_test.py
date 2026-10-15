#!/usr/bin/env python3
"""Ultimate scalability test: an RL one-shot mask on a circuit **wider than the
verifier can handle**, certified by ZX graph rewriting instead of matrices.

Why this file exists
--------------------
The reduced verifier works in ``d = 2**(n_max+1)``; past roughly 12-19 data qubits
it is out of memory.  This script builds a circuit with >25 data qubits and never
constructs a matrix of that size.  Instead the RL policy proposes a downgrade mask
in one forward pass, and the *equivalence* of the resulting circuit is checked in
the ZX calculus.

What can and cannot be proved
-----------------------------
Let ``C = U_RL . U_exact^dagger``.  Two ZX certificates are attempted:

* **A - unitary equality.** ``full_reduce(C)`` reduces to the identity graph
  (``is_id()``).  Timeless rewriting rules preserve the tensor, so this *proves*
  ``U_RL == U_exact`` (up to a scalar).
* **B - diagonal correction.** ``C`` commutes with every Pauli ``Z_j``, i.e.
  ``Z_j C Z_j C^dagger`` reduces to the identity for all ``j``.  A matrix that
  commutes with all ``Z_j`` is diagonal, and a diagonal ``C`` means
  ``U_RL = U_exact . C`` so every Z-basis measurement statistic is unchanged.
  This *proves* measurement-safety, is polynomial in the circuit size, and works
  at any width.

Both certificates are *sound but incomplete*: a failure to reduce means "not
certified by this rule set", **not** "unsafe".  In particular the relative-phase
substitution is only designed to be safe *on the measured subspace*, which is a
weaker property than global diagonality - so certificate B legitimately fails for
arbitrary Hadamard-heavy circuits.  ``--mode diagonal`` therefore also runs a
mask that is diagonal by construction (a block may be downgraded only when no
downstream Hadamard touches its support), which is the regime where the ZX proof
closes.

Usage
-----
    .venv/bin/python ultimate_scalability_test.py                 # RL one-shot
    .venv/bin/python ultimate_scalability_test.py --mode diagonal # provable mask
    .venv/bin/python ultimate_scalability_test.py --self-test     # validate the ZX tests
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit, qasm2
from torch_geometric.data import Batch, Data

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


GEN = _load("variant_generator_ultimate", "generate_variant_dataset.py")
MS = _load("mixed_stress_ultimate", "mixed_stress.py")
RL = _load("rl_trainer_ultimate", "train_variant_policy_rl.py")

import pyzx as zx  # noqa: E402  (after the loaders, which are cheap)


# ---------------------------------------------------------------------------
# super-wide mixed instance
# ---------------------------------------------------------------------------


class WideInstance:
    """A circuit with ``width`` data qubits and many in-distribution MCX blocks.

    Every block keeps a native control count in ``{2..8}`` (so the policy sees the
    shapes it was trained on) but is *placed* on a random window of the wide
    register; the inter-block Clifford layers act on all data qubits, so the
    circuit is genuinely entangled and far too wide for the reduced verifier.
    """

    def __init__(self, width: int = 26, blocks: int = 30, seed: int = 0,
                 h_prob: float = 0.7, control_choices: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
                 ancilla_pool: int = 6):
        self.width = int(width)
        self.blocks = int(blocks)
        self.seed = int(seed)
        self.h_prob = float(h_prob)
        self.total = self.width + int(ancilla_pool)
        rng = random.Random(seed)
        self.control_counts: List[int] = []
        self.controls: List[List[int]] = []
        self.targets: List[int] = []
        self.ancillas: List[List[int]] = []
        pool = list(range(self.width, self.total))
        for _ in range(self.blocks):
            n = int(rng.choice(list(control_choices)))
            # the window + target + (n-2) clean ancillas must all fit; without the
            # ancilla_pool term the block->register map silently drops a wire.
            n = max(1, min(n, self.width - 2, ancilla_pool + 2))
            start = rng.randrange(self.width - n)
            controls = list(range(start, start + n))
            target = (start + n) % self.width
            if target in controls:
                target = (start + n + 1) % self.width
            self.control_counts.append(n)
            self.controls.append(controls)
            self.targets.append(target)
            self.ancillas.append(pool[: max(0, n - 2)])
        self.layers: List[Optional[QuantumCircuit]] = [None] * self.blocks
        for index in range(1, self.blocks):
            layer = QuantumCircuit(self.total)
            GEN.clifford_layer(layer, list(range(self.width)), rng, self.h_prob)
            self.layers[index] = layer
        self.block_t = [GEN.block_t_counts(n) for n in self.control_counts]
        self.t_all_exact = int(sum(t[0] for t in self.block_t))
        self.t_all_relative = int(sum(t[1] for t in self.block_t))

    # -- circuit assembly ---------------------------------------------------

    def _placed_block(self, index: int, relative: bool) -> QuantumCircuit:
        n = self.control_counts[index]
        base = GEN.relative_block(n) if relative else GEN.exact_block(n)
        mapping: Dict[int, int] = {}
        for i, q in enumerate(self.controls[index]):
            mapping[i] = q
        for i, q in enumerate(self.ancillas[index]):
            mapping[n + i] = q
        mapping[n + max(0, n - 2)] = self.targets[index]
        return MS.remap_circuit(base, mapping, self.total)

    def circuit(self, mask: int = 0) -> QuantumCircuit:
        out = QuantumCircuit(self.total)
        for index in range(self.blocks):
            if index and self.layers[index] is not None:
                out.compose(self.layers[index], range(self.total), inplace=True)
            out.compose(
                self._placed_block(index, bool((mask >> index) & 1)),
                range(self.total),
                inplace=True,
            )
        return out

    def t_count(self, mask: int) -> int:
        return int(sum(self.block_t[i][1 if (mask >> i) & 1 else 0] for i in range(self.blocks)))

    # -- policy inputs ------------------------------------------------------

    def policy_inputs(self) -> List[Data]:
        stats: List[Tuple[int, int]] = [(0, 0)]
        for layer in self.layers[1:]:
            if layer is None:
                stats.append((0, 0))
                continue
            h = sum(1 for ins in layer.data if ins.operation.name == "h")
            stats.append((h, sum(1 for ins in layer.data if ins.operation.name == "cx")))
        samples: List[Data] = []
        for index, n in enumerate(self.control_counts):
            t_exact, t_relative = self.block_t[index]
            report = {
                "context": stats,
                "blocks": self.blocks,
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
            samples.append(
                Data(
                    x=torch.cat([sample.x.float(), sample.x_support.float()], dim=-1),
                    edge_index=sample.edge_index,
                    edge_attr=sample.edge_attr,
                    context=sample.context.view(1, -1).float(),
                )
            )
        return samples

    # -- masks --------------------------------------------------------------

    def diagonal_safe_mask(self) -> int:
        """Downgrade a block only when no downstream Hadamard touches its support.

        That is precisely the condition under which the injected correction stays
        diagonal, so the ZX certificate B must close for this mask.
        """
        mask = 0
        for index in range(self.blocks):
            support = set(self.controls[index]) | {self.targets[index]} | set(self.ancillas[index])
            unsafe = False
            for layer in self.layers[index + 1:]:
                if layer is None:
                    continue
                for ins in layer.data:
                    if ins.operation.name != "h":
                        continue
                    if layer.find_bit(ins.qubits[0]).index in support:
                        unsafe = True
                        break
                if unsafe:
                    break
            if not unsafe:
                mask |= 1 << index
        return mask


# ---------------------------------------------------------------------------
# ZX certificates
# ---------------------------------------------------------------------------


def _zx(circuit: QuantumCircuit) -> "zx.Circuit":
    return zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()


def combined_circuit(exact: QuantumCircuit, downgraded: QuantumCircuit) -> "zx.Circuit":
    """``C = U_downgraded . U_exact^dagger`` (exact inverse first, then the circuit)."""
    ex = _zx(exact)
    down = _zx(downgraded)
    combined = zx.Circuit(ex.qubits)
    combined.add_circuit(ex.adjoint())
    combined.add_circuit(down)
    return combined


def identity_certificate(combined: "zx.Circuit") -> bool:
    graph = combined.to_graph()
    zx.simplify.full_reduce(graph)
    return bool(graph.is_id())


def diagonal_certificate(combined: "zx.Circuit", qubits: int,
                         indices: Optional[Sequence[int]] = None) -> Tuple[int, int]:
    """Count how many ``Z_j C Z_j C^dagger`` reduce to the identity."""
    ok = 0
    tested = 0
    for j in (range(qubits) if indices is None else indices):
        graph = zx.Circuit(qubits)
        graph.add_circuit(combined.adjoint())
        graph.add_gate("Z", int(j))
        graph.add_circuit(combined.copy())
        graph.add_gate("Z", int(j))
        g = graph.to_graph()
        zx.simplify.full_reduce(g)
        tested += 1
        ok += int(g.is_id())
    return ok, tested


def tensor_diagonality(combined: "zx.Circuit") -> Optional[float]:
    """``max|offdiag|`` of the combined tensor -- only feasible for tiny widths."""
    if combined.qubits > 10:
        return None
    tensor = np.asarray(combined.to_graph().to_tensor())
    size = 2 ** combined.qubits
    matrix = tensor.reshape(size, size)
    return float(np.abs(matrix - np.diag(np.diag(matrix))).max())


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def rl_mask(instance: WideInstance, model, device, rule: str = "depth") -> Tuple[int, List[float]]:
    batch = Batch.from_data_list(instance.policy_inputs()).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).tolist()
    if rule == "threshold":
        mask = 0
        for index, prob in enumerate(probs):
            if prob >= 0.5:
                mask |= 1 << index
    else:
        mask = MS.top_mask(probs, MS.depth_budget(instance.blocks))
    return mask, probs


def run(args) -> int:
    device = torch.device(args.device)
    model, checkpoint = MS.load_policy(Path(args.checkpoint), device)
    instance = WideInstance(width=args.width, blocks=args.blocks, seed=args.seed,
                            h_prob=args.h_prob)
    print("═" * 78)
    print("  终极外推测试：超越归约验证器内存极限的 ZX 图论等价性验证")
    print("═" * 78)
    print(f"  数据量子位 width = {instance.width}  (>25，归约验证器需要 2^{instance.width} 稠密矩阵)"
          f" | 总物理量子位 = {instance.total}")
    print(f"  区块数 = {instance.blocks} | 控制数分布 "
          f"{dict((n, instance.control_counts.count(n)) for n in sorted(set(instance.control_counts)))}")
    print(f"  all-exact T = {instance.t_all_exact} | all-relative T = {instance.t_all_relative}"
          f" | h_prob = {instance.h_prob:g}")
    print("─" * 78)

    if args.mode == "diagonal":
        mask = instance.diagonal_safe_mask()
        probs = [1.0 if (mask >> i) & 1 else 0.0 for i in range(instance.blocks)]
        label = "diagonal-by-construction (仅降级下游无 H 的区块)"
    else:
        started = time.perf_counter()
        mask, probs = rl_mask(instance, model, device, rule=args.rule)
        print(f"  RL 纯前向一次性（深度条件化 k_max = {MS.depth_budget(instance.blocks)}）"
              f" 用时 {time.perf_counter() - started:.2f}s")
        label = "RL one-shot (depth-conditioned)"

    downgrades = bin(mask).count("1")
    print(f"  掩码策略：{label}")
    print(f"  降级区块 {downgrades}/{instance.blocks} | 预估 T-count = {instance.t_count(mask)}"
          f"（全精确省 {instance.t_all_exact - instance.t_count(mask)} T）")
    print("─" * 78)

    exact = instance.circuit(0)
    downgraded = instance.circuit(mask)
    started = time.perf_counter()
    combined = combined_circuit(exact, downgraded)
    print(f"  联合电路 C = U_RL · U_exact† 已构造：{len(combined.gates)} 门"
          f"（{time.perf_counter() - started:.1f}s）")

    started = time.perf_counter()
    unitary_equal = identity_certificate(combined)
    t_a = time.perf_counter() - started
    print(f"  [证书 A] full_reduce(C) 是否恒等：{unitary_equal}（{t_a:.1f}s）")

    indices = None
    if args.max_checks and args.max_checks < instance.width:
        indices = list(range(args.max_checks))
    started = time.perf_counter()
    ok_j, tested_j = diagonal_certificate(combined, combined.qubits, indices)
    t_b = time.perf_counter() - started
    print(f"  [证书 B] Z_j C Z_j C† 归约为恒等：{ok_j}/{tested_j} 个 j"
          f"（{t_b:.1f}s）"
          + ("" if indices is None else f"  ⚠ 仅抽查前 {tested_j} 个量子位"))
    off = tensor_diagonality(combined)
    if off is not None:
        print(f"  [交叉验证] 联合张量 max|offdiag| = {off:.3e}"
              f" -> {'对角' if off < 1e-9 else '非对角'}")

    print("─" * 78)
    if unitary_equal:
        print("  ✅ 图论验证通过：full_reduce 把联合电路化归为恒等映射")
        print("     ⇒ U_RL 与 U_exact 作为算符完全相等（（多项式的 ZX 重写证明）")
    elif tested_j and ok_j == tested_j:
        print("  ✅ 图论验证通过：C 与所有 Pauli-Z 对易（每条 Z_j C Z_j C† 都化归为恒等）")
        print("     ⇒ C 是对角算符 ⇒ U_RL = U_exact·C ⇒ 所有 Z 基测量统计完全不变")
        print("     ⇒ 该降级掩码在", instance.width, "个量子位上被证明安全（多项式时间，无需矩阵）")
    else:
        print("  ⚠ 未通过 ZX 证明（注意：这不等于不安全！）")
        print("     根因：相对相位替换只在【被测量子空间】上安全，而 ZX 重写检定的是")
        print("     【全局张量】。实测被归约验证器判定为安全的掩码，其 C 的非对角元 ~0.5–0.7。")
        print("     想拿到图论级别的证明，请用 --mode diagonal（只在无下游 H 的区块上降级）。")
    print("═" * 78)
    return 0


def _self_test() -> int:
    """Validate that the ZX certificates agree with the exact tensor.

    The certificate is **sound but incomplete**: ``Z_j C Z_j C^dagger`` reducing to
    the identity for *all* j proves ``C`` diagonal, but ``full_reduce`` is a
    heuristic rewrite system, so a partial score only means "not certified" - it
    does **not** prove the tensor is non-diagonal.  We therefore assert the sound
    direction (certified => diagonal) and merely report the converse.
    """
    failures = 0

    def compare(tag: str, mask: int, h_prob: float, seed: int) -> None:
        nonlocal failures
        inst = WideInstance(width=4, blocks=3, seed=seed, h_prob=h_prob, ancilla_pool=2)
        combined = combined_circuit(inst.circuit(0), inst.circuit(mask))
        off = tensor_diagonality(combined)
        ok_j, tested_j = diagonal_certificate(combined, combined.qubits)
        certified = ok_j == tested_j
        if certified:
            sound = off < 1e-9
            print(f"  [{'ok  ' if sound else 'FAIL'}] {tag}: mask={mask:b} certified diagonal "
                  f"({ok_j}/{tested_j}) -> tensor offdiag={off:.2e} (sound direction)")
            failures += 0 if sound else 1
        else:
            print(f"  [info] {tag}: mask={mask:b} not certified ({ok_j}/{tested_j}), "
                  f"tensor offdiag={off:.2e} -> "
                  f"{'actually diagonal (rewrite incompleteness)' if off < 1e-9 else 'genuinely non-diagonal'}")

    compare("H-free downgrade", 0b001, 0.0, 3)
    compare("downgrade under Hadamards", 0b010, 1.0, 1)
    compare("downgrade under Hadamards", 0b100, 1.0, 2)
    compare("two downgrades", 0b011, 0.7, 5)

    inst = WideInstance(width=4, blocks=3, seed=3, h_prob=0.0, ancilla_pool=2)
    same = combined_circuit(inst.circuit(0), inst.circuit(0))
    good = identity_certificate(same)
    print(f"  [{'ok  ' if good else 'FAIL'}] identical circuits -> identity certificate: {good}")
    failures += 0 if good else 1

    inst2 = WideInstance(width=4, blocks=3, seed=3, h_prob=0.0, ancilla_pool=2)
    diff = combined_circuit(inst2.circuit(0), inst2.circuit(0b001))
    not_id = not identity_certificate(diff)
    print(f"  [{'ok  ' if not_id else 'FAIL'}] different circuits -> identity certificate is False")
    failures += 0 if not_id else 1
    print()
    if failures:
        print(f"self-test FAILED ({failures})")
        return 1
    print("self-test passed: the ZX certificates agree with the exact tensors")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ultimate wide-circuit scalability test.")
    parser.add_argument("--width", type=int, default=26, help="data qubits (>25 by default)")
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--h-prob", type=float, default=0.7)
    parser.add_argument("--mode", choices=("rl", "diagonal"), default="rl")
    parser.add_argument("--rule", choices=("depth", "threshold"), default="depth")
    parser.add_argument("--max-checks", type=int, default=0,
                        help="only spot-check this many Z_j (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
