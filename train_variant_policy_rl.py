#!/usr/bin/env python3
"""Strategy-B part (b): budget-constrained RL over the whole circuit.

Part (a) trained an independent per-block classifier: it scores each MCX block
in isolation and a verifier-in-the-loop greedy pass turns the scores into a
safe mask.  Part (b) asks the harder question -- *how should the choices be
made jointly, for a whole circuit, under a resource budget?*

The environment is one "big circuit": ``k`` MCX blocks (same control count for
now) separated by random Clifford layers, measured in the Z basis.  The agent
emits one Bernoulli action per block (0 = keep the exact Clifford+T ladder,
1 = downgrade to the cheaper relative-phase RCCX ladder).  The episode ends with

    t_policy = (k - nd) * t_exact + nd * t_relative          (nd = downgrades)
    usable   = the *whole-circuit* Z-basis statistics are unchanged (verifier)
    t_final  = t_policy if usable else k * t_exact           (hard safety net)
    reward   = (k*t_exact - t_final) / (k*t_exact)
               - risk_penalty  * (0 if usable else 1)
               - budget_penalty * max(0, nd - budget)

so an unsafe proposal earns nothing (the circuit falls back to exact) and is
additionally punished.  Training is REINFORCE with a batch-mean baseline and an
entropy bonus.  Crucially the verifier is *one simulation per episode*, not an
enumeration, so the same trainer scales to circuits far too large to enumerate.

Usage
-----
    .venv/bin/python train_variant_policy_rl.py --epochs 40 --max-blocks 6
    .venv/bin/python train_variant_policy_rl.py --eval-only
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


GEN = _load("variant_generator_rl", "generate_variant_dataset.py")
TRAINER = _load("variant_policy_trainer_rl", "train_variant_policy.py")
SYM = _load("variant_symbolic_rl", "symbolic_verifier.py")


# ---------------------------------------------------------------------------
# instance pool: blocks (policy input) + a whole-circuit reward oracle
# ---------------------------------------------------------------------------


def _policy_data(sample: Data) -> Data:
    """The exact fields the policy consumes (no metadata, no batch pollution)."""
    return Data(
        x=torch.cat([sample.x.float(), sample.x_support.float()], dim=-1),
        edge_index=sample.edge_index,
        edge_attr=sample.edge_attr,
        context=sample.context.view(1, -1).float(),
    )


_BLOCK_GRAPH_CACHE: Dict[int, Any] = {}
_TCOUNT_CACHE: Dict[int, Tuple[int, int]] = {}


def _block_graph(num_controls: int):
    if num_controls not in _BLOCK_GRAPH_CACHE:
        _BLOCK_GRAPH_CACHE[num_controls] = GEN.block_graph_features(GEN.exact_block(num_controls))
    return _BLOCK_GRAPH_CACHE[num_controls]


def _t_counts(num_controls: int) -> Tuple[int, int]:
    if num_controls not in _TCOUNT_CACHE:
        _TCOUNT_CACHE[num_controls] = GEN.block_t_counts(num_controls)
    return _TCOUNT_CACHE[num_controls]


class Instance:
    """One episode: the block inputs plus everything needed to score a mask.

    Built *without* the enumerating oracle -- the reward only needs one
    simulation per sampled mask, so the same code scales past ``2**k``.
    """

    __slots__ = ("nc", "blocks", "seed", "h_prob", "tol", "t_exact", "t_relative",
                 "t_all_exact", "t_all_relative", "headroom", "symbolic", "blocks_data")

    def __init__(self, num_controls: int, blocks: int, seed: int, h_prob: float, config):
        circuit, context = GEN.build_instance(num_controls, blocks, 0, seed, h_prob)
        t_exact, t_relative = _t_counts(num_controls)
        report = {
            "context": context,
            "blocks": blocks,
            "num_controls": num_controls,
            "t_per_block": {"exact": t_exact, "relative": t_relative},
        }
        graph = _block_graph(num_controls)
        self.blocks_data = []
        for index in range(blocks):
            sample = Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                x_support=graph.x_support,
            )
            sample.context = GEN.context_features(report, index)
            self.blocks_data.append(_policy_data(sample))
        self.symbolic = SYM.ReducedSymbolicVerifier(
            num_controls, blocks, seed, h_prob, tolerance=config.tolerance
        )
        self.nc = int(num_controls)
        self.blocks = int(blocks)
        self.seed = int(seed)
        self.h_prob = float(h_prob)
        self.tol = float(config.tolerance)
        self.t_exact = int(t_exact)
        self.t_relative = int(t_relative)
        self.t_all_exact = int(self.blocks * self.t_exact)
        self.t_all_relative = int(self.blocks * self.t_relative)
        # largest saving any mask could legally reach (what all-relative would give)
        self.headroom = int(self.t_all_exact - self.t_all_relative)

    def usable(self, mask: int) -> bool:
        return self.symbolic.is_safe(mask)

    def t_policy(self, mask: int) -> int:
        downgraded = bin(mask).count("1")
        return (self.blocks - downgraded) * self.t_exact + downgraded * self.t_relative


def make_pool(specs, config) -> List[Instance]:
    return [Instance(nc, blocks, seed, config.h_prob, config) for nc, blocks, seed in specs]


def reward(instance: Instance, mask: int, risk_penalty: float, budget: Optional[int],
           budget_penalty: float) -> Tuple[float, bool, int]:
    downgraded = bin(mask).count("1")
    t_policy = instance.t_policy(mask)
    usable = instance.usable(mask)
    t_final = t_policy if usable else instance.t_all_exact
    # Normalise by the *legal headroom* (the saving an all-relative circuit would
    # give), not by the absolute T-count.  Otherwise a 40-block circuit's best
    # possible saving is ~0.3% of its T-count while a 4-block circuit's is ~15%,
    # so the shallow instances dominate the policy gradient and the deep-narrow
    # curriculum can only ever teach "never downgrade anything".
    value = (instance.t_all_exact - t_final) / max(instance.headroom, 1)
    value -= risk_penalty * (0.0 if usable else 1.0)
    if budget is not None and downgraded > budget:
        value -= budget_penalty * (downgraded - budget)
    return float(value), usable, t_policy


# ---------------------------------------------------------------------------
# batched action sampling
# ---------------------------------------------------------------------------


def instance_batch(instances: List[Instance], device) -> Tuple[Batch, List[int]]:
    data: List[Data] = []
    counts: List[int] = []
    for instance in instances:
        data.extend(instance.blocks_data)
        counts.append(instance.blocks)
    return Batch.from_data_list(data).to(device), counts


def sample_masks(model, instances: List[Instance], device, risk_penalty, budget,
                 budget_penalty, greedy: bool = False):
    batch, counts = instance_batch(instances, device)
    logits = model(batch)
    chunks = torch.split(logits, counts)
    rewards: List[float] = []
    logps: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    infos: List[Dict[str, Any]] = []
    for instance, chunk in zip(instances, chunks):
        if greedy:
            action = (torch.sigmoid(chunk) >= 0.5).float()
            logp = torch.tensor(0.0, device=device)
            entropy = torch.tensor(0.0, device=device)
        else:
            dist = torch.distributions.Bernoulli(logits=chunk)
            action = dist.sample()
            logp = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()
        mask = 0
        for index, bit in enumerate(action.tolist()):
            if bit >= 0.5:
                mask |= 1 << index
        value, usable, t_policy = reward(instance, mask, risk_penalty, budget, budget_penalty)
        rewards.append(value)
        logps.append(logp)
        entropies.append(entropy)
        infos.append({"mask": mask, "usable": usable, "t_policy": t_policy,
                      "t_all_exact": instance.t_all_exact, "downgraded": bin(mask).count("1")})
    return rewards, logps, entropies, infos


def greedy_verified_mask(instance: Instance, probs: List[float]) -> int:
    """Verifier-in-the-loop greedy using the model's per-block confidence order."""
    mask = 0
    for block in sorted(range(instance.blocks), key=lambda b: -probs[b]):
        candidate = mask | (1 << block)
        if instance.usable(candidate):
            mask = candidate
    return mask


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def build(args) -> int:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = GEN.VariantConfig(h_prob=args.h_prob)

    ss = np.random.SeedSequence(args.seed)
    seeds = ss.generate_state(args.instances + args.eval_instances, dtype=np.uint32).tolist()
    rng = np.random.default_rng(args.seed)
    control_values = [int(v) for v in args.control_values.split(",") if v.strip()]

    deep_controls = [int(v) for v in args.deep_controls.split(",") if v.strip()]

    def specs(count: int, offset: int, deep_share: float = 0.0) -> List[Tuple[int, int, int]]:
        """Curriculum pool: ``deep_share`` of the instances are deep-but-narrow.

        Deep and narrow means few data qubits (``n_c = 2`` -> 3 data qubits, so the
        verifier is microseconds) but many blocks (30..50).  There the agent gets to
        feel how *consecutive* downgrades destroy coherence, which is exactly the
        lesson an one-shot policy cannot learn from shallow 2..7 block circuits.
        """
        out = []
        n_deep = int(round(count * deep_share))
        for i in range(count):
            if i < n_deep:
                nc = int(rng.choice(deep_controls))
                blocks = int(rng.integers(args.deep_min_blocks, args.deep_max_blocks + 1))
            else:
                nc = int(rng.choice(control_values))
                blocks = int(rng.integers(args.min_blocks, args.max_blocks + 1))
            out.append((nc, blocks, int(seeds[offset + i])))
        return out

    def held_out_specs(count: int, salt: int, deep_share: float = 0.0) -> List[Tuple[int, int, int]]:
        """Held-out circuits from an *independent* RNG.

        The evaluation pools must be byte-identical between runs that differ only in
        ``--deep-fraction``, otherwise the comparison is meaningless.  Using a
        separate ``default_rng(seed + salt)`` decouples them from the training RNG
        (whose stream position shifts when the curriculum changes draw sizes).
        """
        local = np.random.default_rng(args.seed + salt)
        local_seeds = np.random.SeedSequence(args.seed + salt).generate_state(count, dtype=np.uint32)
        out = []
        n_deep = int(round(count * deep_share))
        for i in range(count):
            if i < n_deep:
                nc = int(local.choice(deep_controls))
                blocks = int(local.integers(args.deep_min_blocks, args.deep_max_blocks + 1))
            else:
                nc = int(local.choice(control_values))
                blocks = int(local.integers(args.min_blocks, args.max_blocks + 1))
            out.append((nc, blocks, int(local_seeds[i])))
        return out

    print(
        f"  building {args.instances} train / {args.eval_instances} eval instances "
        f"(blocks {args.min_blocks}..{args.max_blocks}, controls {control_values}) "
        f"| curriculum: {args.deep_fraction:.0%} deep-narrow "
        f"(controls {deep_controls}, blocks {args.deep_min_blocks}..{args.deep_max_blocks})",
        flush=True,
    )
    train_pool = make_pool(specs(args.instances, 0, args.deep_fraction), config)
    eval_pool = make_pool(held_out_specs(args.eval_instances, 123), config)
    deep_pool = make_pool(held_out_specs(args.deep_eval_instances, 999, deep_share=1.0), config)

    sample = train_pool[0].blocks_data[0]
    node_dim = int(sample.x.size(1))
    context_dim = int(sample.context.numel())
    print(f"  dims: node_dim={node_dim} context_dim={context_dim}", flush=True)
    model = TRAINER.BlockVariantPolicy(
        node_dim=node_dim, context_dim=context_dim, hidden=args.hidden
    ).to(device)
    init = Path(args.init_checkpoint)
    if init.exists():
        checkpoint = torch.load(init, map_location="cpu", weights_only=False)
        if int(checkpoint.get("node_dim", node_dim)) != node_dim or int(
            checkpoint.get("context_dim", context_dim)
        ) != context_dim:
            print(
                f"  warm-start skipped: checkpoint dims "
                f"({checkpoint.get('node_dim')}, {checkpoint.get('context_dim')}) != "
                f"({node_dim}, {context_dim})",
                flush=True,
            )
        else:
            model.load_state_dict(checkpoint["model_state"])
            print(f"  warm-start from {init.name} (epoch {checkpoint.get('epoch')})", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "variant_policy_rl_best.pt"
    running = None
    best_eval = -1e9
    started = time.time()
    order = np.arange(len(train_pool))

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(order)
        epoch_rewards: List[float] = []
        epoch_safe: List[float] = []
        for start in range(0, len(train_pool), args.batch_size):
            chunk_ids = order[start:start + args.batch_size]
            instances = [train_pool[i] for i in chunk_ids]
            rewards, logps, entropies, infos = sample_masks(
                model, instances, device, args.risk_penalty, args.budget, args.budget_penalty
            )
            rewards_t = torch.tensor(rewards, dtype=torch.float, device=device)
            baseline = rewards_t.mean()
            advantage = rewards_t - baseline
            logp_t = torch.stack(logps)
            entropy_t = torch.stack(entropies)
            loss = -(logp_t * advantage.detach()).mean() - args.entropy * entropy_t.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_rewards.extend(rewards)
            epoch_safe.extend(1.0 if info["usable"] else 0.0 for info in infos)

        mean_reward = float(np.mean(epoch_rewards))
        running = mean_reward if running is None else 0.9 * running + 0.1 * mean_reward
        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} | reward={mean_reward:+.4f} (running {running:+.4f}) "
                f"| sampled masks safe={np.mean(epoch_safe):.1%}",
                flush=True,
            )
        if running > best_eval:
            best_eval = running
            torch.save({"model_state": model.state_dict(), "node_dim": node_dim,
                        "context_dim": context_dim, "hidden": args.hidden,
                        "epoch": epoch, "running_reward": running}, best_path)

    print(f"  trained in {time.time() - started:.1f}s | best running reward {best_eval:+.4f}", flush=True)
    model.load_state_dict(torch.load(best_path, weights_only=False)["model_state"])
    evaluate(model, eval_pool, args, device, compare_checkpoint=args.init_checkpoint)
    evaluate_deep(model, deep_pool, device, threshold=args.threshold)
    return 0


def _repair(verifier, probs, mask):
    """Drop the least-confident downgrades until the mask is safe (never unsafe)."""
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


@torch.no_grad()
def evaluate_deep(model, pool: List[Instance], device, threshold: float = 0.5) -> Dict[str, Any]:
    """Evaluate on deep circuits, where the oracle (2**blocks) is out of reach.

    Reports the two deployed strategies plus the metric the curriculum is supposed
    to move: how often the *raw* one-shot threshold mask is already safe.
    """
    model.eval()
    exact: List[float] = []
    greedy_saving: List[float] = []
    one_shot_saving: List[float] = []
    raw_safe: List[float] = []
    drops: List[float] = []
    greedy_kept: List[float] = []
    repaired_kept: List[float] = []
    for instance in pool:
        batch, _ = instance_batch([instance], device)
        probs = torch.sigmoid(model(batch)).tolist()
        base = instance.t_all_exact
        greedy_mask = instance.symbolic.greedy(probs)
        raw_mask = instance.symbolic.mask_from_probs(probs, threshold)
        safe = instance.symbolic.is_safe(raw_mask)
        repaired_mask, dropped = _repair(instance.symbolic, probs, raw_mask)
        exact.append(base)
        greedy_saving.append(base - instance.t_policy(greedy_mask))
        one_shot_saving.append(base - instance.t_policy(repaired_mask))
        raw_safe.append(1.0 if safe else 0.0)
        drops.append(dropped)
        greedy_kept.append(bin(greedy_mask).count("1"))
        repaired_kept.append(bin(repaired_mask).count("1"))
    result = {
        "instances": len(pool),
        "mean_blocks": float(np.mean([i.blocks for i in pool])),
        "mean_all_exact_t": float(np.mean(exact)),
        "greedy_saving": float(np.mean(greedy_saving)),
        "one_shot_saving": float(np.mean(one_shot_saving)),
        "one_shot_raw_safe_rate": float(np.mean(raw_safe)),
        "mean_one_shot_drops": float(np.mean(drops)),
        "mean_greedy_downgrades": float(np.mean(greedy_kept)),
        "mean_repaired_downgrades": float(np.mean(repaired_kept)),
    }
    print("  [deep-eval] held-out deep-narrow circuits (no oracle available):", flush=True)
    print(
        f"    instances={result['instances']} (avg {result['mean_blocks']:.0f} blocks) | "
        f"all-exact={result['mean_all_exact_t']:.1f} T",
        flush=True,
    )
    print(
        f"    greedy+safe     saves {result['greedy_saving']:.1f} T "
        f"({result['mean_greedy_downgrades']:.1f} blocks)",
        flush=True,
    )
    print(
        f"    one-shot+repair saves {result['one_shot_saving']:.1f} T "
        f"({result['mean_repaired_downgrades']:.1f} blocks, dropped "
        f"{result['mean_one_shot_drops']:.1f}) | raw-threshold safe rate "
        f"{result['one_shot_raw_safe_rate']:.0%}",
        flush=True,
    )
    return result


@torch.no_grad()
def evaluate(model, pool: List[Instance], args, device, compare_checkpoint=None) -> Dict[str, Any]:
    model.eval()
    # deterministic mask (threshold 0.5) and greedy+verifier decode
    rewards, _, _, infos = sample_masks(
        model, pool, device, args.risk_penalty, args.budget, args.budget_penalty, greedy=True
    )
    exact = np.mean([it.t_all_exact for it in pool])
    policy = np.mean([info["t_policy"] for info in infos])
    safe = np.mean([1.0 if info["usable"] else 0.0 for info in infos])
    fallback = np.mean([
        info["t_policy"] if info["usable"] else info["t_all_exact"] for info in infos
    ])

    def probs_of(instance: Instance) -> List[float]:
        batch, _ = instance_batch([instance], device)
        return torch.sigmoid(model(batch)).tolist()

    greedy_t: List[float] = []
    oracle_t: List[float] = []
    for instance in pool:
        mask = greedy_verified_mask(instance, probs_of(instance))
        greedy_t.append(instance.t_policy(mask))
        # shared partial-product sweep: one matrix product per mask
        deviations = instance.symbolic.deviation_table()
        tolerance = instance.symbolic.tolerance
        best = instance.t_all_exact
        for candidate in range(1 << instance.blocks):
            if instance.t_policy(candidate) < best and deviations[candidate] <= tolerance:
                best = instance.t_policy(candidate)
        oracle_t.append(best)

    greedy = float(np.mean(greedy_t))
    oracle = float(np.mean(oracle_t))
    headroom = exact - oracle
    result = {
        "instances": len(pool),
        "mean_all_exact": float(exact),
        "mean_oracle": oracle,
        "mean_policy_threshold": float(policy),
        "mean_policy_threshold_fallback": float(fallback),
        "mean_policy_greedy": greedy,
        "policy_threshold_safe_rate": float(safe),
        "oracle_saving": float(headroom),
        "policy_greedy_saving": float(exact - greedy),
        "headroom_captured_greedy": float((exact - greedy) / headroom) if headroom else float("nan"),
        "mean_reward": float(np.mean(rewards)),
    }
    print("  [rl-eval] whole-circuit budget-constrained policy:", flush=True)
    print(
        f"    instances={result['instances']} | all-exact={exact:.2f} | oracle={oracle:.2f} "
        f"(saves {headroom:.2f}) | threshold-raw={policy:.2f} (safe {safe:.0%}) | "
        f"one-shot threshold+fallback={fallback:.2f} | greedy+verifier={greedy:.2f} "
        f"(saves {result['policy_greedy_saving']:.2f}, "
        f"{result['headroom_captured_greedy']:.0%} of headroom)",
        flush=True,
    )
    if compare_checkpoint and Path(compare_checkpoint).exists():
        checkpoint = torch.load(compare_checkpoint, map_location="cpu", weights_only=False)
        other = TRAINER.BlockVariantPolicy(
            node_dim=int(checkpoint.get("node_dim", 27)),
            context_dim=int(checkpoint.get("context_dim", GEN.CONTEXT_DIM)),
            hidden=int(checkpoint.get("hidden", args.hidden)),
        ).to(device)
        other.load_state_dict(checkpoint["model_state"])
        other.eval()
        other_greedy: List[float] = []
        other_threshold: List[float] = []
        other_safe: List[float] = []
        for instance in pool:
            batch, _ = instance_batch([instance], device)
            probs = torch.sigmoid(other(batch)).tolist()
            other_greedy.append(instance.t_policy(greedy_verified_mask(instance, probs)))
            mask = 0
            for block, prob in enumerate(probs):
                if prob >= 0.5:
                    mask |= 1 << block
            ok = instance.usable(mask)
            other_safe.append(1.0 if ok else 0.0)
            other_threshold.append(instance.t_policy(mask) if ok else instance.t_all_exact)
        compare = float(np.mean(other_greedy))
        result["compare_checkpoint"] = str(compare_checkpoint)
        result["compare_greedy"] = compare
        result["compare_headroom_captured"] = (
            float((exact - compare) / headroom) if headroom else float("nan")
        )
        result["compare_threshold_fallback"] = float(np.mean(other_threshold))
        result["compare_threshold_safe"] = float(np.mean(other_safe))
        print(
            f"    [compare] {Path(compare_checkpoint).name} (supervised): "
            f"greedy+verifier={compare:.2f} (saves {exact - compare:.2f}, "
            f"{result['compare_headroom_captured']:.0%} of headroom) | "
            f"one-shot threshold+fallback={result['compare_threshold_fallback']:.2f} "
            f"(safe {result['compare_threshold_safe']:.0%})",
            flush=True,
        )

    out = Path(args.out_dir) / "variant_policy_rl_report.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def eval_only(args) -> int:
    device = torch.device(args.device)
    config = GEN.VariantConfig(h_prob=args.h_prob)
    ss = np.random.SeedSequence(args.seed)
    seeds = ss.generate_state(args.eval_instances, dtype=np.uint32).tolist()
    rng = np.random.default_rng(args.seed)
    control_values = [int(v) for v in args.control_values.split(",") if v.strip()]
    specs = [
        (int(rng.choice(control_values)), int(rng.integers(args.min_blocks, args.max_blocks + 1)), int(seeds[i]))
        for i in range(args.eval_instances)
    ]
    pool = make_pool(specs, config)
    checkpoint = torch.load(args.rl_checkpoint, map_location="cpu", weights_only=False)
    model = TRAINER.BlockVariantPolicy(
        node_dim=int(checkpoint.get("node_dim", 27)),
        context_dim=int(checkpoint.get("context_dim", GEN.CONTEXT_DIM)),
        hidden=int(checkpoint.get("hidden", args.hidden)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    evaluate(model, pool, args, device, compare_checkpoint=args.init_checkpoint)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Strategy-B (b): budget-constrained RL.")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--eval-instances", type=int, default=40)
    parser.add_argument("--min-blocks", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=6)
    parser.add_argument("--control-values", default="2,3,4")
    parser.add_argument("--deep-fraction", type=float, default=0.0,
                        help="fraction of the training pool made of deep-narrow instances "
                             "(off by default: measured to make the raw threshold mask safe "
                             "on deep circuits, but it collapses the downgrade rate and costs "
                             "savings on shallow circuits too -- see the deep evaluation)")
    parser.add_argument("--deep-min-blocks", type=int, default=30)
    parser.add_argument("--deep-max-blocks", type=int, default=50)
    parser.add_argument("--deep-controls", default="2",
                        help="control counts used for the deep-narrow curriculum subset")
    parser.add_argument("--deep-eval-instances", type=int, default=8,
                        help="held-out deep circuits used only for evaluation")
    parser.add_argument("--h-prob", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--risk-penalty", type=float, default=0.5)
    parser.add_argument("--budget", type=int, default=None,
                        help="max downgrades per circuit before the budget penalty bites")
    parser.add_argument("--budget-penalty", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="one-shot threshold used by the deep evaluation")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--init-checkpoint", default=str(ROOT / "checkpoints" / "variant_policy_best.pt"))
    parser.add_argument("--rl-checkpoint", default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_only:
        return eval_only(args)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
