#!/usr/bin/env python3
"""Train the strategy-B per-block variant policy.

The policy is a small graph network: it reads a single MCX block's simplified
ZX diagram (the ``x``/``edge_index``/``edge_attr``/``x_support`` fields the
generator wrote) plus the 10-dimensional ``context`` of the surrounding
instance, and emits one logit per block:

    logit > 0  ->  downgrade this block to the relative-phase RCCX ladder
    logit <= 0 ->  keep the exact Clifford+T ladder

Training is supervised binary cross-entropy against the oracle label
``y_variant``.  The evaluation replays the policy's per-block decisions on the
whole instance and looks the resulting mask up in the generator's own
``mask_table`` (stored in ``instances.json``), which gives the three numbers
that actually matter:

    all-exact : T-count if nothing is downgraded (the safe baseline)
    oracle    : T-count of the best usable assignment (the ceiling)
    policy    : T-count actually realised, with a verifier fallback -- any
                instance whose predicted mask is not usable (it would change
                the Z-basis measurement statistics) is repaired or dropped
                back to all-exact.

Usage
-----
    .venv/bin/python train_variant_policy.py
    .venv/bin/python train_variant_policy.py --epochs 120 --lr 1e-3 --hidden 64
    .venv/bin/python train_variant_policy.py --eval-only --checkpoint checkpoints/variant_policy_best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool

ROOT = Path(__file__).resolve().parent
NODE_BASE = 11          # generator's per-node features
SUPPORT_WIDTH = 16      # per-qubit boundary-support mask (Linux-MCX MAX_QUBITS)
NODE_DIM = NODE_BASE + SUPPORT_WIDTH   # 27
CONTEXT_DIM = 19   # generator's corrected instance-aware context layout


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class BlockVariantPolicy(nn.Module):
    """Graph embedding of one MCX block + context -> single downgrade logit."""

    def __init__(self, node_dim: int = NODE_DIM, context_dim: int = CONTEXT_DIM,
                 hidden: int = 64, edge_dim: int = 2):
        super().__init__()
        self.node_dim = int(node_dim)
        self.context_dim = int(context_dim)
        self.hidden = int(hidden)
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(self.node_dim, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, self.hidden),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GCNConv(self.hidden, self.hidden)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        h = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        h = F.relu(self.conv2(h, data.edge_index))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch)
        context = data.context.view(-1, self.context_dim)
        context = self.context_mlp(context)
        logit = self.head(torch.cat([pooled, context], dim=-1)).squeeze(-1)
        return logit


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def widen(sample: Data) -> Data:
    """Concatenate the block's support mask into ``x`` and make context 2-D."""
    sample.x = torch.cat([sample.x.float(), sample.x_support.float()], dim=-1)
    sample.context = sample.context.view(1, -1).float()
    return sample


def load_split(path: Path) -> List[Data]:
    samples = torch.load(path, weights_only=False)
    return [widen(sample) for sample in samples]


def load_reports(path: Path) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        reports = json.load(handle)
    return {int(report["instance_id"]): report for report in reports}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _block_metrics(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    prob = 1.0 / (1.0 + np.exp(-logits))
    acc = float(((prob > 0.5).astype(np.int64) == labels).mean())
    order = np.argsort(prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(prob) + 1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos and n_neg:
        auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = float("nan")
    return acc, auc


def run_epoch(model, loader, optimizer, pos_weight, device):
    model.train()
    total = 0.0
    n = 0
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logit = model(batch)
        label = batch.y_variant.float()
        loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
        logits_all.append(logit.detach().cpu().numpy())
        labels_all.append(label.detach().cpu().numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all).astype(np.int64)
    acc, auc = _block_metrics(logits, labels)
    return total / max(n, 1), acc, auc


@torch.no_grad()
def predict_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits_all.append(model(batch).detach().cpu().numpy())
        labels_all.append(batch.y_variant.detach().cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all).astype(np.int64)


@torch.no_grad()
def predict_per_instance(model, loader, device) -> Dict[int, Dict[int, float]]:
    """Map instance_id -> {block_index: probability of downgrade}."""
    model.eval()
    out: Dict[int, Dict[int, float]] = defaultdict(dict)
    for batch in loader:
        batch = batch.to(device)
        prob = torch.sigmoid(model(batch)).detach().cpu().numpy()
        inst = batch.y_instance.detach().cpu().numpy()
        block = batch.y_block.detach().cpu().numpy()
        for i in range(len(inst)):
            out[int(inst[i])][int(block[i])] = float(prob[i])
    return out


# ---------------------------------------------------------------------------
# oracle-based instance evaluation
# ---------------------------------------------------------------------------


def _instance_outcome(mask: int, rows: List[Dict[str, Any]]) -> Tuple[int, bool]:
    row = rows[mask]
    return int(row["t_count"]), bool(row["usable"])


def _repair(mask: int, probs: Dict[int, float], rows: List[Dict[str, Any]],
            blocks: int) -> Tuple[int, bool]:
    """Drop the least-confident downgrades until the mask is usable."""
    if rows[mask]["usable"]:
        return mask, True
    downgraded = sorted((b for b in range(blocks) if (mask >> b) & 1),
                        key=lambda b: probs.get(b, 0.5))
    for block in downgraded:
        mask &= ~(1 << block)
        if rows[mask]["usable"]:
            return mask, True
    return 0, True  # all-exact is usable by construction


def measure_instances(
    per_instance: Dict[int, Dict[int, float]],
    reports: Dict[int, Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Replay the policy on full instances and compare against the oracle."""
    n = 0
    sum_all_exact = sum_oracle = sum_policy = sum_repaired = sum_greedy = 0.0
    sum_all_relative = 0.0
    policy_usable = 0
    oracle_touched = captured = 0.0
    all_relative_usable = 0
    illegal_saving = 0.0
    per_block_hit = per_block_total = 0
    block_logits: List[float] = []
    block_labels: List[int] = []

    for instance_id, probs in per_instance.items():
        report = reports.get(instance_id)
        if report is None or not report.get("mask_table"):
            continue
        rows = report["mask_table"]
        blocks = int(report["blocks"])
        mask = 0
        for block in range(blocks):
            p = probs.get(block, 0.0)
            block_logits.append(float(np.log(max(p, 1e-6) / max(1 - p, 1e-6))))
            oracle_bit = 1 if (int(report["oracle_mask"]) >> block) & 1 else 0
            block_labels.append(oracle_bit)
            per_block_total += 1
            per_block_hit += int((p >= threshold) == bool(oracle_bit))
            if p >= threshold:
                mask |= 1 << block

        t_all_exact = int(rows[0]["t_count"])
        t_all_relative = int(rows[-1]["t_count"])
        t_oracle = int(report["oracle_t_count"])
        t_policy, ok = _instance_outcome(mask, rows)
        repaired_mask, _ = _repair(mask, probs, rows, blocks)
        t_repaired = int(rows[repaired_mask]["t_count"])
        # verifier-in-the-loop greedy decode: add downgrades in order of the
        # model's confidence, keeping only those the verifier still accepts.
        greedy_mask = 0
        for block in sorted(range(blocks), key=lambda b: -probs.get(b, 0.0)):
            candidate = greedy_mask | (1 << block)
            if rows[candidate]["usable"]:
                greedy_mask = candidate
        t_greedy = int(rows[greedy_mask]["t_count"])

        n += 1
        sum_all_exact += t_all_exact
        sum_all_relative += t_all_relative
        sum_oracle += t_oracle
        sum_policy += t_policy if ok else t_all_exact   # verifier fallback
        sum_repaired += t_repaired
        sum_greedy += t_greedy
        policy_usable += int(ok)
        all_relative_usable += int(bool(report["all_relative_usable"]))
        illegal_saving += (t_all_exact - t_policy) if not ok else 0
        oracle_touched += (t_all_exact - t_oracle)
        captured += (t_all_exact - (t_policy if ok else t_all_exact))

    denom = max(n, 1)
    acc, auc = _block_metrics(np.array(block_logits), np.array(block_labels))
    return {
        "instances": n,
        "blocks": per_block_total,
        "block_acc": acc,
        "block_auc": auc,
        "mean_all_exact": sum_all_exact / denom,
        "mean_oracle": sum_oracle / denom,
        "mean_policy": sum_policy / denom,
        "mean_policy_repaired": sum_repaired / denom,
        "mean_all_relative": sum_all_relative / denom,
        "oracle_saving": oracle_touched / denom,
        "policy_saving": captured / denom,
        "policy_repaired_saving": (sum_all_exact - sum_repaired) / denom,
        "mean_policy_greedy": sum_greedy / denom,
        "policy_greedy_saving": (sum_all_exact - sum_greedy) / denom,
        "headroom_captured_greedy": ((sum_all_exact - sum_greedy) / oracle_touched)
        if oracle_touched else float("nan"),
        "policy_usable_rate": policy_usable / denom,
        "all_relative_usable_rate": all_relative_usable / denom,
        "illegal_saving_per_instance": illegal_saving / denom,
        "headroom_captured": (captured / oracle_touched) if oracle_touched else float("nan"),
    }


def format_report(result: Dict[str, Any]) -> str:
    lines = [
        f"  instances={result['instances']}  blocks={result['blocks']}  "
        f"block-acc={result['block_acc']:.3f}  block-auc={result['block_auc']:.3f}",
        f"  T-count per instance: all-exact={result['mean_all_exact']:.2f} | "
        f"oracle={result['mean_oracle']:.2f} (saves {result['oracle_saving']:.2f}) | "
        f"policy={result['mean_policy']:.2f} (saves {result['policy_saving']:.2f}) | "
        f"policy+repair={result['mean_policy_repaired']:.2f} "
        f"(saves {result['policy_repaired_saving']:.2f})",
        f"  all-relative(illegal)={result['mean_all_relative']:.2f} | "
        f"policy mask usable={result['policy_usable_rate']:.1%} | "
        f"all-relative usable={result['all_relative_usable_rate']:.1%}",
        f"  headroom captured={result['headroom_captured']:.1%} | "
        f"T a naive (unverified) policy would wrongly claim per instance="
        f"{result['illegal_saving_per_instance']:.2f}",

        f"  verifier-greedy: T={result['mean_policy_greedy']:.2f} (saves {result['policy_greedy_saving']:.2f}) | "
        f"headroom captured={result['headroom_captured_greedy']:.1%}",    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build(args) -> int:
    device = torch.device(args.device)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    base = Path(args.data)
    train = load_split(base / "train.pt")
    val = load_split(base / "val.pt")
    test = load_split(base / "test.pt")
    reports = load_reports(base / "instances.json")
    print(f"  data: train={len(train)} val={len(val)} test={len(test)} block samples "
          f"(instances={len(reports)})", flush=True)

    labels = torch.tensor([float(s.y_variant) for s in train])
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1e-3)], device=device)
    print(f"  train downgrade rate={n_pos / len(labels):.3f}  pos_weight={float(pos_weight):.3f}",
          flush=True)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    node_dim = int(train[0].x.size(1))
    context_dim = int(train[0].context.numel())
    print(f"  dims: node_dim={node_dim} context_dim={context_dim}", flush=True)
    model = BlockVariantPolicy(
        node_dim=node_dim, context_dim=context_dim, hidden=args.hidden
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "variant_policy_best.pt"
    last_path = out_dir / "variant_policy_last.pt"

    best_val = float("inf")
    best_epoch = 0
    patience = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_auc = run_epoch(
            model, train_loader, optimizer, pos_weight, device
        )
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logit = model(batch)
                loss = F.binary_cross_entropy_with_logits(
                    logit, batch.y_variant.float(), pos_weight=pos_weight
                )
                val_loss += float(loss) * batch.num_graphs
                vn += batch.num_graphs
        val_loss /= max(vn, 1)
        v_logits, v_labels = predict_logits(model, val_loader, device)
        val_acc, val_auc = _block_metrics(v_logits, v_labels)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} | train loss={train_loss:.4f} acc={train_acc:.3f} "
                f"auc={train_auc:.3f} | val loss={val_loss:.4f} acc={val_acc:.3f} "
                f"auc={val_auc:.3f}",
                flush=True,
            )

        state = {
            "model_state": model.state_dict(),
            "node_dim": node_dim,
            "context_dim": context_dim,
            "hidden": args.hidden,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_auc": val_auc,
        }
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(state, best_path)
        else:
            patience += 1
        torch.save({**state, "epoch": epoch}, last_path)
        if args.patience and patience >= args.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    print(f"  trained in {time.time() - started:.1f}s | best epoch={best_epoch} "
          f"val_loss={best_val:.4f}", flush=True)

    # ---- test: block accuracy + oracle instance replay ---------------------
    checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    t_logits, t_labels = predict_logits(model, test_loader, device)
    acc, auc = _block_metrics(t_logits, t_labels)
    print(f"\n  [test] block-level: acc={acc:.3f}  auc={auc:.3f}", flush=True)

    per_instance = predict_per_instance(model, test_loader, device)
    result = measure_instances(per_instance, reports, threshold=args.threshold)
    print("  [test] instance-level (oracle replay):", flush=True)
    print(format_report(result), flush=True)

    with open(out_dir / "variant_policy_report.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(best_path), "test": result}, handle, indent=2)
    return 0


def eval_only(args) -> int:
    device = torch.device(args.device)
    base = Path(args.data)
    test = load_split(base / "test.pt")
    reports = load_reports(base / "instances.json")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = BlockVariantPolicy(
        node_dim=int(checkpoint.get("node_dim", NODE_DIM)),
        context_dim=int(checkpoint.get("context_dim", CONTEXT_DIM)),
        hidden=int(checkpoint.get("hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    logits, labels = predict_logits(model, test_loader, device)
    acc, auc = _block_metrics(logits, labels)
    print(f"  [test] block-level: acc={acc:.3f}  auc={auc:.3f}", flush=True)
    per_instance = predict_per_instance(model, test_loader, device)
    result = measure_instances(per_instance, reports, threshold=args.threshold)
    print("  [test] instance-level (oracle replay):", flush=True)
    print(format_report(result), flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "variant_policy_report.json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(args.checkpoint), "test": result}, handle, indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the strategy-B variant policy.")
    parser.add_argument("--data", default=str(ROOT / "variant_dataset"))
    parser.add_argument("--out-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "variant_policy_best.pt"))
    args = parser.parse_args(argv)
    if args.eval_only:
        return eval_only(args)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
