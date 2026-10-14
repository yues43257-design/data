#!/usr/bin/env python3
"""Deployment runtime for the strategy-B per-block variant policy.

This module glues the three pieces together *without duplicating* any of them:

* ``generate_variant_dataset`` provides the instance builder (``build_instance``),
  the per-block featuriser (``instance_samples``) and the oracle
  (``oracle_for_instance`` / ``measurement_distributions``),
* ``train_variant_policy`` provides the model architecture,
* the caller (``evaluate_model.py``) provides ``Linux-MCX`` for the FTQC
  T-count bookkeeping.

``evaluate_instance`` therefore measures exactly the same quantity the trainer
reports, but for a freshly built instance: the T-count of the policy's chosen
mix of exact / relative-phase MCX blocks, the all-exact baseline, the oracle
ceiling and the (illegal) all-relative circuit, plus a live measurement-statistics
verification of the policy's own choice.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
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


def load_generator():
    """Import ``generate_variant_dataset.py`` (instance builder + oracle)."""
    return _load("variant_generator_runtime", "generate_variant_dataset.py")


def load_trainer():
    """Import ``train_variant_policy.py`` (model architecture)."""
    return _load("variant_policy_trainer_runtime", "train_variant_policy.py")


def load_symbolic():
    """Import ``symbolic_verifier.py`` (incremental safety verifier)."""
    return _load("variant_symbolic_runtime", "symbolic_verifier.py")


def load_policy(checkpoint_path, device):
    """Rebuild the trained policy from a checkpoint and put it in eval mode."""
    trainer = load_trainer()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = trainer.BlockVariantPolicy(
        node_dim=int(checkpoint.get("node_dim", trainer.NODE_DIM)),
        context_dim=int(checkpoint.get("context_dim", trainer.CONTEXT_DIM)),
        hidden=int(checkpoint.get("hidden", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _batch_blocks(samples: List[Data], device) -> Batch:
    """Batch only the fields the policy consumes (avoids collating metadata)."""
    items = []
    for sample in samples:
        items.append(
            Data(
                x=torch.cat([sample.x.float(), sample.x_support.float()], dim=-1),
                edge_index=sample.edge_index,
                edge_attr=sample.edge_attr,
                context=sample.context.view(1, -1).float(),
            )
        )
    return Batch.from_data_list(items).to(device)


def block_probabilities(gen, model, report, device) -> List[float]:
    """Downgrade probability for every block of one instance."""
    samples = gen.instance_samples(report, int(report["instance_id"]))
    batch = _batch_blocks(samples, device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).detach().cpu().numpy()
    return [float(p) for p in probs]


def mask_from_probs(probs: List[float], threshold: float = 0.5) -> int:
    mask = 0
    for index, prob in enumerate(probs):
        if prob >= threshold:
            mask |= 1 << index
    return mask


def t_count(linux_mcx, gen, circuit) -> Tuple[int, int]:
    """Independent FTQC T-count / T-depth of a Qiskit circuit."""
    source = linux_mcx.zx.Circuit.from_qasm(gen.qasm2.dumps(circuit)).to_basic_gates()
    metrics = linux_mcx.compute_ftq_metrics(source)
    return int(metrics.t_count), int(metrics.t_depth)


def evaluate_instance(
    linux_mcx,
    gen,
    model,
    num_controls: int,
    blocks: int,
    seed: int,
    h_prob: float,
    device,
    threshold: float = 0.5,
    decoder: str = "greedy",
    verifier: str = "symbolic",
) -> Dict[str, Any]:
    """Run the policy on one instance and measure it against the oracle.

    ``decoder="greedy"`` (default) is the deployment decoder: blocks are
    considered in order of the model's downgrade confidence and each is accepted
    only if the verifier -- a full simulation of the whole circuit -- still
    finds the measurement statistics unchanged.  ``decoder="threshold"`` takes
    every block whose probability exceeds ``threshold`` at once.
    """
    config = gen.VariantConfig(h_prob=h_prob)
    report = gen.oracle_for_instance(num_controls, blocks, seed, config)
    report["instance_id"] = 0
    probs = block_probabilities(gen, model, report, device)
    rows = report["mask_table"]
    tolerance = config.tolerance

    if verifier == "symbolic":
        symbolic = load_symbolic().ReducedSymbolicVerifier(
            num_controls, blocks, seed, h_prob, tolerance=tolerance
        )

        def verify(mask: int) -> Tuple[bool, float]:
            return symbolic.is_safe(mask), symbolic.deviation(mask)
    else:
        exact_circuit, _ = gen.build_instance(num_controls, blocks, 0, seed, h_prob)
        reference = gen.measurement_distributions(exact_circuit, num_controls)

        def verify(mask: int) -> Tuple[bool, float]:
            circuit, _ = gen.build_instance(num_controls, blocks, mask, seed, h_prob)
            dist = gen.measurement_distributions(circuit, num_controls)
            tv = max(gen.total_variation(reference[basis], dist[basis]) for basis in reference)
            return tv <= tolerance, tv

    threshold_mask = mask_from_probs(probs, threshold)
    threshold_safe, threshold_tv = verify(threshold_mask)

    greedy_mask = 0
    for block in sorted(range(blocks), key=lambda b: -probs[b]):
        candidate = greedy_mask | (1 << block)
        ok, _ = verify(candidate)
        if ok:
            greedy_mask = candidate

    chosen = greedy_mask if decoder == "greedy" else threshold_mask
    chosen_safe, chosen_tv = verify(chosen)
    chosen_circuit, _ = gen.build_instance(num_controls, blocks, chosen, seed, h_prob)
    policy_t, policy_td = t_count(linux_mcx, gen, chosen_circuit)
    threshold_circuit, _ = gen.build_instance(num_controls, blocks, threshold_mask, seed, h_prob)
    threshold_t, _ = t_count(linux_mcx, gen, threshold_circuit)

    return {
        "num_controls": int(num_controls),
        "blocks": int(blocks),
        "seed": int(seed),
        "decoder": decoder,
        "probs": probs,
        "mask": int(chosen),
        "greedy_mask": int(greedy_mask),
        "threshold_mask": int(threshold_mask),
        "oracle_mask": int(report["oracle_mask"]) if report["oracle_mask"] is not None else -1,
        "t_all_exact": int(rows[0]["t_count"]),
        "t_oracle": int(report["oracle_t_count"]),
        "t_policy": policy_t,
        "t_threshold": threshold_t,
        "t_greedy": int(rows[greedy_mask]["t_count"]),
        "t_all_relative": int(rows[-1]["t_count"]),
        "policy_usable": bool(chosen_safe),
        "threshold_usable": bool(threshold_safe),
        "all_relative_usable": bool(report["all_relative_usable"]),
        "verified_tv": float(chosen_tv),
        "threshold_tv": float(threshold_tv),
        "fragmented": bool(report["fragmented"]),
    }


def load_rl():
    """Import ``train_variant_policy_rl.py`` (oracle-free instance builder)."""
    return _load("variant_policy_rl_runtime", "train_variant_policy_rl.py")


def evaluate_large_instance(
    linux_mcx,
    gen,
    model,
    num_controls: int,
    blocks: int,
    seed: int,
    h_prob: float,
    device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Budget-constrained solve for a circuit too large to enumerate.

    Uses the oracle-free instance builder (so ``blocks`` may be in the tens) and
    the incremental symbolic verifier: the greedy pass costs O(1) per block
    instead of a full simulation per candidate.
    """
    rl = load_rl()
    instance = rl.Instance(num_controls, blocks, seed, h_prob, gen.VariantConfig(h_prob=h_prob))
    batch = Batch.from_data_list(instance.blocks_data).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).tolist()
    verifier = instance.symbolic
    greedy_mask = verifier.greedy(probs, threshold=0.0)
    threshold_mask = verifier.mask_from_probs(probs, threshold)
    all_relative = (1 << blocks) - 1
    return {
        "num_controls": int(num_controls),
        "blocks": int(blocks),
        "seed": int(seed),
        "probs": probs,
        "t_all_exact": int(instance.t_all_exact),
        "t_greedy": int(instance.t_policy(greedy_mask)),
        "t_threshold": int(instance.t_policy(threshold_mask)),
        "t_all_relative": int(instance.t_policy(all_relative)),
        "greedy_mask": int(greedy_mask),
        "downgrades_greedy": bin(greedy_mask).count("1"),
        "greedy_safe": bool(verifier.is_safe(greedy_mask)),
        "threshold_safe": bool(verifier.is_safe(threshold_mask)),
        "all_relative_safe": bool(verifier.is_safe(all_relative)),
    }
