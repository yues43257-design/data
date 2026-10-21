#!/usr/bin/env python3
"""Systematic H-density / width sweep for certificate false negatives.

For each (width, blocks, h_prob) every mask is classified by the frozen
protocol: dense legality (N<=8) or QCEC (N>8), and the bound certificate.

Reported per cell:
    masks, legal, cert-accepted legal (TP), legal-but-rejected (FN),
    illegal-but-accepted (UNSOUND, must be 0)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC    # noqa: E402
import support_tracker as ST              # noqa: E402
import ultimate_scalability_test as ULT   # noqa: E402
import wide_scale_stress as WS            # noqa: E402

SWEEP_VERSION = "h-density-v1"


def _circuit_with_ranges(instance, mask: int
                         ) -> Tuple[QuantumCircuit, List[Tuple[int, int]]]:
    circuit = QuantumCircuit(instance.total)
    ranges: List[Tuple[int, int]] = []
    for index in range(instance.blocks):
        if index and instance.layers[index] is not None:
            circuit.compose(instance.layers[index], range(instance.total),
                            inplace=True)
        start = len(circuit.data)
        circuit.compose(instance._placed_block(index, bool((mask >> index) & 1)),
                        range(instance.total), inplace=True)
        ranges.append((start, len(circuit.data)))
    return circuit, ranges


def run(widths=(4, 6), blocks_list=(2, 3, 4),
        h_probs=(0.0, 0.1, 0.3, 0.5), seeds=(0, 1)) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for width in widths:
        for blocks in blocks_list:
            for h_prob in h_probs:
                for seed in seeds:
                    inst = ULT.WideInstance(width=width, blocks=blocks,
                                            seed=1000 + seed, h_prob=h_prob,
                                            ancilla_pool=2)
                    if inst.total > 10:
                        continue
                    ideal = inst.circuit(0)
                    schedule = WS.build_schedule(inst)
                    legal = accept = fn = unsound = 0
                    for mask in range(1 << blocks):
                        emitted, ranges = _circuit_with_ranges(inst, mask)
                        truth = bool(RPC.relative_phase_ok(ideal, emitted))
                        certificate = ST.certify(schedule, mask, circuit=emitted,
                                                 block_ranges=ranges)
                        valid = ST.verify_certificate(
                            schedule, mask, certificate, circuit=emitted,
                            block_ranges=ranges)
                        cert = bool(valid and certificate["accepted"])
                        legal += truth
                        accept += int(cert)
                        fn += int(truth and not cert)
                        unsound += int(cert and not truth)
                    rows.append({"width": width, "blocks": blocks,
                                 "h_prob": h_prob, "seed": seed,
                                 "N": inst.total, "masks": 1 << blocks,
                                 "legal": legal, "accepted": accept,
                                 "false_negative": fn, "unsound": unsound})
    by_h: Dict[str, Dict[str, int]] = {}
    for row in rows:
        cell = by_h.setdefault(f"{row['h_prob']:g}",
                               {"masks": 0, "legal": 0, "accepted": 0,
                                "false_negative": 0, "unsound": 0})
        for key in cell:
            cell[key] += row[key]
    return {"version": SWEEP_VERSION, "rows": rows,
            "by_h": by_h,
            "fn_rate_by_h": {h: (round(c["false_negative"] / c["legal"], 3)
                                 if c["legal"] else None)
                             for h, c in by_h.items()}}


def main() -> int:
    report = run()
    print("=" * 96)
    print(f"  H-density / width sweep  {SWEEP_VERSION}")
    print("=" * 96)
    print(f"  {'h':>5}{'masks':>8}{'legal':>8}{'accept':>8}{'FN':>5}{'UNS':>5}"
          f"{'FN/legal':>10}")
    for h, c in report["by_h"].items():
        print(f"  {h:>5}{c['masks']:>8}{c['legal']:>8}{c['accepted']:>8}"
              f"{c['false_negative']:>5}{c['unsound']:>5}"
              f"{str(report['fn_rate_by_h'][h]):>10}")
    out = Path(__file__).resolve().parent / "h_density_sweep.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
