#!/usr/bin/env python3
"""Optimization / removal attacks against the relative-phase watermark.

Usage
-----
    .venv/bin/python experiments/watermark_probe/attacks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB      # noqa: E402
import experiments.watermark_probe.detect as D   # noqa: E402
import experiments.watermark_probe.embed as E    # noqa: E402

PROBE_VERSION = "watermark-probe-v1"


def flip_blocks(watermarked: Dict[str, Any], count: int = 1) -> QuantumCircuit:
    key = list(watermarked["key_used"])
    flipped = 0
    for i, bit in enumerate(key):
        if bit == 1 and flipped < count:
            key[i] = 0
            flipped += 1
    return E.embed(key, watermarked["variant"])["circuit"]


def _full_reduce(circuit: QuantumCircuit) -> QuantumCircuit:
    zx = RB.LM.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()
    graph = zx.to_graph()
    RB.LM.zx.simplify.full_reduce(graph)
    return qasm2.loads(RB.LM.zx.extract_circuit(graph).to_qasm(),
                       custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def _todd(circuit: QuantumCircuit) -> QuantumCircuit:
    zx = RB.LM.zx.Circuit.from_qasm(qasm2.dumps(circuit)).to_basic_gates()
    optimised = RB.LM.zx.optimize.phase_block_optimize(zx.copy(), quiet=True)
    return qasm2.loads(optimised.to_qasm(),
                       custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


def _row(name: str, circuit: QuantumCircuit, watermarked: Dict[str, Any],
         note: str = "") -> Dict[str, Any]:
    bits, found, total = D.detect_bits(circuit)
    key = watermarked["key_used"]
    return {
        "attack": name,
        "note": note,
        "capacity": watermarked["capacity"],
        "blocks_found": found,
        "bits": bits,
        "key": key,
        "ber": D.ber(bits, key),
        "t_count": D.count_t(circuit),
        "t_count_watermarked": watermarked["t_count"],
        "t_overhead_vs_all_exact": watermarked["t_overhead_vs_all_exact"],
        "t_overhead_vs_all_relative": watermarked["t_overhead_vs_all_relative"],
    }


def run_all(watermarked: Dict[str, Any], write: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(_row("none", watermarked["circuit"], watermarked))
    for name, fn in (("pyzx_full_reduce", _full_reduce), ("todd", _todd)):
        try:
            optimized = fn(watermarked["circuit"])
            rows.append(_row(name, optimized, watermarked))
        except Exception as exc:
            rows.append({"attack": name, "error": f"{type(exc).__name__}: {exc}",
                         "ber": None, "blocks_found": None, "t_count": None,
                         "capacity": watermarked["capacity"]})
    flipped = flip_blocks(watermarked, 1)
    rows.append(_row("flip_one", flipped, watermarked,
                     "one embedded bit flipped 1->0"))
    all_exact = E.embed([0] * watermarked["capacity"], watermarked["variant"])["circuit"]
    rows.append(_row("all_exact", all_exact, watermarked,
                     "attacker strips the watermark by re-synthesising exact blocks"))
    rows.append(_row("false_positive", E.source_circuit(watermarked["variant"]),
                     watermarked, "unwatermarked circuit vs the key"))
    if write:
        out = Path(__file__).resolve().parent / "results.json"
        out.write_text(json.dumps({"version": PROBE_VERSION,
                                   "variant": watermarked["variant"],
                                   "capacity": watermarked["capacity"],
                                   "key": watermarked["key_used"],
                                   "rows": rows}, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    key = [1, 0, 1, 1, 0, 1, 0, 0]
    report = {"version": PROBE_VERSION, "variants": {}}
    for variant in ("cdkm3", "cdkm4"):
        watermarked = E.embed(key, variant)
        rows = run_all(watermarked, write=False)
        report["variants"][variant] = {
            "capacity": watermarked["capacity"],
            "key": watermarked["key_used"],
            "t_count": watermarked["t_count"],
            "t_all_exact": 7 * watermarked["capacity"],
            "t_all_relative": 4 * watermarked["capacity"],
            "rows": rows,
        }
        print("=" * 100)
        print(f"  relative-phase watermark  {variant}  capacity="
              f"{watermarked['capacity']} bits  T={watermarked['t_count']}")
        print("=" * 100)
        print(f"  {'attack':<20}{'blocks':>7}{'BER':>8}{'T':>6}")
        for r in rows:
            ber_text = f"{r['ber']:.2f}" if r["ber"] is not None else "n/a"
            print(f"  {r['attack']:<20}{str(r['blocks_found']):>7}"
                  f"{ber_text:>8}{str(r.get('t_count')):>6}"
                  + (f"  {r['error']}" if r.get("error") else ""))
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
