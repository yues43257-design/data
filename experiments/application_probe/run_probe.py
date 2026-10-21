#!/usr/bin/env python3
"""Scoping probe: relative-phase rewrites on crypto/arithmetic circuit families.

For each family: build it, canonicalise, measure {q, blocks, ccx, H density,
in-domain}, T-count / T-depth / CNOT / ancilla / depth, compare with Qiskit L3,
TODD and pyZX full_reduce, and decide legality with the dense oracle (N<=11)
or the tracker (N>11).

Safety numbers are a **crude proxy** (labelled as such): surface-code
``physical = 2*d^2*logical`` with d=15, ``spacetime = T-count * physical``.
They are not security estimates.

Usage
-----
    .venv/bin/python experiments/application_probe/run_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    CDKMRippleCarryAdder, DraperQFTAdder, HRSCumulativeMultiplier,
    IntegerComparator, ModularAdderGate, QFT, VBERippleCarryAdder,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import realcircuit_window as RCW      # noqa: E402
import realworld_benchmark as RB      # noqa: E402
import relative_phase_criterion as RPC  # noqa: E402
import support_tracker as ST          # noqa: E402

PROBE_VERSION = "application-probe-v1"
ALLOWED = frozenset({"h", "cx", "swap", "x", "z", "s", "sdg", "t", "tdg",
                     "cz", "rz", "id", "barrier", "ccx"})
DISTANCE = 15


def _families() -> List[Dict[str, Any]]:
    def modadd(n: int) -> QuantumCircuit:
        circuit = QuantumCircuit(2 * n)
        circuit.append(ModularAdderGate(n), range(2 * n))
        return circuit.decompose()

    return [
        {"name": "cdkm_adder_3", "data": 6,
         "build": lambda: CDKMRippleCarryAdder(3)},
        {"name": "vbe_adder_3", "data": 6,
         "build": lambda: VBERippleCarryAdder(3)},
        {"name": "draper_adder_3", "data": 6,
         "build": lambda: DraperQFTAdder(3)},
        {"name": "comparator_3", "data": 6,
         "build": lambda: IntegerComparator(3, value=2)},
        {"name": "qft_4", "data": 4, "build": lambda: QFT(4)},
        {"name": "multiplier_2", "data": 8,
         "build": lambda: HRSCumulativeMultiplier(2)},
        {"name": "modular_adder_3", "data": 6, "build": lambda: modadd(3)},
    ]


def _t_depth(circuit: QuantumCircuit) -> int:
    return circuit.depth(filter_function=lambda ins:
                         ins.operation.name in ("t", "tdg"))


def _all_relative(canonical: QuantumCircuit) -> Optional[QuantumCircuit]:
    positions = [i for i, ins in enumerate(canonical.data)
                 if ins.operation.name == "ccx"]
    if not positions:
        return None
    circuit = canonical
    for position in sorted(positions, reverse=True):
        circuit = RCW.replace_one_ccx(circuit, position)
    return circuit


def _clifford_t(circuit: QuantumCircuit, level: int) -> QuantumCircuit:
    """Transpile with fallbacks for gates (cp/swap) missing from the basis."""
    last: Optional[Exception] = None
    for candidate in (circuit, circuit.decompose(),
                      circuit.decompose(reps=3)):
        try:
            return RB.to_clifford_t(candidate, level)
        except Exception as exc:          # try a deeper decomposition
            last = exc
    raise last


def measure(spec: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    row: Dict[str, Any] = {"family": spec["name"]}
    try:
        circuit = spec["build"]()
        canonical = RB.to_canonical(circuit)
    except Exception as exc:
        row.update({"error": f"{type(exc).__name__}: {exc}", "in_domain": False})
        return row

    names = [ins.operation.name for ins in canonical.data]
    blocks = names.count("ccx")
    try:
        clifford_t = _clifford_t(circuit, 1)
    except Exception as exc:
        row.update({"error": f"clifford-t transpile failed: {exc}", "in_domain": False})
        return row
    ct_names = [ins.operation.name for ins in clifford_t.data]
    h_count = ct_names.count("h")
    row.update({
        "qubits": canonical.num_qubits,
        "data_qubits": spec["data"],
        "ancilla_estimate": max(0, canonical.num_qubits - spec["data"]),
        "blocks": blocks,
        "ccx": blocks,
        "gates": len(names),
        "h_count": h_count,
        "h_density": round(h_count / max(1, len(ct_names)), 4),
        "in_domain": bool(blocks > 0
                          and set(names) <= ALLOWED
                          and not {"measure", "reset", "if_else"} & set(names)),
        "canonical_gates": sorted(set(names)),
        "t_count": RB.t_count(clifford_t),
        "t_depth": _t_depth(clifford_t),
        "cnot": ct_names.count("cx"),
        "depth": clifford_t.depth(),
        "qiskit_l3_t": RB.t_count(_clifford_t(circuit, 3)),
    })
    todd = RB.todd_t_count(circuit, timeout=30.0)
    fr = RB.pyzx_full_reduce(canonical, timeout=30.0)
    row["todd_t"] = todd[0] if todd else None
    row["full_reduce_t"] = fr[0]["t_count"] if fr and fr[0] else None

    relative = _all_relative(canonical)
    if relative is None:
        row["legality"] = "n/a (no blocks)"
        row["relative_t"] = None
        row["relative_t_depth"] = None
    elif canonical.num_qubits <= 11:
        row["legality"] = ("legal" if RPC.relative_phase_ok(canonical, relative)
                           else "illegal")
        row["relative_t"] = RB.t_count(relative)
        row["relative_t_depth"] = _t_depth(relative)
    else:
        schedule = RB._tracker_schedule(canonical)
        row["legality"] = ("legal" if ST.track_safe(schedule, (1 << blocks) - 1)
                           else "illegal")
        row["relative_t"] = RB.t_count(relative)
        row["relative_t_depth"] = _t_depth(relative)
    row["full_reduce_t_depth"] = (fr[0]["t_depth"] if fr and fr[0] else None)

    physical = row["qubits"] * 2 * DISTANCE ** 2
    row["physical_qubits_proxy"] = physical
    row["spacetime_proxy"] = row["t_count"] * physical
    row["proxy_note"] = f"crude: 2*d^2*logical with d={DISTANCE}; not a security estimate"
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def main() -> int:
    rows = [measure(spec) for spec in _families()]
    rows.append({"family": "modpow_shor_like", "in_domain": False,
                 "error": "not constructed: no library primitive; composing a "
                          "controlled modular multiplier is algorithm work "
                          "(out of the 1-2 day scope)"})
    print("=" * 112)
    print(f"  application probe  {PROBE_VERSION}")
    print("=" * 112)
    print(f"  {'family':<18}{'q':>4}{'blocks':>7}{'Hdens':>7}{'domain':>8}"
          f"{'T':>6}{'Td':>5}{'L3':>6}{'TODD':>6}{'FR':>6}{'legality':>9}")
    for r in rows:
        if "error" in r:
            print(f"  {r['family']:<18}  {r['error']}")
            continue
        print(f"  {r['family']:<18}{r['qubits']:>4}{r['blocks']:>7}"
              f"{r['h_density']:>7.3f}{str(r['in_domain']):>8}"
              f"{r['t_count']:>6}{r['t_depth']:>5}{r['qiskit_l3_t']:>6}"
              f"{str(r['todd_t']):>6}{str(r['full_reduce_t']):>6}"
              f"{r['legality']:>9}")
    in_domain = [r["family"] for r in rows if r.get("in_domain")]
    print("-" * 112)
    print(f"  in tracker domain: {in_domain}")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps({"version": PROBE_VERSION,
                               "distance": DISTANCE, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
