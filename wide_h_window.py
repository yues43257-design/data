#!/usr/bin/env python3
"""Wide circuit WITH Hadamards: does the two-level certifier have a window?

The constructed H-free circuits of ``realcircuit_window.py`` had zero rejected
sites (the tracker is exact when there is no H), so they could not answer the
applicability question.  Here we build the missing regime directly:

    a wide CDKM adder (Toffoli sites known BY CONSTRUCTION) with interference
    (H) layers inserted between its MAJ/UMA stages.

That pins the input domain to the **Toffoli layer** -- no Toffoli-localisation
problem -- while giving ``N_total >= 40`` and a controllable H density, which is
exactly what the pre-registered, scale-invariant decision rule needs:

    build the two-level certifier iff >= 30% of tracker-REJECTED sites satisfy
    ``W <= N_total / 3`` and ``N_total >= 40``.

Limitation recorded in the output: at ``N_total >= 40`` the exact oracle is
unavailable, so ``N_exact`` / ``N_useful`` cannot be measured here; only
``N_track``, ``W`` and the tail-H count are.  ``W`` is tracker-internal and
level 2 is sound by construction, so the decision rule is still valid.

Usage
-----
    .venv/bin/python wide_h_window.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from qiskit import QuantumCircuit
from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


RB = _load("wh_rb", "realworld_benchmark.py")
ST = _load("wh_tracker", "support_tracker.py")
SETUP = _load("wh_setup", "experiment_setup.py")


def interleave_h_layers(canon: QuantumCircuit, every: int, h_qubits: int) -> QuantumCircuit:
    """Insert an H layer every ``every`` Toffoli sites, on the first ``h_qubits`` qubits."""
    out = QuantumCircuit(canon.num_qubits)
    ccx_seen = 0
    width = canon.num_qubits if h_qubits <= 0 else min(h_qubits, canon.num_qubits)
    for instruction in canon.data:
        name = instruction.operation.name
        qubits = [canon.find_bit(q).index for q in instruction.qubits]
        out.append(instruction.operation, qubits,
                   [canon.find_bit(c).index for c in instruction.clbits])
        if name == "ccx":
            ccx_seen += 1
            if every and ccx_seen % every == 0 and ccx_seen < _total_ccx(canon):
                for q in range(width):
                    out.h(q)
    return out


def _total_ccx(canon: QuantumCircuit) -> int:
    return sum(1 for i in canon.data if i.operation.name == "ccx")


def final_support(schedule: Dict[str, Any], mask: int) -> Set[int]:
    support: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "cx" and int(qubits[1]) in support:
                support.add(int(qubits[0]))
        if block["target"] in support:
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
        if (mask >> index) & 1:
            support.add(int(block["target"]))
            support.update(int(q) for q in block["controls"])
            support.update(int(q) for q in block["ancillas"])
    return support


def tail_h(schedule: Dict[str, Any], site: int, support: Set[int]) -> int:
    count = 0
    for index in range(site + 1, len(schedule["blocks"])):
        for name, qubits in schedule["layers"][index]:
            if name == "h" and int(qubits[0]) in support:
                count += 1
    return count


def run_variant(label: str, canon: QuantumCircuit, every: int, h_qubits: int) -> Dict[str, Any]:
    circuit = interleave_h_layers(canon, every, h_qubits) if h_qubits else canon
    schedule = RB._tracker_schedule(circuit)
    n_total = circuit.num_qubits
    sites = len(schedule["blocks"])
    accepted = rejected = 0
    widths: List[float] = []
    tails: List[int] = []
    for site in range(sites):
        mask = 1 << site
        if ST.track_safe(schedule, mask):
            accepted += 1
            continue
        rejected += 1
        support = final_support(schedule, mask)
        widths.append(len(support) / n_total)
        tails.append(tail_h(schedule, site, support))
    hook = (sum(1 for w in widths if w <= 1 / 3) / len(widths)) if widths else 0.0
    meets = (hook >= 0.30) and (n_total >= 40)
    print(f"  {label:<30}{n_total:>6}{sites:>6}{accepted:>8}{rejected:>9}"
          f"{(f'{statistics.median(widths):.2f}' if widths else 'n/a'):>8}"
          f"{(f'{statistics.median(widths) / 1:.2f}' if False else ''):>0}"
          f"{(f'{statistics.median(tails):.0f}' if tails else 'n/a'):>8}"
          f"{hook:>7.2f}   {'BUILD' if meets else 'CLOSE'}")
    return {"variant": label, "N_total": n_total, "N_sites": sites,
            "N_track": accepted, "rejected_sites": rejected,
            "median_W_over_N": statistics.median(widths) if widths else None,
            "median_tail_H": statistics.median(tails) if tails else None,
            "hook_fraction_W_le_N3": hook,
            "decision": "BUILD level 2" if meets else "CLOSE level 2"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="wide + H window measurement.")
    parser.add_argument("--bits", type=int, default=32, help="CDKM adder size (32 -> 66 qubits)")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "wide_h_window.json"))
    args = parser.parse_args(argv)

    canon = RB.to_canonical(CDKMRippleCarryAdder(args.bits))
    n_total = canon.num_qubits
    sites = _total_ccx(canon)
    print("=" * 100)
    print(f"  wide + Hadamard window measurement   "
          f"(CDKM {args.bits}-bit, N_total={n_total}, {sites} Toffoli sites)")
    print(f"  setup_version = {SETUP.SETUP_VERSION}   "
          f"rule: hook >= 30% AND N_total >= 40")
    print("=" * 100)
    print(f"  {'variant':<30}{'N_tot':>6}{'sites':>6}{'track':>8}{'rejected':>9}"
          f"{'med W/N':>8}{'med tailH':>10}{'hook':>7}   decision")
    rows = [
        run_variant("H-free (control)", canon, 0, 0),
        run_variant("H layer every 16 ccx (sparse)", canon, 16, n_total),
        run_variant("H layer every 8 ccx", canon, 8, n_total),
        run_variant("H layer every 4 ccx (dense)", canon, 4, n_total),
    ]
    print("=" * 100)
    print("  W/N     = median support width / N_total over tracker-REJECTED sites")
    print("  tailH   = median #Hadamards after the site landing on its residual support")
    print("  hook    = fraction of rejected sites with W <= N_total/3")
    print("  NOTE: at N_total >= 40 the exact oracle is unavailable, so N_exact /")
    print("        N_useful cannot be measured here; only N_track, W and tailH are.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"setup_version": SETUP.SETUP_VERSION,
                               "script": "wide_h_window.py",
                               "oracle_available": False, "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
