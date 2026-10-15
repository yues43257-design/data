#!/usr/bin/env python3
"""Real-circuit locality measurement: is the two-level certifier worth building?

For every Toffoli *site* of a constructed real circuit we record

    N            : number of Toffoli sites (and the register size N_total)
    N_exact      : sites whose single-site downgrade is legal by the EXACT verifier
    N_track      : sites the support tracker accepts          (its conservativeness)
    N_useful     : sites that are exact-legal AND unpaired    (only these save T)
    W, W/N_total : support width of the tracker-REJECTED sites (the two-level hook)
    tail_H       : Hadamards after the site that land on its residual support

Pre-registered, scale-invariant decision rule (agreed before running):
    build the two-level certifier for a family iff >= 30% of its tracker-rejected
    sites satisfy ``W <= N_total / 3`` and that family has ``N_total >= 40``.

Circuits are constructed (Toffoli positions known exactly) so that the Toffoli
*localisation* problem -- the step most likely to blow up on QASMBench -- is not
in the critical path.  Benchmark suites are deliberately deferred.

Usage
-----
    .venv/bin/python realcircuit_window.py
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

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import CDKMRippleCarryAdder, IntegerComparator

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LM = _load("rw_lm", "Linux-MCX")
RB = _load("rw_rb", "realworld_benchmark.py")
ST = _load("rw_tracker", "support_tracker.py")
SETUP = _load("rw_setup", "experiment_setup.py")

EXACT_LIMIT = 11


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


def tail_hadamards(schedule: Dict[str, Any], site: int, support: Set[int]) -> int:
    """Hadamards after ``site`` that act on the site's residual support."""
    count = 0
    for index in range(site + 1, len(schedule["blocks"])):
        for name, qubits in schedule["layers"][index]:
            if name == "h" and int(qubits[0]) in support:
                count += 1
    return count


def replace_one_ccx(canon: QuantumCircuit, gate_position: int) -> QuantumCircuit:
    """Copy ``canon`` but turn the ccx at ``gate_position`` into 4-T RCCX."""
    out = QuantumCircuit(canon.num_qubits)
    for position, instruction in enumerate(canon.data):
        qubits = [canon.find_bit(q).index for q in instruction.qubits]
        if position == gate_position and instruction.operation.name == "ccx":
            c0, c1, target = qubits
            for gate in LM._rccx_gates(c0, c1, target):
                if gate[0] == "cx":
                    out.cx(gate[1], gate[2])
                else:
                    getattr(out, gate[0])(gate[1])
        else:
            out.append(instruction.operation, qubits,
                       [canon.find_bit(c).index for c in instruction.clbits])
    return out


def measure(label: str, qc: QuantumCircuit) -> Optional[Dict[str, Any]]:
    canon = RB.to_canonical(qc)
    n_total = canon.num_qubits
    ccx_positions = [i for i, ins in enumerate(canon.data)
                     if ins.operation.name == "ccx"]
    if not ccx_positions:
        print(f"  [skip] {label}: no Toffoli sites after canonicalisation")
        return None
    schedule = RB._tracker_schedule(canon)
    triples = []
    for position in ccx_positions:
        q = [canon.find_bit(x).index for x in canon.data[position].qubits]
        triples.append((min(q[0], q[1]), max(q[0], q[1]), q[2]))
    import collections
    counter = collections.Counter(triples)
    unpaired = {t for t, c in counter.items() if c % 2 == 1}

    exact_legal = track_ok = useful = 0
    widths: List[float] = []
    tails: List[int] = []
    exact_available = n_total <= EXACT_LIMIT
    for site, position in enumerate(ccx_positions):
        mask = 1 << site
        track = bool(ST.track_safe(schedule, mask))
        track_ok += track
        ok = None
        if exact_available:
            ok = RB.relative_phase_ok(canon, replace_one_ccx(canon, position),
                                      EXACT_LIMIT)
        exact_legal += int(bool(ok))
        if ok and triples[site] in unpaired:
            useful += 1
        if not track:
            support = final_support(schedule, mask)
            widths.append(len(support) / n_total)
            tails.append(tail_hadamards(schedule, site, support))

    hook = (sum(1 for w in widths if w <= 1 / 3) / len(widths)) if widths else 0.0
    meets = (hook >= 0.30) and (n_total >= 40)
    return {
        "circuit": label, "N_total": n_total, "N_sites": len(ccx_positions),
        "N_exact_legal": exact_legal if exact_available else None,
        "N_track": track_ok, "N_useful": useful if exact_available else None,
        "exact_available": exact_available,
        "rejected_sites": len(widths),
        "median_W_over_N": statistics.median(widths) if widths else None,
        "hook_fraction_W_le_N3": hook, "median_tail_H": statistics.median(tails) if tails else None,
        "decision": "BUILD level 2" if meets else "CLOSE level 2",
        "setup_version": SETUP.SETUP_VERSION,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="real-circuit locality measurement.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "realcircuit_window.json"))
    args = parser.parse_args(argv)

    circuits: List[tuple] = [("adder-3bit", CDKMRippleCarryAdder(3)),
                             ("adder-4bit", CDKMRippleCarryAdder(4)),
                             ("comparator-3bit", IntegerComparator(3, value=2)),
                             ("comparator-4bit", IntegerComparator(4, value=2))]
    for n in (3, 4, 5):
        circuits.append((f"mcx-ladder-{n}", qasm2.loads(
            LM.make_mcx_source(n, basis="clifford_t").to_qasm())))

    print("=" * 100)
    print("  real-circuit locality  (constructed circuits; exact verifier <= 11q)")
    print(f"  setup_version = {SETUP.SETUP_VERSION}   "
          f"pre-registered rule: hook >= 30% AND N_total >= 40")
    print("=" * 100)
    print(f"  {'circuit':<18}{'N_tot':>6}{'sites':>6}{'exact':>7}{'track':>7}"
          f"{'useful':>7}{'rej':>5}{'W/N':>7}{'tailH':>7}{'hook':>7}   decision")
    rows = []
    for label, qc in circuits:
        row = measure(label, qc)
        if row is None:
            continue
        rows.append(row)
        def fmt(key, dec=2):
            v = row[key]
            return "n/a" if v is None else (f"{v:.{dec}f}" if isinstance(v, float) else f"{v}")
        print(f"  {row['circuit']:<18}{row['N_total']:>6}{row['N_sites']:>6}"
              f"{fmt('N_exact_legal'):>7}{row['N_track']:>7}{fmt('N_useful'):>7}"
              f"{row['rejected_sites']:>5}{fmt('median_W_over_N'):>7}"
              f"{fmt('median_tail_H', 1):>7}{row['hook_fraction_W_le_N3']:>7.2f}"
              f"   {row['decision']}")
    print("=" * 100)
    print("  W/N = median support width / register size, over tracker-REJECTED sites")
    print("  hook = fraction of rejected sites with W <= N_total/3 (level-2 window)")
    print("  tailH = median #Hadamards after the site landing on its residual support")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"setup_version": SETUP.SETUP_VERSION,
                               "script": "realcircuit_window.py", "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
