#!/usr/bin/env python3
"""How much is the RELAXATION freedom actually worth?

For a family, the relaxation-aware gain is

    gain = max(0, T_best_exact - T_up_to_diagonal)

with ``T_up_to_diagonal`` = our pass (4 T per Toffoli) and ``T_best_exact``
approximated by pyZX ``full_reduce`` (an upper bound on the exact optimum, so the
gain is a *lower* bound).  If this is O(1) on every family, a
relaxation-aware global optimiser is worth only a constant; if it grows with the
number of Toffolis, there is a real research target.

The families below are the ones the earlier scan was missing -- structures with
*unpaired* Toffolis, where the residual is expected to be non-trivial:
AND-tree, compute-only adder prefix, Toffoli fan-out, unary-style chain.

NOTE: the earlier ``blk2/blk4`` columns are VOID -- ``blk4 > naive`` on a single
adder is impossible without double counting, so that harness is buggy and its
numbers must not be used.

Usage
-----
    .venv/bin/python residual_scan.py
"""

from __future__ import annotations

import collections
import importlib.machinery
import importlib.util
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


RB = _load("rs_rb", "realworld_benchmark.py")


class _TO(Exception):
    pass


def _raise(s, f):
    raise _TO()


def fr_tcount(circ: QuantumCircuit, timeout: float = 30.0):
    import qiskit.qasm2 as qasm2

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        c = RB.LM.zx.Circuit.from_qasm(qasm2.dumps(circ)).to_basic_gates()
        g = c.to_graph()
        RB.LM.zx.simplify.full_reduce(g)
        return int(RB.LM.compute_ftq_metrics(RB.LM.zx.extract_circuit(g)).t_count)
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def and_tree(n: int) -> QuantumCircuit:
    """Compute-only AND chain: n-2 Toffolis on distinct triples (all unpaired)."""
    qc = QuantumCircuit(n + 1)
    anc = list(range(n, n + n - 2))
    total = n + 1 + max(0, n - 2)
    qc = QuantumCircuit(total)
    if n == 2:
        qc.ccx(0, 1, n)
        return qc
    qc.ccx(0, 1, n)
    for i in range(1, n - 1):
        qc.ccx(n + i - 1, i + 1, n + i)
    return qc


def compute_only_adder(n: int) -> QuantumCircuit:
    """First half of a CDKM adder: the MAJ chain without its UMA uncompute."""
    canon = RB.to_canonical(CDKMRippleCarryAdder(n))
    data = list(canon.data)
    half = len(data) // 2
    out = QuantumCircuit(canon.num_qubits)
    for ins in data[:half]:
        out.append(ins.operation, [canon.find_bit(q).index for q in ins.qubits])
    return out


def fanout(n: int) -> QuantumCircuit:
    """Same controls, distinct targets: every triple is unique -> k = m."""
    qc = QuantumCircuit(n + 2)
    for t in range(2, n + 2):
        qc.ccx(0, 1, t)
    return qc


def main() -> int:
    families: List[Tuple[str, Any]] = [
        ("AND tree (compute only)", and_tree),
        ("compute-only adder", compute_only_adder),
        ("Toffoli fan-out", fanout),
        ("CDKM adder (paired, control)", lambda n: CDKMRippleCarryAdder(n)),
    ]
    print("=" * 96)
    print("  relaxation gain = max(0, full_reduce - 4m)   [lower bound on the real gain]")
    print("=" * 96)
    print(f"  {'family':<30}{'n':>3}{'m':>5}{'k':>5}{'k/m':>7}{'ours':>7}"
          f"{'FR':>6}{'GAIN':>7}")
    rows: List[Dict[str, Any]] = []
    for name, build in families:
        for n in (3, 4, 5, 6, 8):
            try:
                canon = RB.to_canonical(build(n))
            except Exception as exc:
                print(f"  {name:<30}{n:>3}  build failed {type(exc).__name__}")
                continue
            tri = []
            for ins in canon.data:
                if ins.operation.name == "ccx":
                    a, b, t = (canon.find_bit(q).index for q in ins.qubits)
                    tri.append((min(a, b), max(a, b), t))
            if not tri:
                print(f"  {name:<30}{n:>3}  no Toffoli sites")
                continue
            m = len(tri)
            k = sum(1 for v in collections.Counter(tri).values() if v % 2 == 1)
            ours = 4 * m
            fr = fr_tcount(canon)
            gain = max(0, fr - ours) if fr is not None else None
            print(f"  {name:<30}{n:>3}{m:>5}{k:>5}{k / m:>7.2f}{ours:>7}"
                  f"{(fr if fr is not None else -1):>6}"
                  f"{(gain if gain is not None else -1):>7}")
            rows.append({"family": name, "n": n, "m": m, "k": k,
                         "k_over_m": k / m, "ours": ours, "fr": fr, "gain": gain})
    print("=" * 96)
    out = ROOT / "reports" / "figure_data" / "residual_scan.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
