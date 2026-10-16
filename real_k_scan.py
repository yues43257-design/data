#!/usr/bin/env python3
"""How dense are UNPAIRED Toffolis in real algorithm classes?

This is the measurement that decides case D (relaxation-aware global
optimisation): if ``k/m`` (or the residual gain) is ~0 on real classes, the
relaxation freedom buys nothing there and D closes; if it is substantially > 0,
that class is the demo.

QASMBench / RevLib are not available offline, so this builds faithful structural
analogues of the same classes:

    one-hot QROM / lookup   : 2**k multi-controlled X, one per address  (unpaired)
    unary iteration         : nested-prefix MCX chain                  (unpaired)
    comparator network      : m independent comparators                (paired inside)
    modexp                  : m chained CDKM adders                    (paired inside)
    Grover oracle           : one marked-state MCX

``k`` = number of (controls-unordered, target) triples with ODD multiplicity.
``GAIN = max(0, full_reduce - 4m)`` is a lower bound on the relaxation gain.

Usage
-----
    .venv/bin/python real_k_scan.py
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
from qiskit.circuit.library import CDKMRippleCarryAdder, IntegerComparator

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


RB = _load("rks_rb", "realworld_benchmark.py")


class _TO(Exception):
    pass


def _raise(s, f):
    raise _TO()


def fr_tcount(circ: QuantumCircuit, timeout: float = 60.0):
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


def qrom_like(k: int) -> QuantumCircuit:
    """One-hot lookup: 2**k multi-controlled X, each with its own control set."""
    qc = QuantumCircuit(k + (1 << k))
    for a in range(1 << k):
        controls = [q for q in range(k) if (a >> q) & 1]
        if not controls:
            qc.x(k + a)                    # address 0 -> plain X, no Toffoli
            continue
        qc.mcx(controls, k + a, mode="noancilla")
    return qc


def unary_like(k: int) -> QuantumCircuit:
    """Nested-prefix chain: mcx([0..i], i+1) for i = 1..k-1."""
    qc = QuantumCircuit(k + 1)
    for i in range(1, k):
        qc.mcx(list(range(i)), i, mode="noancilla")   # controls 0..i-1
    return qc


def comparator_net(m: int) -> QuantumCircuit:
    parts = [RB.to_canonical(IntegerComparator(3, value=v)) for v in range(m)]
    width = max(p.num_qubits for p in parts)
    out = QuantumCircuit(width)
    for p in parts:
        for ins in p.data:
            out.append(ins.operation, [p.find_bit(q).index for q in ins.qubits])
    return out


def modexp_like(m: int) -> QuantumCircuit:
    parts = [RB.to_canonical(CDKMRippleCarryAdder(3)) for _ in range(m)]
    width = max(p.num_qubits for p in parts)
    out = QuantumCircuit(width)
    for p in parts:
        for ins in p.data:
            out.append(ins.operation, [p.find_bit(q).index for q in ins.qubits])
    return out


def main() -> int:
    families: List[Tuple[str, Any, List[int]]] = [
        ("one-hot QROM", qrom_like, [2, 3, 4]),
        ("unary iteration", unary_like, [3, 4, 5, 6]),
        ("comparator network", comparator_net, [2, 3, 4]),
        ("modexp (chained adder)", modexp_like, [2, 3, 4]),
        ("Grover oracle", RB.circuit_grover, [3]),
    ]
    print("=" * 104)
    print("  real-class scan: how dense are UNPAIRED Toffolis?  (decides case D)")
    print("=" * 104)
    print(f"  {'class':<26}{'n':>3}{'q':>5}{'m':>5}{'k':>5}{'k/m':>7}"
          f"{'ours':>7}{'FR':>7}{'GAIN':>7}")
    rows: List[Dict[str, Any]] = []
    for name, build, sizes in families:
        for n in sizes:
            try:
                canon = RB.to_canonical(build(n))
            except Exception as exc:
                print(f"  {name:<26}{n:>3}  build failed {type(exc).__name__}")
                continue
            tri = []
            for ins in canon.data:
                if ins.operation.name == "ccx":
                    a, b, t = (canon.find_bit(q).index for q in ins.qubits)
                    tri.append((min(a, b), max(a, b), t))
            if not tri:
                print(f"  {name:<26}{n:>3}  no Toffoli sites")
                continue
            m = len(tri)
            k = sum(1 for v in collections.Counter(tri).values() if v % 2 == 1)
            ours = 4 * m
            fr = fr_tcount(canon)
            gain = max(0, fr - ours) if fr is not None else None
            print(f"  {name:<26}{n:>3}{canon.num_qubits:>5}{m:>5}{k:>5}{k / m:>7.2f}"
                  f"{ours:>7}{(fr if fr is not None else -1):>7}"
                  f"{(gain if gain is not None else -1):>7}")
            rows.append({"class": name, "n": n, "qubits": canon.num_qubits, "m": m,
                         "k": k, "k_over_m": k / m, "ours": ours, "fr": fr,
                         "gain": gain})
    print("=" * 104)
    print("  k/m >> 0  -> case D has a target in this class")
    print("  k/m ~ 0   -> relaxation buys nothing here")
    out = ROOT / "reports" / "figure_data" / "real_k_scan.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
