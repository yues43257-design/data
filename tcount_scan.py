#!/usr/bin/env python3
"""Family scan: where does the relative-phase pass actually reduce T-count?

The pricing law says (verified 14/14 on a controlled series)

    FR_full_reduce = 4m + 3k        ours = 4m        naive = 7m
    k = #{(c0,c1,t) triples with ODD multiplicity}   (controls unordered, by target)

so

    advantage over the naive 7-T Toffoli : 3m   (always, ~43%)
    advantage over pyZX full_reduce      : 3k   (only where k > 0)

This script ranks real circuit families by k/m, because that ratio -- not the
absolute T-count -- decides whether the pass beats the *strong* baseline.

Usage
-----
    .venv/bin/python tcount_scan.py
"""

from __future__ import annotations

import argparse
import collections
import importlib.machinery
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


RB = _load("scan_rb", "realworld_benchmark.py")


def triples(canon: QuantumCircuit) -> List[Tuple[int, int, int]]:
    out = []
    for ins in canon.data:
        if ins.operation.name == "ccx":
            a, b, t = (canon.find_bit(q).index for q in ins.qubits)
            out.append((min(a, b), max(a, b), t))
    return out


def fr_tcount(canon: QuantumCircuit, timeout: float = 30.0) -> Optional[int]:
    import signal

    class _TO(Exception):
        pass

    def _raise(s, f):
        raise _TO()

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        import qiskit.qasm2 as qasm2
        circ = RB.LM.zx.Circuit.from_qasm(qasm2.dumps(canon)).to_basic_gates()
        graph = circ.to_graph()
        RB.LM.zx.simplify.full_reduce(graph)
        return int(RB.LM.compute_ftq_metrics(RB.LM.zx.extract_circuit(graph)).t_count)
    except Exception:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def two_adders(n: int) -> QuantumCircuit:
    a = RB.to_canonical(CDKMRippleCarryAdder(n))
    b = RB.to_canonical(CDKMRippleCarryAdder(n))
    out = QuantumCircuit(max(a.num_qubits, b.num_qubits))
    for src in (a, b):
        for ins in src.data:
            out.append(ins.operation, [src.find_bit(q).index for q in ins.qubits])
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="T-count family scan.")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "tcount_scan.json"))
    args = parser.parse_args(argv)

    families: List[Tuple[str, Any]] = [
        ("CDKM adder", lambda n: CDKMRippleCarryAdder(n)),
        ("two CDKM adders", two_adders),
        ("comparator", lambda n: IntegerComparator(n, value=2)),
        ("MCX ladder (ours)", RB.circuit_mcx),
        ("Grover oracle", RB.circuit_grover),
    ]
    sizes = (3, 4, 5, 6)

    print("=" * 108)
    print("  family scan: where does the pass beat the STRONG baseline (pyZX full_reduce)?")
    print("=" * 108)
    print(f"  {'family':<20}{'n':>3}{'q':>4}{'m':>5}{'k':>5}{'k/m':>7}"
          f"{'naive':>7}{'ours':>7}{'FR':>7}{'ours-FR':>9}{'naive-ours':>12}"
          f"{'QiskitL0':>8}{'L3':>7}{'blk2':>8}{'blk4':>8}")
    rows: List[Dict[str, Any]] = []
    for name, build in families:
        for n in sizes:
            try:
                qc = build(n)
                canon = RB.to_canonical(qc)
            except Exception as exc:
                print(f"  {name:<20}{n:>3}  build failed: {type(exc).__name__}")
                continue
            tri = triples(canon)
            if not tri:
                print(f"  {name:<20}{n:>3}  no Toffoli sites (not in the declared layer)")
                continue
            m = len(tri)
            k = sum(1 for v in collections.Counter(tri).values() if v % 2 == 1)
            naive, ours = 7 * m, 4 * m
            l0 = RB.t_count(RB.to_clifford_t(canon, 0))
            l3 = RB.t_count(RB.to_clifford_t(canon, 3))
            started = time.perf_counter()
            fr = fr_tcount(canon)
            fr_s = time.perf_counter() - started
            def blocked(circ, chunks):
                data = list(circ.data); nn = len(data); tot = 0
                for i in range(chunks):
                    sub = QuantumCircuit(circ.num_qubits)
                    for ins in data[i*nn//chunks:(i+1)*nn//chunks]:
                        sub.append(ins.operation,
                                   [circ.find_bit(q).index for q in ins.qubits])
                    val = fr_tcount(sub)
                    if val is None:
                        return None
                    tot += val
                return tot
            b2 = blocked(canon, 2)
            b4 = blocked(canon, 4)
            print(f"  {name:<20}{n:>3}{canon.num_qubits:>4}{m:>5}{k:>5}{k / m:>7.2f}"
                  f"{naive:>7}{ours:>7}{(fr if fr is not None else -1):>7}"
                  f"{(ours - fr) if fr is not None else 0:>9}"
                  f"{naive - ours:>12}"
                  f"{l0:>8}{l3:>7}"
                  f"{(b2 if b2 is not None else -1):>8}"
                  f"{(b4 if b4 is not None else -1):>8}")
            rows.append({"family": name, "n": n, "qubits": canon.num_qubits,
                         "m": m, "k": k, "k_over_m": k / m, "naive_T": naive,
                         "ours_T": ours, "fr_T": fr, "fr_seconds": fr_s})
    print("=" * 108)
    print("  ours-FR > 0 means the pass beats pyZX full_reduce on this instance")
    rows.sort(key=lambda r: (-r["k_over_m"], r["family"], r["n"]))
    print("  ranking by k/m (higher = the pass has more room vs the strong baseline):")
    for r in rows:
        print(f"    {r['family']:<20} n={r['n']}  k/m={r['k_over_m']:.2f}  "
              f"ours-FR={r['ours_T'] - r['fr_T'] if r['fr_T'] is not None else 'n/a'}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
