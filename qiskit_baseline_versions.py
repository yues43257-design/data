#!/usr/bin/env python3
"""Recompute every Qiskit L3 claim against two pinned compiler versions.

The primary baseline is the current environment (Qiskit 2.5.2).  Qiskit 1.3.0
is a sensitivity baseline.  Circuit sources are serialized once, then the same
OpenQASM is transpiled in each isolated environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.compiler_bench.version_axis import _older_python  # noqa: E402

VERSION = "qiskit-baseline-sensitivity-v1"
PRIMARY_QISKIT = "2.5.2"
SENSITIVITY_QISKIT = "1.3.0"
CLIFFORD_T = ["cx", "h", "x", "z", "s", "sdg", "t", "tdg"]
OUT = ROOT / "reports" / "figure_data" / "qiskit_baseline_sensitivity.json"


def _ccx_count(circuit) -> int:
    return sum(1 for instruction in circuit.data
               if instruction.operation.name == "ccx")


def _claim_sources() -> List[Dict[str, Any]]:
    from qiskit import qasm2
    from qiskit.circuit.library import CDKMRippleCarryAdder, IntegerComparator

    import realworld_benchmark as RB
    import tcount_scan as TS

    sota = {row["name"]: row for row in json.loads(
        (ROOT / "reports" / "figure_data" / "realworld_sota.json").read_text()
    )}
    rows: List[Dict[str, Any]] = []

    for n in (3, 4, 5):
        circuit = RB.to_canonical(RB.circuit_mcx(n))
        rows.append({
            "circuit": f"mcx({n})",
            "family": "mcx_ladder",
            "qasm": qasm2.dumps(circuit),
            "qubits": circuit.num_qubits,
            "ours_safe": int(sota[f"mcx({n})"]["t_ours_safe"]),
            "legal_rewrite": True,
        })

    for n in (3, 4, 8, 16, 32):
        circuit = RB.to_canonical(CDKMRippleCarryAdder(n))
        rows.append({
            "circuit": f"adder({n})",
            "family": "cdkm_adder",
            "qasm": qasm2.dumps(circuit),
            "qubits": circuit.num_qubits,
            "ours_safe": int(sota[f"adder({n})"]["t_ours_safe"]),
            "legal_rewrite": True,
        })

    for n in (3, 4):
        circuit = RB.to_canonical(TS.two_adders(n))
        rows.append({
            "circuit": f"two_adders({n})",
            "family": "chained_adder",
            "qasm": qasm2.dumps(circuit),
            "qubits": circuit.num_qubits,
            "ours_safe": 4 * _ccx_count(circuit),
            "legal_rewrite": True,
        })

    for n in (3, 4):
        circuit = RB.to_canonical(IntegerComparator(n, 2))
        rows.append({
            "circuit": f"comparator({n})",
            "family": "comparator",
            "qasm": qasm2.dumps(circuit),
            "qubits": circuit.num_qubits,
            "ours_safe": 4 * _ccx_count(circuit),
            "legal_rewrite": True,
        })

    grover = RB.to_canonical(RB.circuit_grover(3))
    rows.append({
        "circuit": "grover(3)",
        "family": "grover",
        "qasm": qasm2.dumps(grover),
        "qubits": grover.num_qubits,
        "ours_safe": int(sota["grover(3)"]["t_ours_safe"]),
        "legal_rewrite": False,
    })
    return rows


def _worker(manifest: Path) -> Dict[str, Any]:
    import qiskit
    from qiskit import qasm2
    from qiskit.compiler import transpile

    rows = json.loads(manifest.read_text(encoding="utf-8"))
    outputs = []
    for row in rows:
        circuit = qasm2.loads(row["qasm"])
        values = {}
        for level in (0, 3):
            compiled = transpile(
                circuit, basis_gates=CLIFFORD_T, optimization_level=level,
            )
            values[f"qiskit_l{level}"] = sum(
                1 for instruction in compiled.data
                if instruction.operation.name in ("t", "tdg")
            )
        outputs.append({"circuit": row["circuit"], **values})
    return {"qiskit_version": qiskit.__version__, "rows": outputs}


def _run_environment(python: Path, rows: Sequence[Dict[str, Any]],
                     timeout: float) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text(json.dumps([
            {"circuit": row["circuit"], "qasm": row["qasm"]} for row in rows
        ]), encoding="utf-8")
        proc = subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--worker",
             str(manifest)],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Qiskit worker failed for {python}: "
            f"{(proc.stderr or proc.stdout).strip()[:1000]}"
        )
    return json.loads(proc.stdout)


def run(old_python: Optional[str] = None,
        timeout: float = 240.0) -> Dict[str, Any]:
    rows = _claim_sources()
    current = _run_environment(Path(sys.executable), rows, timeout)
    older = _run_environment(_older_python(old_python), rows, timeout)
    if current["qiskit_version"] != PRIMARY_QISKIT:
        raise RuntimeError(
            f"primary Qiskit is {current['qiskit_version']}, "
            f"expected {PRIMARY_QISKIT}; pin the environment or bump the artifact"
        )
    if older["qiskit_version"] != SENSITIVITY_QISKIT:
        raise RuntimeError(
            f"sensitivity Qiskit is {older['qiskit_version']}, "
            f"expected {SENSITIVITY_QISKIT}"
        )

    primary_by_name = {row["circuit"]: row for row in current["rows"]}
    older_by_name = {row["circuit"]: row for row in older["rows"]}
    report_rows = []
    for row in rows:
        primary = primary_by_name[row["circuit"]]
        sensitivity = older_by_name[row["circuit"]]
        ours = row["ours_safe"]
        pct_primary = (round(100.0 * (ours - primary["qiskit_l3"])
                             / primary["qiskit_l3"], 1)
                       if row["legal_rewrite"] else None)
        pct_sensitivity = (round(100.0 * (ours - sensitivity["qiskit_l3"])
                                 / sensitivity["qiskit_l3"], 1)
                           if row["legal_rewrite"] else None)
        report_rows.append({
            "circuit": row["circuit"],
            "family": row["family"],
            "qubits": row["qubits"],
            "ours_safe": ours,
            "legal_rewrite": row["legal_rewrite"],
            "qiskit_l0_primary": primary["qiskit_l0"],
            "qiskit_l3_primary": primary["qiskit_l3"],
            "qiskit_l0_sensitivity": sensitivity["qiskit_l0"],
            "qiskit_l3_sensitivity": sensitivity["qiskit_l3"],
            "saving_pct_primary": pct_primary,
            "saving_pct_sensitivity": pct_sensitivity,
            "pp_change": (round(pct_sensitivity - pct_primary, 1)
                          if pct_primary is not None else None),
        })

    legal = [row for row in report_rows if row["legal_rewrite"]]
    primary_values = [row["saving_pct_primary"] for row in legal]
    sensitivity_values = [row["saving_pct_sensitivity"] for row in legal]
    return {
        "version": VERSION,
        "primary_qiskit": current["qiskit_version"],
        "sensitivity_qiskit": older["qiskit_version"],
        "rows": report_rows,
        "summary": {
            "rows_with_legal_rewrite": len(legal),
            "primary_min_pct": min(primary_values),
            "primary_max_pct": max(primary_values),
            "sensitivity_min_pct": min(sensitivity_values),
            "sensitivity_max_pct": max(sensitivity_values),
            "max_abs_pp_change": max(
                abs(row["pp_change"]) for row in legal
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, default=None,
                        help="internal worker manifest")
    parser.add_argument("--old-python", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(_worker(args.worker), sort_keys=True))
        return 0

    report = run(args.old_python, args.timeout)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 104)
    print(f"  Qiskit L3 sensitivity  {VERSION}")
    print(f"  primary={report['primary_qiskit']}  "
          f"sensitivity={report['sensitivity_qiskit']}")
    print("=" * 104)
    print(f"  {'circuit':<20}{'ours':>6}{'L3 primary':>12}"
          f"{'saving':>10}{'L3 1.3':>10}{'saving':>10}")
    for row in report["rows"]:
        print(f"  {row['circuit']:<20}{row['ours_safe']:>6}"
              f"{row['qiskit_l3_primary']:>12}"
              f"{str(row['saving_pct_primary']):>10}"
              f"{row['qiskit_l3_sensitivity']:>10}"
              f"{str(row['saving_pct_sensitivity']):>10}")
    print(f"  summary={report['summary']}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
