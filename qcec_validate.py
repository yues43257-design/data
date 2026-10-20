#!/usr/bin/env python3
"""Bidirectional QCEC validation + configuration sweep (Task 1, §5.54).

Criterion: ``C = U_cand U_ideal^dagger`` is diagonal  <=>  ``[C, Z_q] = 0`` for
every q.  QCEC decides each commutation exactly.

Direction A (must hold for every LEGAL mask)
    all q report ``equivalent``
Direction B (must hold for every ILLEGAL mask)
    at least one q reports ``not_equivalent``
Plus: at N <= 10 the QCEC verdict must agree with the dense criterion
``relative_phase_ok`` on every mask (0 disagreements).

Then: sweep QCEC configurations at N = 18 / 24 / 30 to see whether a tuned
configuration can decide the same query that the default could not.

Usage
-----
    .venv/bin/python qcec_validate.py --qcec-python /tmp/qcec-venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from qiskit import QuantumCircuit, qasm2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC      # noqa: E402
import support_tracker as ST                # noqa: E402
import ultimate_scalability_test as ULT     # noqa: E402
import wide_scale_stress as WS              # noqa: E402

VALIDATOR = "qcec-validate-v1"

_HELPER = r"""
import json, sys, time
import mqt.qcec as qcec

manifest, out, cfg_name = sys.argv[1], sys.argv[2], sys.argv[3]
CONFIGS = {
    "default":      {},
    "zx_only":      {"run_zx_checker": True, "run_simulation_checker": False,
                     "run_construction_checker": False,
                     "run_alternating_checker": False},
    "construct":    {"run_zx_checker": False, "run_simulation_checker": False,
                     "run_construction_checker": True,
                     "run_alternating_checker": False},
    "alternating":  {"run_zx_checker": False, "run_simulation_checker": False,
                     "run_construction_checker": False,
                     "run_alternating_checker": True},
    "simulation":   {"run_zx_checker": False, "run_simulation_checker": True,
                     "run_construction_checker": False,
                     "run_alternating_checker": False},
}
kwargs = dict(CONFIGS[cfg_name])
results = []
for item in json.load(open(manifest)):
    t0 = time.perf_counter()
    try:
        r = qcec.verify(item["a"], item["b"], timeout=item["timeout"], **kwargs)
        verdict = str(r.equivalence).split(".")[-1]
    except Exception as exc:
        verdict = "EXC:" + type(exc).__name__
    results.append({"q": item["q"], "verdict": verdict,
                    "seconds": round(time.perf_counter() - t0, 3)})
json.dump(results, open(out, "w"))
"""


def _diag_pairs(case_id: int, ideal: QuantumCircuit,
                candidate: QuantumCircuit) -> List[Dict[str, Any]]:
    """One QCEC query per qubit: C Z_q  vs  Z_q C."""
    n = ideal.num_qubits
    correction = QuantumCircuit(n)
    # compose() APPENDS, so later circuits multiply on the LEFT.
    # The declared residual is C = U_cand . U_ideal^dagger -- therefore the
    # inverse goes in first and the candidate second.  (Building it the other
    # way round gives U_ideal^dagger . U_cand, a different operator: for one
    # N=6 case its off-diagonal was 1.7e-16 while the declared one was 0.5.)
    correction.compose(ideal.inverse(), inplace=True)
    correction.compose(candidate, inplace=True)
    pairs = []
    for q in range(n):
        left = QuantumCircuit(n)
        left.z(q)
        left.compose(correction, inplace=True)        # Z then C = C . Z_q
        right = QuantumCircuit(n)
        right.compose(correction, inplace=True)
        right.z(q)                                     # C then Z = Z_q . C
        pairs.append({"case": case_id, "q": q,
                      "a": qasm2.dumps(left), "b": qasm2.dumps(right)})
    return pairs


def _run_config(python: str, pairs: Sequence[Dict[str, Any]], config: str,
                timeout: float, hard_timeout: float,
                tmpdir: Path) -> List[Dict[str, Any]]:
    manifest = tmpdir / "manifest.json"
    out = tmpdir / "out.json"
    manifest.write_text(json.dumps(
        [{"q": p["q"], "a": str(tmpdir / f"a{p['case']}_{p['q']}.qasm"),
          "b": str(tmpdir / f"b{p['case']}_{p['q']}.qasm"), "timeout": timeout}
         for p in pairs]))
    for p in pairs:
        (tmpdir / f"a{p['case']}_{p['q']}.qasm").write_text(p["a"])
        (tmpdir / f"b{p['case']}_{p['q']}.qasm").write_text(p["b"])
    try:
        subprocess.run([python, "-c", _HELPER, str(manifest), str(out), config],
                       check=True, timeout=hard_timeout,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return [{"q": p["q"], "verdict": "HARD-TIMEOUT", "seconds": hard_timeout}
                for p in pairs]
    return json.loads(out.read_text())


def bidirectional(python: str, timeout: float, hard_timeout: float) -> Dict[str, Any]:
    """Dense agreement + both directions, N <= 10."""
    print("\n  (A) bidirectional check + dense agreement (N <= 10)")
    rows: List[Dict[str, Any]] = []
    contradictions: List[Dict[str, Any]] = []
    inconclusive: List[Dict[str, Any]] = []
    missed_illegal: List[Dict[str, Any]] = []
    cases = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for width, blocks, h_prob in ((4, 3, 0.7), (6, 3, 0.7), (8, 2, 0.7),
                                      (4, 4, 0.0), (6, 4, 0.0)):
            inst = ULT.WideInstance(width=width, blocks=blocks, seed=20261020,
                                    h_prob=h_prob, ancilla_pool=2)
            if inst.total > 10:
                continue
            ideal = inst.circuit(0)
            schedule = WS.build_schedule(inst)
            pairs: List[Dict[str, Any]] = []
            truth: List[Dict[str, Any]] = []
            for mask in range(1 << blocks):
                candidate = inst.circuit(mask)
                dense_legal = bool(RPC.relative_phase_ok(ideal, candidate))
                tracked = bool(ST.track_safe(schedule, mask))
                pairs.extend(_diag_pairs(mask, ideal, candidate))
                truth.append({"case": mask, "mask": mask, "N": inst.total,
                              "dense_legal": dense_legal, "tracked": tracked})
            results = _run_config(python, pairs, "default", timeout, hard_timeout, tmpdir)
            cases += len(truth)
            by_case: Dict[int, List[Dict[str, Any]]] = {}
            for r, p in zip(results, pairs):
                by_case.setdefault(p["case"], []).append(r)
            for t in truth:
                verdicts = by_case.get(t["case"], [])
                n_equiv = sum(1 for r in verdicts if r["verdict"] == "equivalent")
                n_ne = sum(1 for r in verdicts if r["verdict"] == "not_equivalent")
                n_unknown = len(verdicts) - n_equiv - n_ne
                qcec_legal = (n_equiv == len(verdicts)) and n_unknown == 0
                qcec_illegal = n_ne > 0
                if qcec_legal:
                    qcec_verdict = "legal"
                elif qcec_illegal:
                    qcec_verdict = "illegal"
                else:
                    qcec_verdict = "unknown"
                t.update({"q_eq": n_equiv, "q_ne": n_ne, "q_unknown": n_unknown,
                          "qcec_legal": qcec_legal, "qcec_illegal": qcec_illegal,
                          "qcec_verdict": qcec_verdict})
                rows.append(t)
                if qcec_verdict == "unknown":
                    inconclusive.append(t)
                elif (qcec_verdict == "legal") != t["dense_legal"]:
                    contradictions.append(t)
                if (not t["dense_legal"]) and not qcec_illegal:
                    if qcec_verdict != "unknown":
                        missed_illegal.append(t)
    legal = sum(1 for r in rows if r["dense_legal"])
    illegal = len(rows) - legal
    print(f"      cases={len(rows)} (dense-legal={legal}, dense-illegal={illegal})")
    print(f"      definite QCEC vs dense contradictions = {len(contradictions)}")
    print(f"      QCEC inconclusive (some q unknown)    = {len(inconclusive)}")
    print(f"      illegal masks with NO q detecting it = {len(missed_illegal)}")
    for d in contradictions[:3]:
        print(f"        e.g. {d}")
    return {"cases": len(rows), "dense_legal": legal, "dense_illegal": illegal,
            "contradictions": contradictions, "inconclusive": inconclusive,
            "missed_illegal": missed_illegal,
            "rows": rows}


def config_sweep(python: str, timeout: float, hard_timeout: float) -> Dict[str, Any]:
    """N = 18 / 24 / 30 under several QCEC configurations (single q = 0)."""
    print("\n  (B) configuration sweep at N = 18 / 24 / 30 (qubit q=0 only)")
    configs = ("default", "zx_only", "construct", "alternating", "simulation")
    sizes = ((16, 2), (18, 6), (24, 6))
    out: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for width, pool in sizes:
            inst = ULT.WideInstance(width=width, blocks=6, seed=3, h_prob=0.0,
                                    ancilla_pool=pool)
            ideal, candidate = inst.circuit(0), inst.circuit((1 << 6) - 1)
            row = {"N": inst.total, "gates": len(ideal.data)}
            for kind in ("legal", "illegal_explicit"):
                ideal_k = inst.circuit(0)
                candidate_k = inst.circuit((1 << 6) - 1)
                if kind == "illegal_explicit":
                    # guaranteed non-diagonal: append H on the last block's
                    # target, so C = H_t . D with D diagonal  =>  not diagonal.
                    candidate_k = candidate_k.copy()
                    candidate_k.h(inst.targets[-1])
                # a non-diagonal residual only breaks commutation on the qubit
                # it touches, so probe q = the H's target for the illegal case
                probe_q = 0 if kind == "legal" else int(inst.targets[-1])
                pairs = _diag_pairs(0, ideal_k, candidate_k)[probe_q:probe_q + 1]
                for cfg in configs:
                    key = f"{kind}_{cfg}"
                    if cfg == "simulation" and inst.total > 18:
                        # statevector needs 2**N amplitudes; 30q is ~16 GB here.
                        row[key] = "skipped(2^N)"
                        row[key + "_s"] = 0.0
                        continue
                    res = _run_config(python, pairs, cfg, timeout, hard_timeout, tmpdir)
                    row[key] = res[0]["verdict"]
                    row[key + "_s"] = res[0]["seconds"]
            print(f"      N={row['N']:>3} " + "  ".join(
                f"{k}={row[k]}({row[k + '_s']}s)" for k in row
                if isinstance(row[k], str) and not k.endswith("_s")), flush=True)
            out.append(row)
    return {"rows": out, "configs": list(configs)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qcec-python", default="/tmp/qcec-venv/bin/python")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--hard-timeout", type=float, default=120)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "qcec_validate.json"))
    args = parser.parse_args(argv)
    if not Path(args.qcec_python).exists():
        print(f"QCEC interpreter not found: {args.qcec_python}")
        return 2
    started = time.perf_counter()
    report = {"validator": VALIDATOR, "qcec_python": args.qcec_python}
    report["bidirectional"] = bidirectional(args.qcec_python, args.timeout,
                                            args.hard_timeout)
    report["sweep"] = config_sweep(args.qcec_python, args.timeout, args.hard_timeout)
    report["seconds"] = round(time.perf_counter() - started, 1)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    ok = (len(report["bidirectional"]["contradictions"]) == 0 and
          len(report["bidirectional"]["missed_illegal"]) == 0)
    print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
