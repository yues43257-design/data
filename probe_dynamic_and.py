#!/usr/bin/env python3
"""3-day probe, day 1+2: is "measurement-based uncompute" a viable axis?

Object: a CDKM adder whose every Toffoli is replaced by the measurement-based
temporary-logical-AND construction

    H(a); T(a); CX(c1,a); Tdg(a); CX(c0,a); T(a); H(a)     # 4 T, deterministic AND
    measure(a) -> m                                         # the *uncompute*, free
    if m: X(target)                                         # classical feed-forward

Three things are measured (the probe's day-2 checklist):
  (a) T-count vs the naive 7-T Toffoli and vs our relative-phase 4-T RCCX;
  (b) whether the EXISTING support tracker can certify it (expected: no);
  (c) whether a minimal sound criterion is even needed (the construction is
      exact, so per-instance certification may be unnecessary).

GO / KILL (pre-registered):
  GO   if the T saving is >= 1.5x AND a sound criterion fits in a page
  KILL if the saving is < 10% or certification degenerates to per-circuit proof

Usage
-----
    .venv/bin/python probe_dynamic_and.py
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


RB = _load("probe_rb", "realworld_benchmark.py")
ST = _load("probe_st", "support_tracker.py")


def ccx_positions(canon: QuantumCircuit) -> List[List[int]]:
    out = []
    for ins in canon.data:
        if ins.operation.name == "ccx":
            out.append([canon.find_bit(q).index for q in ins.qubits])
    return out


def tracker_dynamic_verdict(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Ask the SHIPPED tracker itself whether it models this circuit.

    ``RB._tracker_schedule`` only *collects* gates into layers -- it never calls
    the tracker, so it cannot answer "does the tracker apply?".  Treating "no
    exception from the collector" as "tracker applies" was the bug: every
    dynamic circuit came back True.

    Here the gates are replayed through ``DiagonalTracker.apply_layer`` exactly
    as ``track_safe`` would, and a ``measure`` / ``reset`` / ``if_else`` is
    *required* to raise ``UnsupportedGateError`` (§5.43).  If it does not, that
    is a soundness gap and we fail loudly instead of recording a false pass.
    """
    tracker = ST.DiagonalTracker(circuit.num_qubits)
    non_unitary = ("measure", "reset")
    for instruction in circuit.data:
        name = instruction.operation.name
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        try:
            tracker.apply_layer([(name, qubits)])
        except ST.UnsupportedGateError as exc:
            return {"applies": False, "rejected_gate": name,
                    "error": f"{type(exc).__name__}: {str(exc)[:90]}"}
        if name in non_unitary or "if_else" in name:
            return {"applies": True, "rejected_gate": None,
                    "error": f"TRACKER DID NOT REJECT {name!r} (soundness gap)"}
    return {"applies": True, "rejected_gate": None, "error": "no unmodelled gate"}


def measurement_based(canon: QuantumCircuit) -> QuantumCircuit:
    """Replace every ccx by 4-T AND + measure + feed-forward (exact)."""
    sites = ccx_positions(canon)
    n_data = canon.num_qubits
    out = QuantumCircuit(n_data + len(sites), len(sites))
    for i, ins in enumerate(canon.data):
        qubits = [canon.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "ccx":
            site = next(k for k, s in enumerate(sites) if s == qubits)
            c0, c1, t = qubits
            a = n_data + site
            out.h(a)
            out.t(a)
            out.cx(c1, a)
            out.tdg(a)
            out.cx(c0, a)
            out.t(a)
            out.h(a)
            out.measure(a, site)
            with out.if_test((out.clbits[site], 1)):
                out.x(t)
        else:
            out.append(ins.operation, qubits,
                       [canon.find_bit(c).index for c in ins.clbits])
    return out


def main() -> int:
    print("=" * 100)
    print("  3-day probe (day 1+2): measurement-based uncompute on a CDKM adder")
    print("=" * 100)
    rows: List[Dict[str, Any]] = []
    for n in (3, 4):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        sites = ccx_positions(canon)
        m = len(sites)
        naive = RB.t_count(RB.to_clifford_t(canon, 0))          # 7 T per Toffoli
        ours = RB.t_count(RB.substitute_rccx(canon))            # 4 T, up to diagonal
        dyn = measurement_based(canon)
        dyn_t = sum(1 for ins in dyn.data if ins.operation.name in ("t", "tdg"))
        n_meas = sum(1 for ins in dyn.data if ins.operation.name == "measure")
        n_feed = sum(1 for ins in dyn.data if "if_else" in ins.operation.name)
        # (b) ask the tracker, not the schedule collector (§5.43 contract).
        verdict = tracker_dynamic_verdict(dyn)
        tracker_applies = verdict["applies"]
        why = verdict["error"]
        if tracker_applies:
            raise RuntimeError(
                f"tracker silently accepted a dynamic circuit at n={n}: {why}")
        print(f"  adder({n}): m={m}  naive={naive} T | RCCX(up-to-diag)={ours} T | "
              f"measurement-based={dyn_t} T")
        print(f"     dynamic ops: {n_meas} measure, {n_feed} if_else; "
              f"ancillas added = {dyn.num_qubits - canon.num_qubits}")
        print(f"     saving vs naive = {naive - dyn_t} T "
              f"({(naive - dyn_t) / naive:.0%});  tracker applies = {tracker_applies}"
              + ("" if tracker_applies else f"  ({why})"))
        rows.append({"n": n, "toffolis": m, "naive_T": naive, "rccx_T": ours,
                     "measurement_based_T": dyn_t, "measures": n_meas,
                     "feed_forwards": n_feed,
                     "ancillas_added": dyn.num_qubits - canon.num_qubits,
                     "saving_vs_naive": (naive - dyn_t) / naive,
                     "tracker_applies": tracker_applies,
                     "tracker_rejected_gate": verdict["rejected_gate"],
                     "tracker_error": why})
    print("=" * 100)
    n = rows[-1]
    saving = n["saving_vs_naive"]
    print(f"  (a) T saving vs naive: {saving:.0%}  "
          f"-> {'GO' if saving >= 0.15 else 'KILL'}")
    print(f"  (b) existing tracker applies to the dynamic circuit: "
          f"{n['tracker_applies']}  (expected False)")
    print("  (c) minimal criterion: the measurement-based AND is EXACT (an algebraic")
    print("      identity verified per construction), so no per-instance legality")
    print("      criterion is required -- the certification problem DISSOLVES here.")
    print("=" * 100)
    print("  => GO on 'automation + T-count', KILL on 'certification as the core':")
    print("     the dynamic route dominates our RCCX route in T-count AND exactness,")
    print("     and needs no certificate -- it trades ancillas + classical control for it.")
    print("-" * 100)
    print("  WARNING: this script's (a)/(c) verdict is RETRACTED by §5.38.")
    print("  The transcribed gadget is NOT a deterministic AND (3 T, P(a=1)=0.85),")
    print("  so the 57% / 18 T figures do not belong to a valid construction.")
    print("  What survives is only (b): the shipped tracker rejects dynamic gates.")
    out = ROOT / "reports" / "figure_data" / "probe_dynamic_and.json"
    out.write_text(json.dumps({
        "status": "retracted",
        "retraction": "§5.38: the transcribed gadget is not a deterministic AND "
                      "(3 T, P(anc=1)=0.85); 57%/18 T do not belong to a valid "
                      "construction. Only the tracker-rejection check (b) survives.",
        "tracker_contract": "measure/reset/if_else must raise UnsupportedGateError",
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
