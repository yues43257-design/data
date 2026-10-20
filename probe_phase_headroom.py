#!/usr/bin/env python3
"""4.3 feasibility probe (measure first, build later).

Question: among masks the tracker REJECTS, how many are

  (a) actually legal  -> false alarms, recoverable in principle;
  (b) of those, how many have a **second H on the same qubit** after the first
      hit -- i.e. recoverable by a cheap Pauli-type tracker (H: X <-> Z),
      without a full phase-polynomial engine;
  (c) the rest -> need genuine phase-term cancellation, i.e. the full 4.3.

No new tracker is written.  The exact verdict comes from the existing
``ReducedSymbolicVerifier``; the hit analysis comes from ``certify``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import experiment_setup as SETUP      # noqa: E402
import support_tracker as ST          # noqa: E402
import symbolic_verifier as SYM       # noqa: E402


def _stage(c: Dict[str, Any]) -> int:
    return int(c["stage"].split()[-1])


def analyse(schedule: Dict[str, Any], mask: int, legal: bool) -> Dict[str, Any]:
    cert = ST.certify(schedule, mask)
    if cert["accepted"]:
        return {"mask": mask, "accepted": True}
    hits = [c for c in cert["h_checks"] if c["in_support"]]
    first = hits[0]
    later_same = [c for c in hits if c["qubit"] == first["qubit"]
                  and _stage(c) > _stage(first)]
    # A *sound* Pauli-type tracker needs everything acting on q between the
    # two H's to be a permutation (CX / block).  T, Tdg, S, Z on q map X to
    # something that is not Z again (`H T H` is the classic counterexample).
    sound_type = False
    if later_same and legal:
        q = first["qubit"]
        l1, l2 = _stage(first), _stage(later_same[0])
        ok = True
        for i in range(l1 + 1, l2 + 1):
            for name, qubits in schedule["layers"][i]:
                if i == l2 and name == "h" and int(qubits[0]) == q:
                    continue                      # the second H itself
                if q in [int(x) for x in qubits] and name != "cx":
                    ok = False
            # blocks between are permutations, hence harmless
        sound_type = ok
    return {"mask": mask, "accepted": False, "legal": legal,
            "first_hit_stage": _stage(first), "first_hit_qubit": first["qubit"],
            "n_hits": len(hits),
            "second_h_on_same_qubit": bool(later_same),
            "type_recoverable_upper": bool(legal and later_same),
            "type_recoverable": bool(legal and sound_type),
            "needs_phase_terms": bool(legal and not sound_type)}


def main() -> int:
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for nc in (2, 3, 4, 5):
        for h_prob in (0.1, 0.3, 0.7):
            for seed_index in range(3):
                seed = SETUP.seed_for(seed_index)
                verifier = SYM.ReducedSymbolicVerifier(nc, 6, seed, h_prob)
                schedule = ST.homogeneous_schedule(nc, 6, seed, h_prob)
                for mask in range(1 << 6):
                    legal = bool(verifier.is_safe(mask))
                    rows.append(analyse(schedule, mask, legal))
    rejected = [r for r in rows if not r["accepted"]]
    false_alarm = [r for r in rejected if r.get("legal")]
    true_reject = [r for r in rejected if not r.get("legal")]
    type_rec = [r for r in false_alarm if r["type_recoverable"]]
    upper = [r for r in false_alarm if r["type_recoverable_upper"]]
    phase_only = [r for r in false_alarm if r["needs_phase_terms"]]
    n = len(rows)
    print("=" * 100)
    print("  4.3 feasibility probe: rejected masks, false alarms, cheap recoverability")
    print("=" * 100)
    print(f"  masks={n}  accepted={n - len(rejected)}  rejected={len(rejected)}")
    print(f"  rejected but LEGAL (false alarms) = {len(false_alarm)} "
          f"({len(false_alarm) / len(rejected):.1%} of rejections)")
    print(f"  rejected and ILLEGAL (correct)    = {len(true_reject)}")
    print(f"  of the false alarms:")
    print(f"    second H anywhere later on that qubit (upper bound) = {len(upper)} "
          f"({len(upper) / max(1, len(false_alarm)):.1%})")
    print(f"    SOUND Pauli-type (only permutations between)        = {len(type_rec)} "
          f"({len(type_rec) / max(1, len(false_alarm)):.1%})")
    print(f"    need genuine phase cancellation                    = {len(phase_only)} "
          f"({len(phase_only) / max(1, len(false_alarm)):.1%})")
    if phase_only:
        import statistics
        print(f"    [phase-only] mean downgraded blocks = "
              f"{statistics.mean(bin(r['mask']).count('1') for r in phase_only):.2f}, "
              f"mean H hits = {statistics.mean(r['n_hits'] for r in phase_only):.2f}")
        for r in phase_only[:3]:
            print(f"      e.g. mask={r['mask']:#08b} hits={r['n_hits']} "
                  f"first_hit=(layer {r['first_hit_stage']}, q{r['first_hit_qubit']})")
    print(f"  ({time.perf_counter() - started:.1f}s)")
    print("=" * 100)
    verdict = ("GO" if len(false_alarm) / max(1, len(rejected)) > 0.5 else
               "BORDERLINE" if len(false_alarm) / max(1, len(rejected)) >= 0.2 else
               "KILL")
    print(f"  headroom (false alarms / rejections) = "
          f"{len(false_alarm) / max(1, len(rejected)):.1%}  -> {verdict}")
    out = ROOT / "reports" / "figure_data" / "probe_phase_headroom.json"
    out.write_text(json.dumps({
        "masks": n, "rejected": len(rejected),
        "false_alarms": len(false_alarm), "correct_rejects": len(true_reject),
        "type_recoverable": len(type_rec), "needs_phase_terms": len(phase_only),
        "headroom": len(false_alarm) / max(1, len(rejected)),
        "verdict": verdict, "rows": rows}, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
