#!/usr/bin/env python3
"""§5.15 / §5.16 recomputed under the DECLARED definition, with a real
conservativeness number and a confidence interval.

Declared definition (frozen in §5.23): LEGAL(candidate|ideal) iff
``C = U_cand . U_ideal^dagger`` is diagonal with unit-modulus entries.  A CDKM
adder has no post-selected ancillas, so the criterion is evaluated on the full
space.

The earlier attempt used ``(legal - accepted)/legal``, which silently assumes
``accepted subset legal``.  Here we report the full 2x2 table:

        legal & accepted   (hit)          legal & rejected   (MISS)
        illegal & accepted (UNSOUND!)     illegal & rejected (fine)

and the two derived numbers

        conservativeness = 1 - hit / legal        (how much recall is lost)
        unsoundness      = unsafe_accept / accepted (must be 0)

Usage
-----
    .venv/bin/python recompute_conservativeness.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import realworld_benchmark as RB          # noqa: E402
import realcircuit_window as RCW          # noqa: E402
import support_tracker as ST              # noqa: E402
import wide_h_window as WH                # noqa: E402
import relative_phase_criterion as RPC    # noqa: E402


def wilson(successes: int, total: int, z: float = 1.96):
    if not total:
        return float("nan"), float("nan"), float("nan")
    centre = (successes + z * z / 2) / (total + z * z)
    half = z * math.sqrt(centre * (1 - centre) / (total + z * z))
    return (successes / total, max(0.0, centre - half), min(1.0, centre + half))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "recompute_conservativeness.json"))
    args = parser.parse_args(argv)

    print("=" * 104)
    print("  §5.15/§5.16 recomputed under the DECLARED definition "
          "(clean-ancilla post-selection)")
    print("=" * 104)
    print(f"  {'variant':<30}{'sites':>6}{'legal':>7}{'track':>7}{'HIT':>6}"
          f"{'MISS':>6}{'UNSOUND':>9}{'conserv':>9}")
    rows: List[Dict[str, Any]] = []
    L = T = HIT = MISS = UNS = 0
    for n in (3, 4):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        for every in (0, 1, 2, 3, 4):
            for h_qubits in (canon.num_qubits, max(2, canon.num_qubits // 2)):
                if every == 0 and h_qubits != canon.num_qubits:
                    continue
                circuit = (WH.interleave_h_layers(canon, every, h_qubits)
                           if every else canon)
                schedule = RB._tracker_schedule(circuit)
                positions = [i for i, ins in enumerate(circuit.data)
                             if ins.operation.name == "ccx"]
                legal = track = hit = miss = unsound = 0
                for site, position in enumerate(positions):
                    mask = 1 << site
                    accepted = bool(ST.track_safe(schedule, mask))
                    is_legal = bool(RPC.relative_phase_ok(
                        circuit, RCW.replace_one_ccx(circuit, position)))
                    track += accepted
                    legal += is_legal
                    hit += int(accepted and is_legal)
                    miss += int((not accepted) and is_legal)
                    unsound += int(accepted and not is_legal)
                tag = ("H-free" if every == 0
                       else f"H/{every}ccx-{'full' if h_qubits == canon.num_qubits else 'half'}")
                label = f"adder{n}-{tag}"
                conserv = (1 - hit / legal) if legal else float("nan")
                print(f"  {label:<30}{len(positions):>6}{legal:>7}{track:>7}{hit:>6}"
                      f"{miss:>6}{unsound:>9}{conserv:>9.2f}")
                rows.append({"variant": label, "sites": len(positions),
                             "legal": legal, "tracker": track, "hit": hit,
                             "miss": miss, "unsound": unsound,
                             "conservativeness": conserv})
                L += legal
                T += track
                HIT += hit
                MISS += miss
                UNS += unsound
    print("=" * 104)
    p_conserv, lo, hi = wilson(MISS, L)
    p_unsound = (UNS / T) if T else float("nan")
    print(f"  POOLED  instances={len(rows)}  sites={sum(r['sites'] for r in rows)}")
    print(f"    legal={L}  tracker_accepted={T}")
    print(f"    HIT(legal&accepted)={HIT}   MISS(legal&rejected)={MISS}   "
          f"UNSOUND(accepted&illegal)={UNS}")
    print(f"    conservativeness = MISS/legal = {p_conserv:.3f} "
          f"(95% Wilson CI {lo:.3f} .. {hi:.3f})")
    print(f"    unsoundness      = UNS/accepted = {p_unsound:.3f}  (must be 0)")
    print("=" * 104)
    print("  count alignment: legal + illegal = sites;  accepted = HIT + UNSOUND;")
    print("                   rejected = MISS + (illegal&rejected)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "declared_definition": "C = U_cand U_ideal^dagger diagonal (full space; "
                               "no post-selected ancillas in CDKM adders)",
        "rows": rows,
        "pooled": {"instances": len(rows), "sites": sum(r["sites"] for r in rows),
                   "legal": L, "tracker": T, "hit": HIT, "miss": MISS,
                   "unsound": UNS, "conservativeness": p_conserv,
                   "conservativeness_ci": [lo, hi], "unsoundness": p_unsound}},
        indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
