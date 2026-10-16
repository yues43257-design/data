#!/usr/bin/env python3
"""Item 0: do the independent "truth" paths actually agree?

Three truth sources are in play and each may carry its own convention
(left/right multiplication, ancilla handling, post-selection):

    1. relative_phase_criterion  -- C = U_cand . U_ideal^dagger diagonal,
                                    optionally restricted to the clean-ancilla subspace
    2. mixed_stress.MixedVerifier -- data-register measurement statistics,
                                    ancillas post-selected clean
    3. symbolic_verifier.ReducedSymbolicVerifier -- same criterion, reduced space

If (1) and (2) disagree on the same circuits, then §5.2/§5.3 (which rest on (2))
and §5.5 (which rests on (3)) cannot be trusted either, and recomputing the
tables would be wasted work.

Usage
-----
    .venv/bin/python truth_crosscheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mixed_stress as MS                 # noqa: E402
import relative_phase_criterion as RPC    # noqa: E402


def main() -> int:
    print("=" * 88)
    print("  item 0: truth-source cross-check  (MixedVerifier vs corrected criterion)")
    print("=" * 88)
    total_masks = total_agree = 0
    for counts, seed, h_prob in (([2, 3, 4], 11, 0.7),
                                 ([3, 2, 4, 2], 13, 0.7),
                                 ([2, 4], 17, 0.1),
                                 ([4, 3, 2], 19, 0.0)):
        n_max, total, data, ancillas = MS.mixed_layout(counts)
        instance = MS.MixedInstance(list(counts), seed, h_prob)
        verifier = instance.verifier
        base = MS.build_mixed_circuit(counts, seed, h_prob, 0)[0]
        k = len(counts)
        print(f"  counts={counts} N={total} ancillas={ancillas} h={h_prob} "
              f"({1 << k} masks)")
        agree = 0
        for mask in range(1 << k):
            candidate = MS.build_mixed_circuit(counts, seed, h_prob, mask)[0]
            mv = bool(verifier.is_safe(mask))
            new = RPC.relative_phase_ok(base, candidate, ancillas=ancillas)
            total_masks += 1
            agree += int(mv == new)
            total_agree += int(mv == new)
            flag = "" if mv == new else "   <-- DISAGREE"
            print(f"     mask {mask:>2}: MixedVerifier={str(mv):>5} "
                  f"corrected={str(new):>5}{flag}")
        print(f"     agree {agree}/{1 << k}")
    print("=" * 88)
    print(f"  TOTAL agreement: {total_agree}/{total_masks}")
    if total_agree == total_masks:
        print("  -> the two truth paths agree on this sample; §5.2/§5.3 may be recomputed")
    else:
        print("  -> DISAGREEMENT: fix the conventions before recomputing any table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
