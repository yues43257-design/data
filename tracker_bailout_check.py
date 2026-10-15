#!/usr/bin/env python3
"""How much recall would a *sound* tracker-v2 actually recover?

!!!! NOT FOR CERTIFICATION USE !!!!
``naive_v2`` below is **UNSOUND** and is kept only as the diagnostic that proves
it: the "type flip" model assumes the residual is a Pauli, but ``H T H`` maps a
diagonal residual to ``(1/sqrt2)(Z - Y)``, which is *not* diagonal.  Never route
this file's predicates into a certification path.  See
``test_tracker_soundness.py`` for the permanent counterexample tests.

v2 idea: replace the permanent ``broken`` flag with a per-qubit X/Z type, so
``Z -> X -> Z`` through an even number of Hadamards is accepted.

That is **unsound as stated** (``H T H`` is not a Pauli conjugation), so the
sound version must add a **bail-out**: as soon as a gate that is not compatible
with the type model acts on an X-type (flipped) qubit, reject.

This script measures the three quantities that decide whether v2 is worth
building:

    naive gain   : masks the old tracker rejects but the *unsound* v2 accepts
    sound gain   : masks the old tracker rejects and the *sound* v2 accepts
    UNSOUND(snd) : masks the sound v2 accepts but the exact verifier rejects

Usage
-----
    .venv/bin/python tracker_bailout_check.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


ST = _load("bo_tracker", "support_tracker.py")
SYM = _load("bo_symbolic", "symbolic_verifier.py")
SETUP = _load("bo_setup", "experiment_setup.py")


def naive_v2(schedule: Dict[str, Any], mask: int) -> bool:
    """Type-flip tracker WITHOUT bail-out (the unsound version)."""
    support: Set[int] = set()
    flipped: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "h":
                q = int(qubits[0])
                if q in support:
                    flipped.symmetric_difference_update({q})
            elif name == "cx":
                a, b = int(qubits[0]), int(qubits[1])
                if b in support:                    # Z-type: target -> control
                    support.add(a)
                if a in flipped:                    # X-type: control -> target
                    support.add(b)
                    flipped.add(b)
        if block["target"] in support:
            support.update(block["controls"])
            support.update(block["ancillas"])
        if (mask >> index) & 1:
            support.add(block["target"])
            support.update(block["controls"])
            support.update(block["ancillas"])
    return not flipped


def sound_v2(schedule: Dict[str, Any], mask: int) -> bool:
    """Type-flip tracker WITH bail-out on non-compatible gates."""
    support: Set[int] = set()
    flipped: Set[int] = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "h":
                q = int(qubits[0])
                if q in support:
                    flipped.symmetric_difference_update({q})
            elif name == "cx":
                a, b = int(qubits[0]), int(qubits[1])
                if b in support:                    # Z-type: target -> control
                    support.add(a)
                if a in flipped:                    # X-type: control -> target
                    support.add(b)
                    flipped.add(b)
            elif name in ("t", "tdg", "rz", "ccz"):     # non-Clifford diagonal
                if set(int(q) for q in qubits) & flipped:
                    return False
        acting = set(int(q) for q in block["controls"]) | \
            set(int(q) for q in block["ancillas"]) | {int(block["target"])}
        if acting & flipped:                            # non-Clifford permutation
            return False
        if block["target"] in support:
            support.update(block["controls"])
            support.update(block["ancillas"])
        if (mask >> index) & 1:
            support.add(block["target"])
            support.update(block["controls"])
            support.update(block["ancillas"])
    return not flipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="sound tracker-v2 payoff check.")
    parser.add_argument("--controls", default="2,3,4,5")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--h-probs", default="0.1,0.3,0.7")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "tracker_bailout_check.json"))
    args = parser.parse_args(argv)
    controls = [int(c) for c in args.controls.split(",") if c.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]

    cols = ["exact_legal", "old_accept", "naive_v2_accept", "sound_v2_accept"]
    site_cols = [c + "_site" for c in cols]
    print("=" * 104)
    print("  would a *sound* tracker-v2 pay off?  (exact verifier = ground truth)")
    print("=" * 104)
    print(f"  {'h':>5}{'legal':>7}{'old':>6}{'naive_v2':>9}{'sound_v2':>9}"
          f"{'naive gain':>11}{'SOUND gain':>11}{'UNSOUND naive':>14}"
          f"{'UNSOUND sound':>14}")

    rows: List[Dict[str, Any]] = []
    for h_prob in h_probs:
        legal = old = naive = sound = 0
        naive_unsound = sound_unsound = 0
        s_legal = s_old = s_naive = s_sound = 0
        for num_controls in controls:
            for seed in range(args.seeds):
                instance_seed = SETUP.seed_for(seed)
                verifier = SYM.ReducedSymbolicVerifier(num_controls, args.blocks,
                                                       instance_seed, h_prob)
                schedule = ST.homogeneous_schedule(num_controls, args.blocks,
                                                   instance_seed, h_prob)
                for mask in range(1 << args.blocks):
                    exact = bool(verifier.is_safe(mask))
                    old_ok = bool(ST.track_safe(schedule, mask))
                    n_ok = naive_v2(schedule, mask)
                    s_ok = sound_v2(schedule, mask)
                    legal += int(exact); old += int(old_ok)
                    naive += int(n_ok); sound += int(s_ok)
                    naive_unsound += int(n_ok and not exact)
                    sound_unsound += int(s_ok and not exact)
                    if bin(mask).count("1") == 1:
                        s_legal += int(exact); s_old += int(old_ok)
                        s_naive += int(n_ok); s_sound += int(s_ok)
        row = {"h_prob": h_prob, "exact_legal": legal, "old_accept": old,
               "naive_v2_accept": naive, "sound_v2_accept": sound,
               "naive_gain": naive - old, "sound_gain": sound - old,
               "naive_unsound": naive_unsound, "sound_unsound": sound_unsound,
               "site_exact_legal": s_legal, "site_old_accept": s_old,
               "site_naive_v2_accept": s_naive, "site_sound_v2_accept": s_sound}
        rows.append(row)
        print(f"  {h_prob:>5.1f}{legal:>7}{old:>6}{naive:>9}{sound:>9}"
              f"{naive - old:>11}{sound - old:>11}{naive_unsound:>14}"
              f"{sound_unsound:>14}")

    print("-" * 104)
    print("  single-block (1-bit) masks only -- the metric the deployment uses")
    print(f"  {'h':>5}{'legal':>7}{'old':>6}{'naive_v2':>9}{'sound_v2':>9}"
          f"{'old recall':>12}{'naive rec':>11}{'sound rec':>11}")
    for row in rows:
        def rec(num, den):
            return f"{num / den:.3f}" if den else "n/a"
        print(f"  {row['h_prob']:>5.1f}{row['site_exact_legal']:>7}"
              f"{row['site_old_accept']:>6}{row['site_naive_v2_accept']:>9}"
              f"{row['site_sound_v2_accept']:>9}"
              f"{rec(row['site_old_accept'], row['site_exact_legal']):>12}"
              f"{rec(row['site_naive_v2_accept'], row['site_exact_legal']):>11}"
              f"{rec(row['site_sound_v2_accept'], row['site_exact_legal']):>11}")
    print("=" * 104)
    print("  naive gain = old tracker rejects, UNSOUND v2 accepts (upper bound)")
    print("  SOUND gain = old tracker rejects, SOUND v2 accepts  (what we would get)")
    print("  UNSOUND sound must be 0; UNSOUND naive > 0 is the H-T-H counterexample")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"setup_version": SETUP.SETUP_VERSION,
               "script": "tracker_bailout_check.py", "rows": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
