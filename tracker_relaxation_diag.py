#!/usr/bin/env python3
"""Diagnose *why* support_tracker rejects masks the exact verifier accepts.

The tracker marks a mask ``broken`` the moment an H acts on a supported qubit,
and never un-breaks it.  But ``H X H = Z``: a correction that goes Z -> X -> Z
through a *pair* of Hadamards is diagonal again at the end.  That is one concrete
over-conservatism channel, and this script measures how much of the recall loss
it explains.

For every mask with ``exact_legal = True`` and ``tracker_safe = False`` we replay
the tracker, find the **first** H that hits a supported qubit, and then count how
many further H's act on that same qubit:

    even parity after the break  -> "restorable"  (Z -> X -> Z)
    odd  parity after the break  -> still X-type at the end

Only exact-verifier-available sizes are used, so ``exact_legal`` is ground truth.

Usage
-----
    .venv/bin/python tracker_relaxation_diag.py
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


ST = _load("rd_tracker", "support_tracker.py")
SYM = _load("rd_symbolic", "symbolic_verifier.py")


def first_break_detail(schedule: Dict[str, Any], mask: int):
    """Replay the tracker; return (broke?, qubit, layer_index) at the first H hit."""
    support = set()
    for index, block in enumerate(schedule["blocks"]):
        for name, qubits in schedule["layers"][index]:
            if name == "h":
                if qubits[0] in support:
                    return True, int(qubits[0]), index
            elif name == "cx":
                if qubits[1] in support:
                    support.add(qubits[0])
        if block["target"] in support:
            support.update(block["controls"])
            support.update(block["ancillas"])
        if (mask >> index) & 1:
            support.add(block["target"])
            support.update(block["controls"])
            support.update(block["ancillas"])
    return False, None, None


def h_parity_after(schedule: Dict[str, Any], qubit: int, start: int) -> int:
    count = 0
    for index in range(start, len(schedule["blocks"])):
        for name, qubits in schedule["layers"][index]:
            if name == "h" and int(qubits[0]) == int(qubit):
                count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="tracker over-conservatism diagnosis.")
    parser.add_argument("--controls", default="2,3,4,5")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--h-probs", default="0.1,0.3,0.7")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "tracker_relaxation_diag.json"))
    args = parser.parse_args(argv)
    controls = [int(c) for c in args.controls.split(",") if c.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]

    rows: List[Dict[str, Any]] = []
    print("=" * 90)
    print("  why does the tracker reject masks the exact verifier accepts?")
    print("=" * 90)
    print(f"  {'h':>5}{'legal':>7}{'tracker':>9}{'rejected&legal':>16}"
          f"{'restorable':>12}{'share':>8}")
    for h_prob in h_probs:
        legal_total = track_total = rejected_legal = restorable = 0
        for num_controls in controls:
            for seed in range(args.seeds):
                verifier = SYM.ReducedSymbolicVerifier(num_controls, args.blocks,
                                                       seed * 5 + 3, h_prob)
                schedule = ST.homogeneous_schedule(num_controls, args.blocks,
                                                   seed * 5 + 3, h_prob)
                for mask in range(1 << args.blocks):
                    legal = bool(verifier.is_safe(mask))
                    track = bool(ST.track_safe(schedule, mask))
                    legal_total += int(legal)
                    track_total += int(track)
                    if legal and not track:
                        rejected_legal += 1
                        broke, qubit, index = first_break_detail(schedule, mask)
                        if broke and h_parity_after(schedule, qubit, index + 1) % 2 == 0:
                            restorable += 1
        share = (restorable / rejected_legal) if rejected_legal else 0.0
        print(f"  {h_prob:>5.1f}{legal_total:>7}{track_total:>9}{rejected_legal:>16}"
              f"{restorable:>12}{share:>8.2f}")
        rows.append({"h_prob": h_prob, "legal": legal_total, "tracker_safe": track_total,
                     "rejected_but_legal": rejected_legal, "restorable": restorable,
                     "restorable_share": share})
    print("=" * 90)
    print("  'restorable' = the first breaking H has EVEN parity of later H's on that")
    print("  qubit, i.e. the correction returns to Z-type (Z -> X -> Z) at the end.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
