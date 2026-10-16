#!/usr/bin/env python3
"""Step 2: declare the truth definition, then adversarially try to falsify the
tracker *implementation*.

DECLARED TRUTH DEFINITION (modelling decision, frozen here)
-----------------------------------------------------------
  * Input domain: the **Toffoli layer**, structure preserving
    (``{C^nX/MCX, CNOT, X, Z, S, Pauli, H}``; Toffoli not yet decomposed).
  * Ancillas are **post-selected clean** (they are uncomputed back to |0> and
    may be reused), so legality is judged on the **clean-ancilla subspace**.
  * LEGAL(candidate | ideal)  <=>  ``C = U_cand . U_ideal^dagger`` restricted to
    that subspace is **diagonal with unit-modulus entries**  <=>
    the Z-basis measurement statistics of the data register are unchanged.
    Criterion: ``relative_phase_criterion.relative_phase_ok``.

WHAT THIS SEARCH CAN AND CANNOT DO
----------------------------------
* The soundness **lemma** is a theorem (a diagonal operator conjugated by a
  basis permutation stays diagonal; diagonal gates commute with it; only H-class
  gates break it).  A search cannot falsify a theorem.
* What CAN be falsified is the **implementation**: the support propagation must
  be a *sound over-approximation* of the residual's support.  If we find a mask
  with ``tracker_safe = True`` but ``LEGAL = False``, the propagation rules are
  wrong and the lemma's application is broken.

Usage
-----
    .venv/bin/python adversarial_tracker_search.py
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


ULT = _load("adv_ult", "ultimate_scalability_test.py")
WIDE = _load("adv_wide", "wide_scale_stress.py")
ST = _load("adv_tracker", "support_tracker.py")
RPC = _load("adv_criterion", "relative_phase_criterion.py")
SETUP = _load("adv_setup", "experiment_setup.py")

EXACT_LIMIT = 11


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="adversarial tracker search.")
    parser.add_argument("--widths", default="4,5,6")
    parser.add_argument("--blocks", default="3,4,5")
    parser.add_argument("--h-probs", default="0.3,0.7,1.0")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "adversarial_tracker_search.json"))
    args = parser.parse_args(argv)
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    blocks_list = [int(b) for b in args.blocks.split(",") if b.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]

    print("=" * 92)
    print("  adversarial search: tracker-accepts but DECLARED-DEFINITION-illegal?")
    print("  definition = C = U_cand . U_ideal^dag diagonal on the clean-ancilla subspace")
    print("=" * 92)
    print(f"  {'N':>4}{'blocks':>7}{'h':>5}{'inst':>6}{'masks':>7}{'track':>7}"
          f"{'legal':>7}{'COUNTEREX':>11}{'conserv':>9}")

    total_masks = total_track = total_legal = total_counter = 0
    counterexamples: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for width in widths:
        for blocks in blocks_list:
            for h_prob in h_probs:
                track = legal = counter = masks = 0
                for seed_index in range(args.seeds):
                    seed = SETUP.seed_for(seed_index) * 1000 + width * 10 + blocks
                    inst = ULT.WideInstance(width=width, blocks=blocks, seed=seed,
                                            h_prob=h_prob, ancilla_pool=3)
                    if inst.total > EXACT_LIMIT:
                        continue
                    schedule = WIDE.build_schedule(inst)
                    ancillas = list(range(width, inst.total))
                    base = inst.circuit(0)
                    for mask in range(1 << blocks):
                        masks += 1
                        accepted = bool(ST.track_safe(schedule, mask))
                        if not accepted:
                            continue
                        track += 1
                        ok = RPC.relative_phase_ok(base, inst.circuit(mask), ancillas)
                        legal += int(bool(ok))
                        if not ok:
                            counter += 1
                            counterexamples.append(
                                {"width": width, "blocks": blocks, "h_prob": h_prob,
                                 "seed": seed, "mask": mask})
                if not masks:
                    continue
                conserv = (1 - legal / track) if track else float("nan")
                total_masks += masks
                total_track += track
                total_legal += legal
                total_counter += counter
                print(f"  {width:>4}{blocks:>7}{h_prob:>5.1f}{args.seeds:>6}"
                      f"{masks:>7}{track:>7}{legal:>7}{counter:>11}{conserv:>9.2f}")
                rows.append({"N_total": width, "blocks": blocks, "h_prob": h_prob,
                             "masks": masks, "tracker_accept": track,
                             "legal": legal, "counterexamples": counter,
                             "conservativeness": conserv})
    print("=" * 92)
    print(f"  TOTAL: masks={total_masks}  tracker-accepted={total_track}  "
          f"legal={total_legal}  COUNTEREXAMPLES={total_counter}")
    if total_counter == 0:
        print("  -> no implementation counterexample found in the searched space")
    else:
        print("  -> IMPLEMENTATION BROKEN: the support propagation is not a sound "
              "over-approximation (examples saved)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"declared_definition": "C = U_cand U_ideal^dagger diagonal on the "
                                "clean-ancilla subspace (ancillas post-selected)",
         "setup_version": SETUP.SETUP_VERSION,
         "script": "adversarial_tracker_search.py",
         "total_masks": total_masks, "tracker_accepted": total_track,
         "legal": total_legal, "counterexamples_count": total_counter,
         "counterexamples": counterexamples[:50], "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"  wrote {out}")
    return 1 if total_counter else 0


if __name__ == "__main__":
    raise SystemExit(main())
