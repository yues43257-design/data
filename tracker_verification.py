#!/usr/bin/env python3
"""Bidirectional verification of ``support_tracker``: precision **and** recall.

Two predicates on a downgrade mask:

``legal(mask)``
    the *exact* reduced verifier: the Z-basis measurement statistics of the
    downgraded circuit equal those of the exact one (the definition of "safe").
``track(mask)``
    ``support_tracker``: the accumulated correction stays Z-**diagonal**.

Theory (the completeness draft):

* ``track => legal``.  A diagonal correction ``C`` gives ``U_RL = U_exact C``, so
  every Z-basis probability is unchanged.  This direction is *provable*; the only
  part needing empirical confirmation is the per-gate conjugation bookkeeping
  (H / CNOT / block rules).  Violations of ``track & ~legal`` therefore indicate a
  **bug in the rule set**, not a modelling choice.
* ``legal => track`` is **false by construction** -- diagonality is sufficient,
  not necessary.  The tracker is *incomplete*, and
  ``recall = |legal & track| / |legal|`` is exactly how conservative it is.

Why this matters: the planned "H window" experiment counts, per Toffoli site,
how often the rewrite is legal.  If that count is produced by the tracker, it
measures **tracker conservativeness**, not the true legal rate -- unless recall
is measured first and reported as a bound.

So we measure, *inside the width where the exact verifier still runs*
(``d = 2**(n_max+1)``):

* ``precision``: ``|track & ~legal|``            -- must be 0
* ``recall``   : ``|legal & track| / |legal|``   -- stratified by H density
* per-site recall: same, restricted to 1-bit masks (single downgraded block),
  which is the statistic the H-window experiment actually needs.

Full enumeration over all ``2**blocks`` masks is used (no biased sampling), which
also removes the "natural mask distribution" objection.

Usage
-----
    .venv/bin/python tracker_verification.py
    .venv/bin/python tracker_verification.py --blocks 7 --seeds 4 --json /tmp/tv.json
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


ST = _load("tv_tracker", "support_tracker.py")
SYM = _load("tv_symbolic", "symbolic_verifier.py")


def verify_instance(num_controls: int, blocks: int, seed: int, h_prob: float,
                    tolerance: float = 1e-9) -> Dict[str, Any]:
    """Enumerate every mask of one instance and classify it two ways."""
    verifier = SYM.ReducedSymbolicVerifier(num_controls, blocks, seed, h_prob,
                                           tolerance=tolerance)
    schedule = ST.homogeneous_schedule(num_controls, blocks, seed, h_prob)

    total = exact_safe = track_safe = both = 0
    unsound = 0                       # track but not legal  (precision failure)
    missed = 0                        # legal but not track  (recall loss)
    site_total = site_legal = site_track = 0
    site_missed = 0

    for mask in range(1 << blocks):
        legal = bool(verifier.is_safe(mask))
        track = bool(ST.track_safe(schedule, mask))
        total += 1
        exact_safe += int(legal)
        track_safe += int(track)
        both += int(legal and track)
        unsound += int(track and not legal)
        missed += int(legal and not track)
        if bin(mask).count("1") == 1:      # a single downgraded block
            site_total += 1
            site_legal += int(legal)
            site_track += int(track)
            site_missed += int(legal and not track)

    return {
        "num_controls": num_controls, "blocks": blocks, "seed": seed,
        "h_prob": h_prob,
        "masks": total, "exact_safe": exact_safe, "track_safe": track_safe,
        "both": both, "unsound": unsound, "missed": missed,
        "site_masks": site_total, "site_legal": site_legal,
        "site_track": site_track, "site_missed": site_missed,
    }


def _rate(a: int, b: int) -> Optional[float]:
    return (a / b) if b else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="tracker precision/recall audit.")
    parser.add_argument("--controls", default="2,3,4,5")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--h-probs", default="0.0,0.1,0.3,0.7,1.0")
    parser.add_argument("--json", default=str(ROOT / "reports" / "figure_data" /
                                              "tracker_verification.json"))
    args = parser.parse_args(argv)

    controls = [int(c) for c in args.controls.split(",") if c.strip()]
    h_probs = [float(h) for h in args.h_probs.split(",") if h.strip()]

    rows: List[Dict[str, Any]] = []
    for h_prob in h_probs:
        for num_controls in controls:
            for seed in range(args.seeds):
                rows.append(verify_instance(num_controls, args.blocks,
                                            seed * 7 + 11, h_prob))

    # ---- pooled per h_prob ------------------------------------------------
    print("=" * 96)
    print("  support_tracker audit (exact reduced verifier available at every point)")
    print("=" * 96)
    print(f"  blocks={args.blocks}  controls={controls}  seeds={args.seeds}  "
          f"masks/instance={1 << args.blocks}")
    print()
    header = (f"  {'h_prob':>6}{'masks':>7}{'legal':>7}{'track':>7}{'UNSOUND':>9}"
              f"{'recall':>8}{'site legal':>11}{'site recall':>12}")
    print(header)
    print("  " + "-" * 88)

    summary: Dict[str, Any] = {}
    total_unsound = 0
    for h_prob in h_probs:
        group = [r for r in rows if r["h_prob"] == h_prob]
        masks = sum(r["masks"] for r in group)
        legal = sum(r["exact_safe"] for r in group)
        track = sum(r["track_safe"] for r in group)
        unsound = sum(r["unsound"] for r in group)
        both = sum(r["both"] for r in group)
        s_missed = sum(r["site_missed"] for r in group)
        s_legal_total = sum(r["site_legal"] for r in group)
        total_unsound += unsound
        recall = _rate(both, legal)
        site_recall = _rate(s_legal_total - s_missed, s_legal_total)
        print(f"  {h_prob:>6.1f}{masks:>7}{legal:>7}{track:>7}{unsound:>9}"
              f"{(f'{recall:.3f}' if recall is not None else 'n/a'):>8}"
              f"{s_legal_total:>11}"
              f"{(f'{site_recall:.3f}' if site_recall is not None else 'n/a'):>12}")
        summary[f"h{h_prob:g}"] = {
            "masks": masks, "legal": legal, "track_safe": track,
            "unsound": unsound, "recall": recall,
            "site_legal": s_legal_total, "site_recall": site_recall,
        }

    print("  " + "-" * 88)
    print("  UNSOUND = tracker accepted a mask the exact verifier rejects "
          "(a rule-set bug if > 0)")
    print("  recall  = |legal & track| / |legal|   (tracker conservativeness; "
          "lower = more conservative)")
    print(f"  total UNSOUND over {len(rows)} instances = {total_unsound}")
    print("=" * 96)

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 1 if total_unsound else 0


if __name__ == "__main__":
    raise SystemExit(main())
