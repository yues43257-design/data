#!/usr/bin/env python3
"""⑤ FROZEN RERUN -- one script, one run, every assertion on.

Three things have to agree *at the same time* for the tool paper's claim:

  A. TIGHTNESS, under the DECLARED criterion, **stratified by H spacing**
     (§5.26 / §5.28).  Pooled numbers hide the structure: every miss sits in
     the densest tier.
  B. ADVERSARIAL REGRESSION, 3456 masks (§5.23's 2016 + §5.31's 1440):
     tracker-accepts but the declared definition says illegal?  Must be 0.
  C. SUPERSET CHECK (§5.46 / §5.47): on every accepted mask the residual's
     true support must be covered by the propagated support, plus the four
     one-rule-deleted negative controls must each FAIL.

The point of putting them in one script is that they share one **frozen
definition**, one **frozen grid**, one **version string** and one exit code.
A green A with a stale B is exactly the failure mode this is meant to make
impossible.

Freeze mechanism
----------------
``FROZEN_RERUN_VERSION`` names the grid; ``definition_fingerprint()`` hashes the
*source text* of every file that encodes the declared definition or the
ground truth (criterion, tracker, dense kernel, setup, corpus grid).  If anyone
edits a definition, the printed fingerprint changes and the old numbers no
longer attach to the new code.  Both are written into the JSON.

Expected values are ASSERTED, not printed-and-ignored: a rerun that silently
produces different counts is a FAILED rerun, even if each individual number
still looks plausible.

Usage
-----
    .venv/bin/python frozen_verification.py             # full (~30 min)
    .venv/bin/python frozen_verification.py --quick     # A+B + N<=10 corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

FROZEN_RERUN_VERSION = "frozen-rerun-v1"

# ---------------------------------------------------------------------------
# frozen grids -- these constants ARE the pre-registration
# ---------------------------------------------------------------------------

ADV_SETS: Dict[str, Dict[str, Any]] = {
    # §5.23
    "5.23": {"widths": (4, 5, 6), "blocks": (3, 4, 5),
             "h_probs": (0.3, 0.7, 1.0), "seeds": 4, "masks": 2016,
             "accepted": 230, "legal": 230},
    # §5.31 -- width 4 => every block lands on the same register (max overlap)
    "5.31": {"widths": (4,), "blocks": (3, 4, 5, 6),
             "h_probs": (0.5, 1.0), "seeds": 6, "masks": 1440,
             "accepted": 106, "legal": 106},
}
ADV_TOTAL_MASKS = 3456
ADV_EXACT_LIMIT = 11          # same cutoff as adversarial_tracker_search.py

# §5.26 / §5.28
CONS_EXPECT = {"instances": 26, "sites": 182, "legal": 74, "accepted": 72,
               "hit": 72, "miss": 2, "unsound": 0}
CONS_TIER_EXPECT = {          # H spacing (in ccx) -> (instances, sites, legal, miss)
    0: (2, 14, 14, 0),
    1: (4, 28, 6, 2),         # the only tier with misses: 2/6
    2: (4, 28, 8, 0),
    3: (4, 28, 10, 0),
    4: (4, 28, 12, 0),
    5: (4, 28, 8, 0),
    6: (4, 28, 16, 0),
}

# §5.47 -- two frozen modes, each with its own pre-registered totals
SUPERSET_EXPECT_FULL = {"instances": 678, "accepted": 2278,
                        "variants": {"drop ancilla-merge": 509, "drop cx-push": 63,
                                     "drop block-conjugation": 49,
                                     "drop mask-injection": 819}}
SUPERSET_EXPECT_QUICK = {"instances": 648, "accepted": 2214,
                         "variants": {"drop ancilla-merge": 482, "drop cx-push": 63,
                                      "drop block-conjugation": 47,
                                      "drop mask-injection": 785}}


# ---------------------------------------------------------------------------
# freeze / checks bookkeeping
# ---------------------------------------------------------------------------

DEFINITION_FILES = ("relative_phase_criterion.py", "support_tracker.py",
                    "fast_residual.py", "experiment_setup.py",
                    "superset_corpus.py")


def definition_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in DEFINITION_FILES:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    # the frozen grid itself is part of the definition of the artifact
    digest.update(json.dumps([ADV_SETS, CONS_EXPECT, CONS_TIER_EXPECT,
                              SUPERSET_EXPECT_FULL, SUPERSET_EXPECT_QUICK],
                             sort_keys=True).encode())
    return digest.hexdigest()[:16]


class Checks:
    """Collect assertions; never stop early, so one run reports everything."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:<52} {detail}")
        return bool(ok)

    def eq(self, name: str, got: Any, want: Any) -> bool:
        return self.add(name, got == want, f"got={got} want={want}")

    @property
    def ok(self) -> bool:
        return all(row["ok"] for row in self.rows)


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float, float]:
    if not total:
        return float("nan"), float("nan"), float("nan")
    centre = (successes + z * z / 2) / (total + z * z)
    half = z * math.sqrt(centre * (1 - centre) / (total + z * z))
    return successes / total, max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# A. tightness, stratified by H spacing
# ---------------------------------------------------------------------------

def part_a(checks: Checks) -> Dict[str, Any]:
    from qiskit.circuit.library import CDKMRippleCarryAdder

    import realworld_benchmark as RB
    import realcircuit_window as RCW
    import relative_phase_criterion as RPC
    import support_tracker as ST
    import wide_h_window as WH

    print("\n" + "=" * 104)
    print("  A. TIGHTNESS under the declared criterion, stratified by H spacing")
    print("     LEGAL <=> C = U_cand . U_ideal^dagger diagonal with unit modulus")
    print("=" * 104)
    tiers: Dict[int, Dict[str, int]] = {k: {"instances": 0, "sites": 0, "legal": 0,
                                            "tracker": 0, "hit": 0, "miss": 0,
                                            "unsound": 0}
                                        for k in CONS_TIER_EXPECT}
    per_variant: List[Dict[str, Any]] = []
    for n in (3, 4):
        canonical = RB.to_canonical(CDKMRippleCarryAdder(n))
        for every in (0, 1, 2, 3, 4, 5, 6):
            for h_qubits in (canonical.num_qubits, max(2, canonical.num_qubits // 2)):
                if every == 0 and h_qubits != canonical.num_qubits:
                    continue
                circuit = (WH.interleave_h_layers(canonical, every, h_qubits)
                           if every else canonical)
                schedule = RB._tracker_schedule(circuit)
                positions = [i for i, ins in enumerate(circuit.data)
                             if ins.operation.name == "ccx"]
                legal = tracker = hit = miss = unsound = 0
                for site, position in enumerate(positions):
                    mask = 1 << site
                    accepted = bool(ST.track_safe(schedule, mask))
                    is_legal = bool(RPC.relative_phase_ok(
                        circuit, RCW.replace_one_ccx(circuit, position)))
                    tracker += accepted
                    legal += is_legal
                    hit += int(accepted and is_legal)
                    miss += int((not accepted) and is_legal)
                    unsound += int(accepted and not is_legal)
                tag = ("H-free" if every == 0 else
                       f"H/{every}ccx-{'full' if h_qubits == canonical.num_qubits else 'half'}")
                per_variant.append({"variant": f"adder{n}-{tag}", "tier": every,
                                    "sites": len(positions), "legal": legal,
                                    "tracker": tracker, "hit": hit, "miss": miss,
                                    "unsound": unsound})
                cell = tiers[every]
                cell["instances"] += 1
                cell["sites"] += len(positions)
                cell["legal"] += legal
                cell["tracker"] += tracker
                cell["hit"] += hit
                cell["miss"] += miss
                cell["unsound"] += unsound

    print(f"  {'H spacing':>10}{'inst':>6}{'sites':>7}{'legal':>7}{'track':>7}"
          f"{'HIT':>6}{'MISS':>6}{'UNSND':>7}{'conserv':>9}   95% Wilson")
    pooled = {k: sum(c[k] for c in tiers.values())
              for k in ("instances", "sites", "legal", "tracker", "hit", "miss",
                        "unsound")}
    for every in sorted(tiers):
        cell = tiers[every]
        conserv, lo, hi = wilson(cell["miss"], cell["legal"])
        label = "none" if every == 0 else f"every {every}"
        print(f"  {label:>10}{cell['instances']:>6}{cell['sites']:>7}"
              f"{cell['legal']:>7}{cell['tracker']:>7}{cell['hit']:>6}"
              f"{cell['miss']:>6}{cell['unsound']:>7}{conserv:>9.3f}   "
              f"[{lo:.3f}, {hi:.3f}]")
    pooled_conserv, pooled_lo, pooled_hi = wilson(pooled["miss"], pooled["legal"])
    print(f"  {'POOLED':>10}{pooled['instances']:>6}{pooled['sites']:>7}"
          f"{pooled['legal']:>7}{pooled['tracker']:>7}{pooled['hit']:>6}"
          f"{pooled['miss']:>6}{pooled['unsound']:>7}{pooled_conserv:>9.3f}   "
          f"[{pooled_lo:.3f}, {pooled_hi:.3f}]")

    print("\n  assertions:")
    checks.eq("A.1 instance count", pooled["instances"], CONS_EXPECT["instances"])
    checks.eq("A.2 site count", pooled["sites"], CONS_EXPECT["sites"])
    checks.eq("A.3 legal count", pooled["legal"], CONS_EXPECT["legal"])
    checks.eq("A.4 tracker-accepted count", pooled["tracker"], CONS_EXPECT["accepted"])
    checks.eq("A.5 HIT count", pooled["hit"], CONS_EXPECT["hit"])
    checks.eq("A.6 MISS count", pooled["miss"], CONS_EXPECT["miss"])
    checks.eq("A.7 UNSOUND count (must be 0)", pooled["unsound"], 0)
    checks.add("A.8 count alignment legal+illegal == sites",
               pooled["legal"] + (pooled["sites"] - pooled["legal"]) == pooled["sites"],
               f"{pooled['legal']} + {pooled['sites'] - pooled['legal']}")
    checks.add("A.9 count alignment accepted == HIT + UNSOUND",
               pooled["tracker"] == pooled["hit"] + pooled["unsound"],
               f"{pooled['tracker']} == {pooled['hit']} + {pooled['unsound']}")
    checks.add("A.10 pooled conservativeness == 2/74",
               abs(pooled_conserv - 2 / 74) < 1e-12, f"{pooled_conserv:.6f}")
    for every, (inst, sites, legal, miss) in CONS_TIER_EXPECT.items():
        cell = tiers[every]
        checks.add(f"A.11 tier {every}: misses only where pre-registered",
                   cell["miss"] == miss, f"miss={cell['miss']} want={miss}")
        checks.add(f"A.12 tier {every}: instances/sites/legal",
                   (cell["instances"], cell["sites"], cell["legal"]) == (inst, sites, legal),
                   f"{(cell['instances'], cell['sites'], cell['legal'])} want {(inst, sites, legal)}")
    others = [t for t in tiers if t != 1]
    checks.add("A.13 all misses confined to the densest H tier",
               all(tiers[t]["miss"] == 0 for t in others)
               and tiers[1]["miss"] == CONS_EXPECT["miss"],
               f"tier1={tiers[1]['miss']} others={ {t: tiers[t]['miss'] for t in others} }")
    checks.add("A.14 densest tier conservativeness == 2/6",
               abs(wilson(tiers[1]["miss"], tiers[1]["legal"])[0] - 2 / 6) < 1e-12,
               f"{wilson(tiers[1]['miss'], tiers[1]['legal'])[0]:.4f}")
    return {"tiers": {str(k): v for k, v in tiers.items()}, "pooled": pooled,
            "pooled_conservativeness": pooled_conserv,
            "pooled_ci": [pooled_lo, pooled_hi],
            "variants": per_variant}


# ---------------------------------------------------------------------------
# B. 3456-mask adversarial regression
# ---------------------------------------------------------------------------

def part_b(checks: Checks) -> Dict[str, Any]:
    import relative_phase_criterion as RPC
    import support_tracker as ST
    import ultimate_scalability_test as ULT
    import wide_scale_stress as WIDE
    import experiment_setup as SETUP

    print("\n" + "=" * 104)
    print("  B. ADVERSARIAL REGRESSION: 3456 masks (§5.23's 2016 + §5.31's 1440)")
    print("     question: tracker ACCEPTS but the declared definition says ILLEGAL?")
    print("=" * 104)
    out: Dict[str, Any] = {"sets": {}, "counterexamples": []}
    total_masks = total_accepted = total_legal = total_counter = 0
    for set_name, spec in ADV_SETS.items():
        masks = accepted = legal = counter = 0
        for width in spec["widths"]:
            for width_blocks in spec["blocks"]:
                for h_prob in spec["h_probs"]:
                    for seed_index in range(spec["seeds"]):
                        seed = (SETUP.seed_for(seed_index) * 1000
                                + width * 10 + width_blocks)
                        inst = ULT.WideInstance(width=width, blocks=width_blocks,
                                                seed=seed, h_prob=h_prob,
                                                ancilla_pool=3)
                        if inst.total > ADV_EXACT_LIMIT:
                            continue
                        schedule = WIDE.build_schedule(inst)
                        ancillas = list(range(width, inst.total))
                        base = inst.circuit(0)
                        for mask in range(1 << width_blocks):
                            masks += 1
                            if not ST.track_safe(schedule, mask):
                                continue
                            accepted += 1
                            ok = bool(RPC.relative_phase_ok(
                                base, inst.circuit(mask), ancillas))
                            legal += int(ok)
                            if not ok:
                                counter += 1
                                if len(out["counterexamples"]) < 20:
                                    out["counterexamples"].append(
                                        {"set": set_name, "width": width,
                                         "blocks": width_blocks, "h_prob": h_prob,
                                         "seed": seed, "mask": mask})
        print(f"  §{set_name}: masks={masks} accepted={accepted} legal={legal} "
              f"COUNTEREXAMPLES={counter}")
        out["sets"][set_name] = {"masks": masks, "accepted": accepted,
                                 "legal": legal, "counterexamples": counter}
        total_masks += masks
        total_accepted += accepted
        total_legal += legal
        total_counter += counter
    print(f"  TOTAL: masks={total_masks} accepted={total_accepted} "
          f"legal={total_legal} COUNTEREXAMPLES={total_counter}")

    print("\n  assertions:")
    for set_name, spec in ADV_SETS.items():
        got = out["sets"][set_name]
        checks.eq(f"B.{set_name} mask count", got["masks"], spec["masks"])
        checks.eq(f"B.{set_name} accepted count", got["accepted"], spec["accepted"])
        checks.eq(f"B.{set_name} legal count", got["legal"], spec["legal"])
        checks.eq(f"B.{set_name} counterexamples", got["counterexamples"], 0)
        checks.add(f"B.{set_name} accepted == legal",
                   got["accepted"] == got["legal"], f"{got['accepted']} == {got['legal']}")
    checks.eq("B.total mask count (frozen 3456)", total_masks, ADV_TOTAL_MASKS)
    checks.eq("B.total counterexamples", total_counter, 0)
    checks.add("B.total accepted == legal", total_accepted == total_legal,
               f"{total_accepted} == {total_legal}")
    out.update({"total_masks": total_masks, "total_accepted": total_accepted,
                "total_legal": total_legal, "total_counterexamples": total_counter})
    return out


# ---------------------------------------------------------------------------
# C. superset property + one-rule-deleted negative controls
# ---------------------------------------------------------------------------

def part_c(checks: Checks, phases: Sequence[str], workers: Dict[str, int],
           sink: Path, expect: Dict[str, Any]) -> Dict[str, Any]:
    import superset_corpus as SC

    print("\n" + "=" * 104)
    print("  C. SUPERSET CHECK + negative controls (tracker support covers the truth?)")
    print("=" * 104)
    rows: List[Dict[str, Any]] = []
    for index, phase in enumerate(phases):
        result = SC.run_phase(phase, workers[phase], sink, truncate=(index == 0))
        rows.extend(result["rows"])
    checked = sum(r["checked"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    variants = {name: sum(r["variant_violations"][name] for r in rows)
                for name in SC.VARIANTS}
    undefined = sum(r["undefined"] for r in rows)
    max_off = max(r["off_diagonal_max"] for r in rows)
    min_diag = min(r["diag_min_abs"] for r in rows if r["diag_min_abs"] is not None)

    print("\n  assertions:")
    checks.eq("C.1 instance count", len(rows), expect["instances"])
    checks.eq("C.2 accepted masks", accepted, expect["accepted"])
    checks.eq("C.3 checked masks == accepted masks", checked, accepted)
    checks.eq("C.4 undefined residual support", undefined, 0)
    checks.eq("C.5 shipped rules: violations", variants["full (shipped rules)"], 0)
    for name, want in expect["variants"].items():
        checks.eq(f"C.6 negative control '{name}'", variants[name], want)
    checks.add("C.7 every rule deletion is discriminative",
               all(variants[name] > 0 for name in expect["variants"]),
               {k: v for k, v in variants.items() if not k.startswith("full")})
    checks.add("C.8 accepted mask residuals are diagonal", max_off < 1e-9,
               f"max off-diagonal={max_off:.2e}")
    checks.add("C.9 accepted mask residuals have unit modulus", abs(min_diag - 1) < 1e-9,
               f"min |diag|={min_diag:.15f}")
    tracker_rc = SC.check_tracker_equivalence(phases)
    checks.eq("C.10 local rule set == shipped DiagonalTracker", tracker_rc, 0)
    return {"mode": "quick" if len(phases) == 2 else "full",
            "phases": list(phases), "instances": len(rows), "accepted": accepted,
            "checked": checked, "variant_violations": variants,
            "undefined": undefined, "max_off_diagonal": max_off,
            "min_diag_abs": min_diag, "rows": rows}


# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the N=11/N=12 superset phases")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "frozen_verification.json"))
    args = parser.parse_args(argv)

    import experiment_setup as SETUP

    fingerprint = definition_fingerprint()
    started = time.perf_counter()
    checks = Checks()
    print("=" * 104)
    mode = "quick" if args.quick else "full"
    print(f"  FROZEN RERUN  version={FROZEN_RERUN_VERSION}  mode={mode}  "
          f"setup={SETUP.SETUP_VERSION}  definition={fingerprint}")
    print(f"  python={sys.version.split()[0]}  root={ROOT}")
    print("=" * 104)

    report: Dict[str, Any] = {
        "frozen_rerun_version": FROZEN_RERUN_VERSION,
        "mode": mode,
        "setup_version": SETUP.SETUP_VERSION,
        "definition_fingerprint": fingerprint,
        "python": sys.version.split()[0],
    }
    report["part_a"] = part_a(checks)
    report["part_b"] = part_b(checks)
    phases = ("small", "n10") if args.quick else ("small", "n10", "n11", "n12")
    expect = SUPERSET_EXPECT_QUICK if args.quick else SUPERSET_EXPECT_FULL
    workers = {"small": 5, "n10": 4, "n11": 3, "n12": 2}
    sink = ROOT / "reports" / "figure_data" / "frozen_superset.jsonl"
    report["part_c"] = part_c(checks, phases, workers, sink, expect)
    if sink.exists():
        sink.unlink()                      # regenerable; the JSON below has the rows

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 104)
    print(f"  SUMMARY  version={FROZEN_RERUN_VERSION}  mode={mode}  "
          f"definition={fingerprint}")
    failed = [row for row in checks.rows if not row["ok"]]
    print(f"  checks: {len(checks.rows) - len(failed)}/{len(checks.rows)} passed, "
          f"{len(failed)} failed, {elapsed:.0f}s")
    for row in failed:
        print(f"    FAILED: {row['check']}  {row['detail']}")
    print(f"  VERDICT: {'PASS' if checks.ok else 'FAIL'}")
    print("=" * 104)

    report["checks"] = checks.rows
    report["passed"] = len(checks.rows) - len(failed)
    report["failed"] = len(failed)
    report["verdict"] = "PASS" if checks.ok else "FAIL"
    report["seconds"] = round(elapsed, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0 if checks.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
