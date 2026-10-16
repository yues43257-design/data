#!/usr/bin/env python3
"""b2 probe (3-day plan, compressed): can a CLIFFORD-ONLY H migration unlock sites?

b2 = move Hadamards so that relative-phase downgrade sites become legal, with the
legality certificate (support avoids H) as the scorer.

Judgment 0 for this probe is *structural*: an H migration is only FREE if the
Hadamard can be commuted back past **Clifford gates only** (H / CNOT).  As soon as
a T / Tdg / Toffoli sits between the site and the blocking Hadamard, the migration
needs re-synthesis and costs T.

So the probe answers, before any implementation:

    for every tracker-REJECTED site, is its blocking Hadamard separated from the
    site by Clifford gates only?

    C_count  = sites that CAN be freed for free  (=> Delta N_useful upper bound)
    UB       = 3 * (#legal&unpaired sites after a perfect free migration)
    cost     = 0 by construction for free slides
    net      = 3 * C_count

Pre-registered GO / KILL (run once):
    KILL  if  3*C_count  <= 10% of UB   or  C_count does not grow with n
    GO    if  3*C_count  >= 30% of UB   and it grows with n

Usage
-----
    .venv/bin/python probe_b2_hmigration.py
"""

from __future__ import annotations

import collections
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from qiskit.circuit.library import CDKMRippleCarryAdder

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / filename))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


RB = _load("pb2_rb", "realworld_benchmark.py")
WH = _load("pb2_wh", "wide_h_window.py")
ST = _load("pb2_st", "support_tracker.py")

CLIFFORD = frozenset({"h", "cx", "swap", "x", "z", "s", "sdg", "id"})


def analyse(schedule: Dict[str, Any]):
    """Return per-site (tracker_safe, free_unlockable)."""
    out: List[Dict[str, Any]] = []
    for site in range(len(schedule["blocks"])):
        support: Set[int] = set()
        blocker_layer = None
        blocker_q = None
        for index, block in enumerate(schedule["blocks"]):
            for name, qubits in schedule["layers"][index]:
                if name == "h" and int(qubits[0]) in support and blocker_layer is None:
                    blocker_layer, blocker_q = index, int(qubits[0])
                elif name == "cx" and int(qubits[1]) in support:
                    support.add(int(qubits[0]))
            if block["target"] in support:
                support.update(int(q) for q in block["controls"])
                support.update(int(q) for q in block["ancillas"])
            if site == index:                     # the mask is exactly {site}
                support.add(int(block["target"]))
                support.update(int(q) for q in block["controls"])
                support.update(int(q) for q in block["ancillas"])
        safe = blocker_layer is None
        free = False
        if not safe:
            # any non-Clifford gate between the site and the blocking H?
            blocks_between = any(
                (schedule["blocks"][i].get("controls") or
                 schedule["blocks"][i].get("target") is not None)
                for i in range(site + 1, blocker_layer + 1)
            )
            noncliff = any(
                name not in CLIFFORD
                for i in range(site, blocker_layer + 1)
                for name, _ in schedule["layers"][i]
            )
            free = (not blocks_between) and (not noncliff)
        out.append({"site": site, "safe": safe, "free_unlockable": free})
    return out


def main() -> int:
    print("=" * 96)
    print("  b2 probe: are blocking Hadamards reachable by a FREE (Clifford-only) slide?")
    print("=" * 96)
    print(f"  {'variant':<28}{'sites':>6}{'safe':>6}{'rejected':>9}"
          f"{'free-unlock':>12}{'UB(3x)':>8}{'net(3xC)':>9}{'net/UB':>8}")
    rows: List[Dict[str, Any]] = []
    for n in (3, 4, 6, 8):
        canon = RB.to_canonical(CDKMRippleCarryAdder(n))
        for every in (1, 2, 4):
            circuit = WH.interleave_h_layers(canon, every, canon.num_qubits)
            schedule = RB._tracker_schedule(circuit)
            info = analyse(schedule)
            sites = len(info)
            safe = sum(1 for d in info if d["safe"])
            rejected = sites - safe
            free = sum(1 for d in info if d["free_unlockable"])
            ub = 3 * sites          # perfect migration ⇒ every site legal
            net = 3 * free
            ratio = net / ub if ub else float("nan")
            label = f"adder{n}-H/{every}ccx"
            print(f"  {label:<28}{sites:>6}{safe:>6}{rejected:>9}{free:>12}"
                  f"{ub:>8}{net:>9}{ratio:>8.2f}")
            rows.append({"variant": label, "n": n, "sites": sites, "safe": safe,
                         "rejected": rejected, "free_unlockable": free,
                         "UB": ub, "net": net, "net_over_UB": ratio})
    print("=" * 96)
    tot_ub = sum(r["UB"] for r in rows)
    tot_net = sum(r["net"] for r in rows)
    print(f"  pooled: UB={tot_ub} T   net={tot_net} T   net/UB={tot_net / tot_ub:.2f}")
    verdict = "GO" if tot_net >= 0.30 * tot_ub else "KILL"
    print(f"  pre-registered verdict (net >= 30% of UB): {verdict}")
    out = ROOT / "reports" / "figure_data" / "probe_b2_hmigration.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
