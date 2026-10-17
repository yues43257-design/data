#!/usr/bin/env python3
"""Day 1-3 probe: can *selecting* exact/relative blocks per adjacent window
beat per-block greedy, and how far can that get us towards PyZX?

Circuit family: the chained CDKM adder (``tcount_scan.two_adders``), n = 3..6.
Its gates are ``{cx, ccx}`` only -- i.e. every gate is a basis permutation.

The decision space probed here is **selection only**:

    each ccx site is either   exact (7 T)  or  relative-phase (4 T)

with no change of topology.  Day 1 enumerates every adjacent *pair* and
*triple* of ccx sites (2**2 and 2**3 assignments inside the window, the rest
held at the greedy choice) and records the exact T-count and exact legality of
the resulting circuit.

Exact legality
--------------
The declared criterion is unchanged: ``C = U_cand . U_ideal^dagger`` must be
diagonal with unit-modulus entries.  Two independent oracles are used:

  * DENSE (n <= 4, and a spot-check at n = 5): build ``C`` from qiskit
    ``Operator`` matrices -- the ground truth.
  * STRUCTURAL (all n): the property below, which is *checked* here rather than
    assumed:

      (i)   the exact block is a permutation matrix (a Toffoli);
      (ii)  relative_block . exact_block^dagger is diagonal and unit-modulus;
      (iii) every other gate in the chained adder is a permutation (cx/ccx).

    A product of permutation and diagonal gates equals ``P . D``: conjugating a
    diagonal matrix by a permutation is still diagonal, and permutation factors
    compose.  The candidate has the *same* permutation part as the ideal, so
    ``C = P . D . P^dagger`` is diagonal.  Hence every mask is legal, at every
    width -- which the dense oracle is used to confirm where it is affordable.

The structural oracle is cross-checked against the dense one on every
enumerated window at n = 3 and n = 4 (and a spot-check at n = 5).

Usage
-----
    .venv/bin/python window_selection_probe.py            # full probe
    .venv/bin/python window_selection_probe.py --quick     # n=3 only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import relative_phase_criterion as RPC          # noqa: E402
import support_tracker as ST                    # noqa: E402
import tcount_scan as SCAN                      # noqa: E402
import wide_h_window as WH                      # noqa: E402
import realcircuit_window as RCW                # noqa: E402

PROBE_VERSION = "window-probe-v1"
EXACT_T = 7
RELATIVE_T = 4
TOL = 1e-9


# ---------------------------------------------------------------------------
# circuit construction
# ---------------------------------------------------------------------------

def tcount(circuit: QuantumCircuit) -> int:
    """Mechanical T-count of the emitted circuit (t + tdg)."""
    return sum(1 for ins in circuit.data if ins.operation.name in ("t", "tdg"))


def ccx_positions(circuit: QuantumCircuit) -> List[int]:
    return [i for i, ins in enumerate(circuit.data) if ins.operation.name == "ccx"]


def _append_block(out: QuantumCircuit, block: QuantumCircuit,
                  qubit_map: Sequence[int]) -> None:
    for instruction in block.data:
        qubits = [qubit_map[block.find_bit(q).index] for q in instruction.qubits]
        out.append(instruction.operation, qubits,
                   [block.find_bit(c).index for c in instruction.clbits])


def replace_positions(circuit: QuantumCircuit, positions: Sequence[int],
                      kind: str = "relative") -> QuantumCircuit:
    """Copy ``circuit`` replacing every listed ccx by an explicit decomposition.

    ``kind="relative"`` -> the 4-T RCCX gadget; ``kind="exact"`` -> the 7-T
    Clifford+T Toffoli.  Both are needed: the canonical chained adder keeps
    ``ccx`` as an atom (T-count 0 literally), so the *decomposed* T-count has to
    be materialised to be counted mechanically.
    """
    import variant_choice_headroom as VCH

    wanted = set(int(p) for p in positions)
    if kind not in ("relative", "exact"):
        raise ValueError(kind)
    out = QuantumCircuit(circuit.num_qubits)
    for position, instruction in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        if position in wanted and instruction.operation.name == "ccx":
            if kind == "relative":
                c0, c1, target = qubits
                for gate in RCW.LM._rccx_gates(c0, c1, target):
                    if gate[0] == "cx":
                        out.cx(gate[1], gate[2])
                    else:
                        getattr(out, gate[0])(gate[1])
            else:
                _append_block(out, VCH.exact_block(2), qubits)
        else:
            out.append(instruction.operation, qubits,
                       [circuit.find_bit(c).index for c in instruction.clbits])
    return out


def mask_to_positions(positions: Sequence[int], mask: int) -> List[int]:
    return [positions[i] for i in range(len(positions)) if (mask >> i) & 1]


def total_t(blocks: int, mask: int) -> int:
    """T-count of the circuit with ``mask`` blocks relative and the rest exact.

    The canonical adder keeps unreplaced ``ccx`` as atoms, so a literal count
    would read 0 for them; this materialises the accounting explicitly.  Both
    ends are cross-checked against mechanically counted circuits.
    """
    upgraded = bin(mask).count("1")
    return EXACT_T * (blocks - upgraded) + RELATIVE_T * upgraded


# ---------------------------------------------------------------------------
# exact legality
# ---------------------------------------------------------------------------

def dense_legal(base_operator: np.ndarray, candidate: QuantumCircuit) -> Tuple[bool, float, float]:
    candidate_operator = np.asarray(Operator(candidate).data, dtype=complex)
    offdiag, unit = RPC.relative_phase_metrics(base_operator, candidate_operator)
    return bool(offdiag < TOL and unit < TOL), float(offdiag), float(unit)


def check_structural_premises() -> Dict[str, Any]:
    """The three facts the structural legality oracle rests on."""
    import variant_choice_headroom as VCH

    exact = np.asarray(Operator(VCH.exact_block(2)).data, dtype=complex)
    relative = np.asarray(Operator(VCH.relative_block(2)).data, dtype=complex)
    toffoli = QuantumCircuit(3)
    toffoli.ccx(0, 1, 2)
    toffoli_matrix = np.asarray(Operator(toffoli).data, dtype=complex)
    is_permutation = bool(np.allclose(np.abs(exact) ** 2, np.abs(exact), atol=1e-12)
                          and np.allclose(np.abs(exact).sum(axis=0), 1.0, atol=1e-12)
                          and np.allclose(np.abs(exact).sum(axis=1), 1.0, atol=1e-12))
    correction = relative @ exact.conj().T
    offdiag = float(np.abs(correction - np.diag(np.diag(correction))).max())
    unit = float(np.abs(np.abs(np.diag(correction)) - 1.0).max())
    return {
        "exact_block_is_permutation": is_permutation,
        "exact_block_equals_ccx": float(np.abs(exact - toffoli_matrix).max()),
        "relative_x_exact_dagger_offdiag": offdiag,
        "relative_x_exact_dagger_unit_dev": unit,
        "exact_block_tcount": tcount(VCH.exact_block(2)),
        "relative_block_tcount": tcount(VCH.relative_block(2)),
    }


# ---------------------------------------------------------------------------
# Day 1: window enumeration
# ---------------------------------------------------------------------------

def enumerate_windows(n: int, dense_budget: int,
                      background_mask: int) -> Dict[str, Any]:
    circuit = SCAN.two_adders(n)
    positions = ccx_positions(circuit)
    m = len(positions)
    assert tcount(circuit) == 0, "canonical chained adder should hold ccx atoms"
    base = replace_positions(circuit, [])
    base_t = tcount(replace_positions(circuit, positions, kind="exact"))
    all_relative = replace_positions(circuit, positions)
    all_relative_t = tcount(all_relative)
    if m == 0:
        raise RuntimeError("no Toffoli sites")
    assert total_t(m, 0) == base_t, (total_t(m, 0), base_t)
    assert total_t(m, (1 << m) - 1) == all_relative_t, (
        total_t(m, (1 << m) - 1), all_relative_t)

    # ---- structural premise: every non-ccx gate is a permutation ----------
    gate_names = sorted({ins.operation.name for ins in circuit.data})
    only_permutation_atoms = set(gate_names) <= {"cx", "ccx"}

    # ---- dense ground truth where affordable -------------------------------
    wants_dense = (n <= 4) or (n == 5 and dense_budget > 0)
    base_op = np.asarray(Operator(base).data, dtype=complex) if wants_dense else None
    dense_checks = 0
    dense_disagreements: List[Dict[str, Any]] = []

    windows: List[Dict[str, Any]] = []
    for width in (2, 3):
        for start in range(0, max(0, m - width + 1)):
            sites = list(range(start, start + width))
            window_bits = ((1 << width) - 1) << start
            rows: List[Dict[str, Any]] = []
            for sub in range(1 << width):
                local_mask = sub << start
                # the probe's real question: re-choose THIS window while the
                # rest of the circuit keeps the per-block greedy choice
                joint_mask = (background_mask & ~window_bits) | local_mask
                candidate = replace_positions(
                    circuit, mask_to_positions(positions, local_mask))
                legal_by_lemma = only_permutation_atoms
                step = EXACT_T - RELATIVE_T
                row: Dict[str, Any] = {
                    "window": [sites[0], sites[-1]],
                    "width": width,
                    "submask": sub,
                    "local_mask": local_mask,
                    "joint_mask": joint_mask,
                    "local_upgrades": bin(local_mask).count("1"),
                    "joint_upgrades": bin(joint_mask).count("1"),
                    "local_tcount": total_t(m, local_mask),
                    "joint_tcount": total_t(m, joint_mask),
                    "joint_saving": step * bin(joint_mask).count("1"),
                    "legal_lemma": legal_by_lemma,
                }
                do_dense = (n <= 4) or (n == 5 and dense_checks < dense_budget)
                if do_dense:
                    legal, offdiag, unit = dense_legal(base_op, candidate)
                    row.update({"legal_dense": legal, "offdiag": offdiag,
                                "unit_dev": unit})
                    dense_checks += 1
                    if legal != legal_by_lemma:
                        dense_disagreements.append(
                            {"n": n, "window": row["window"], "submask": sub,
                             "legal_dense": legal, "legal_lemma": legal_by_lemma})
                rows.append(row)
            legal_rows = [r for r in rows
                          if r.get("legal_dense", r["legal_lemma"])]
            windows.append({
                "window": [sites[0], sites[-1]], "width": width,
                "assignments": len(rows),
                "legal": len(legal_rows),
                "best_local_tcount": (min(r["local_tcount"] for r in legal_rows)
                                      if legal_rows else None),
                "best_joint_saving": (max(r["joint_saving"] for r in legal_rows)
                                      if legal_rows else 0),
                "best_joint_tcount": (min(r["joint_tcount"] for r in legal_rows)
                                      if legal_rows else None),
            })

    return {
        "n": n, "qubits": circuit.num_qubits, "blocks": m,
        "gate_names": gate_names,
        "only_permutation_atoms": only_permutation_atoms,
        "tcount_all_exact": base_t, "tcount_all_relative": all_relative_t,
        "tcount_per_block_exact": EXACT_T, "tcount_per_block_relative": RELATIVE_T,
        "windows": windows, "window_rows": sum(w["assignments"] for w in windows),
        "dense_checks": dense_checks,
        "dense_disagreements": dense_disagreements,
        "every_window_all_legal": all(w["legal"] == w["assignments"] for w in windows),
        "best_joint_saving": max(w["best_joint_saving"] for w in windows),
        "background_mask": background_mask,
    }


# ---------------------------------------------------------------------------
# Day 2: baselines
# ---------------------------------------------------------------------------

def baselines(n: int, dense: bool) -> Dict[str, Any]:
    circuit = SCAN.two_adders(n)
    positions = ccx_positions(circuit)
    m = len(positions)
    base = replace_positions(circuit, [])
    base_op = np.asarray(Operator(base).data, dtype=complex) if dense else None

    def legal(mask: int) -> bool:
        candidate = replace_positions(circuit, mask_to_positions(positions, mask))
        if base_op is None:
            return True          # structural oracle: all masks are legal
        ok, _, _ = dense_legal(base_op, candidate)
        return ok

    # (1) our local pass: the shipped support tracker, one block at a time
    schedule = RB_tracker_schedule(circuit)
    local_mask = 0
    for index in range(m):
        if ST.track_safe(schedule, 1 << index):
            local_mask |= 1 << index
    local_t = total_t(m, local_mask)
    local_legal = legal(local_mask)

    # (2) per-block greedy with the EXACT oracle (strongest per-block baseline)
    greedy_mask = 0
    for index in range(m):
        candidate = greedy_mask | (1 << index)
        if legal(candidate):
            greedy_mask = candidate
    greedy_t = total_t(m, greedy_mask)
    greedy_saving = (EXACT_T - RELATIVE_T) * bin(greedy_mask).count("1")

    # (3) PyZX full_reduce
    pyzx_t = SCAN.fr_tcount(circuit, timeout=60.0)

    return {
        "n": n, "blocks": m,
        "exact_t": total_t(m, 0),
        "greedy_mask": greedy_mask,
        "local_pass": {"mask_bits": bin(local_mask).count("1"), "tcount": local_t,
                       "legality_verified": local_legal,
                       "saving": (EXACT_T - RELATIVE_T) * bin(local_mask).count("1")},
        "per_block_greedy": {"mask_bits": bin(greedy_mask).count("1"),
                             "tcount": greedy_t, "saving": greedy_saving},
        "pyzx_full_reduce_t": pyzx_t,
        "selection_lower_bound_t": m * RELATIVE_T,
        "selection_minus_pyzx": (m * RELATIVE_T - pyzx_t) if pyzx_t is not None else None,
    }


def RB_tracker_schedule(circuit: QuantumCircuit):
    """Block schedule for the tracker: one entry per ccx, cx gates as layers."""
    blocks: List[Dict[str, Any]] = []
    layers: List[List[Tuple[str, List[int]]]] = []
    pending: List[Tuple[str, List[int]]] = []
    for ins in circuit.data:
        qubits = [circuit.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "ccx":
            layers.append(pending)
            pending = []
            blocks.append({"controls": qubits[:2], "ancillas": [], "target": qubits[2]})
        else:
            pending.append((ins.operation.name, qubits))
    layers.append(pending)
    return {"total": circuit.num_qubits, "layers": layers, "blocks": blocks}


# ---------------------------------------------------------------------------

def verdict(per_n: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-registered GO/KILL rule, literally."""
    deltas = []
    gaps = []
    for row in per_n:
        best = row["best_joint_saving"]
        greedy = row["baselines"]["per_block_greedy"]["saving"]
        deltas.append(best - greedy)
        gap = row["baselines"]["selection_minus_pyzx"]
        gaps.append(gap)
    growth = len(deltas) >= 2 and deltas[-1] > deltas[0]
    go = (min(deltas) >= 10 and all(g is not None and g <= 5 for g in gaps) and growth)
    if all(d == 0 for d in deltas):
        reason = "delta == 0 in every enumerated window: selection adds nothing"
    elif not all(g is not None and g <= 5 for g in gaps):
        reason = "selection-space lower bound stays >5 T above PyZX"
    elif not growth:
        reason = "gain does not grow with n"
    else:
        reason = "GO criteria satisfied"
    return {"deltas": deltas, "pyzx_gaps": gaps, "growth": growth,
            "GO": bool(go), "reason": reason}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="3,4,5,6")
    parser.add_argument("--quick", action="store_true", help="n = 3 only")
    parser.add_argument("--dense-budget-n5", type=int, default=4,
                        help="dense spot-checks at n>=5")
    parser.add_argument("--out", default=str(ROOT / "reports" / "figure_data" /
                                            "window_selection_probe.json"))
    args = parser.parse_args(argv)
    sizes = [3] if args.quick else [int(s) for s in args.sizes.split(",") if s.strip()]
    t0 = time.perf_counter()

    print("=" * 100)
    print(f"  WINDOW SELECTION PROBE  version={PROBE_VERSION}")
    print("  family: chained CDKM adder (two adders on the same register)")
    print("  decision space: per-ccx EXACT(7T) vs RELATIVE(4T); topology fixed")
    print("=" * 100)

    premises = check_structural_premises()
    print("\n  structural premises (checked, not assumed):")
    for key, value in premises.items():
        print(f"    {key:<40} {value}")
    assert premises["exact_block_is_permutation"]
    assert premises["exact_block_equals_ccx"] < TOL
    assert premises["relative_x_exact_dagger_offdiag"] < TOL
    assert premises["relative_x_exact_dagger_unit_dev"] < TOL
    assert premises["exact_block_tcount"] == EXACT_T
    assert premises["relative_block_tcount"] == RELATIVE_T

    per_n: List[Dict[str, Any]] = []
    for n in sizes:
        print("\n" + "-" * 100)
        print(f"  Day 2 baselines first (they define the greedy background)  (n={n})")
        base = baselines(n, dense=(n <= 4))
        print(f"    local={(base['local_pass']['tcount'], base['local_pass']['mask_bits'])} "
              f"greedy=({base['per_block_greedy']['tcount']}, "
              f"{base['per_block_greedy']['mask_bits']}) "
              f"pyzx={base['pyzx_full_reduce_t']} "
              f"selection-bound={base['selection_lower_bound_t']}")
        print(f"  Day 1: enumerate adjacent pairs/triples against greedy  (n={n})")
        row = enumerate_windows(n, dense_budget=args.dense_budget_n5 if n >= 5 else 0,
                                background_mask=base["greedy_mask"])
        print(f"    qubits={row['qubits']} blocks(m)={row['blocks']} "
              f"gates={row['gate_names']} only-permutations={row['only_permutation_atoms']}")
        print(f"    T: all-exact={row['tcount_all_exact']} "
              f"all-relative={row['tcount_all_relative']} "
              f"(={RELATIVE_T}*m)  enumerated candidates={row['window_rows']}")
        print(f"    dense exact checks={row['dense_checks']} "
              f"disagreements with structural={len(row['dense_disagreements'])}")
        print(f"    windows: {len(row['windows'])}  all-legal: "
              f"{row['every_window_all_legal']}  "
              f"best joint saving={row['best_joint_saving']} "
              f"(greedy={base['per_block_greedy']['saving']})")
        row["baselines"] = base
        per_n.append(row)

    result = verdict(per_n)
    print("\n" + "=" * 100)
    print("  Day 3: pre-registered decision")
    print(f"    delta (best window - greedy)  = {result['deltas']}")
    print(f"    gap  (selection bound - pyzx) = {result['pyzx_gaps']}")
    print(f"    growth with n                 = {result['growth']}")
    print(f"    GO                            = {result['GO']}  ({result['reason']})")
    print(f"    VERDICT                       = "
          f"{'GO' if result['GO'] else 'KILL'}")
    print("=" * 100)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe_version": PROBE_VERSION,
        "family": "chained CDKM adder (two_adders)",
        "decision_space": "per-ccx exact(7T) vs relative(4T), topology fixed",
        "structural_premises": premises,
        "sizes": sizes,
        "per_n": per_n,
        "decision": result,
        "verdict": "GO" if result["GO"] else "KILL",
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
