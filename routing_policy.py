#!/usr/bin/env python3
"""Per-block routing-policy search ("step 3") for the hybrid GNN solver.

Why this exists
---------------
The phase polynomial of every Hadamard-separated block is *exactly* computable
symbolically (``PhasePoly.fromCircuit``), so "predict the terms" is not where
learning can pay off -- the symbolic extractor already returns them, and the
unitary can be verified per block.  The real decision is *how to synthesise*
each block: the Steiner/Meijer router takes a method, an elimination mode, root
and split heuristics, and is sensitive to the order in which parity terms are
fed.  Those choices change T-depth, T-count and CX-count for the same block.

This module provides
-------------------
``ROUTING_POLICIES``      a small enumerated policy space (6 policies),
``route_with_policy``     route one phase polynomial with one policy,
``best_routing_policy``   oracle search over the space (upper bound),
``headroom_report``       per-block default vs oracle costs -> tells you whether
                          a *learned* policy head is worth training at all.

It is intentionally a separate file: it imports ``Linux-MCX`` through the same
source-file loader that ``evaluate_model.py`` uses and modifies nothing.

Usage
-----
    .venv/bin/python routing_policy.py --num-controls 4 --mcx-basis clifford_t
    .venv/bin/python routing_policy.py --num-controls 4 --attempts 3 --json /tmp/policy.json

Wiring it into the evaluator (three lines, when the file is writable)::

    import routing_policy as rp
    index, table = rp.best_routing_policy(block_terms, qubit_count)
    routed = rp.route_with_policy(block_terms, qubit_count, rp.ROUTING_POLICIES[index])
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parent
LINUX_MCX_PATH = ROOT / "Linux-MCX"


def load_linux_mcx():
    """Import ``Linux-MCX`` (no ``.py`` suffix) without modifying it."""
    loader = importlib.machinery.SourceFileLoader("linux_mcx_policy", str(LINUX_MCX_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {LINUX_MCX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


LINUX_MCX = load_linux_mcx()

from pyzx.routing.architecture import FULLY_CONNECTED, create_architecture  # noqa: E402
from pyzx.routing.cnot_mapper import ElimMode  # noqa: E402
from pyzx.routing.parity_maps import Parity  # noqa: E402
from pyzx.routing.phase_poly import (  # noqa: E402
    PhasePoly,
    RootHeuristic,
    RoutingMethod,
    SplitHeuristic,
    route_phase_poly,
)


# ---------------------------------------------------------------------------
# policy space
# ---------------------------------------------------------------------------

#: ``order`` controls the order in which parity terms are handed to the router
#: (the Gray/Steiner synthesis is order sensitive), ``method``/``mode``/``root``
#: /``split`` are the router's own knobs.
ROUTING_POLICIES: "tuple[dict[str, str], ...]" = (
    {
        "name": "gray_meijer_recursive_as_is",
        "method": "GRAY_MEIJER",
        "mode": "STEINER_MODE",
        "root": "RECURSIVE",
        "split": "RANDOM",
        "order": "as_is",
    },
    {
        "name": "meijer_recursive_count",
        "method": "MEIJER",
        "mode": "STEINER_MODE",
        "root": "RECURSIVE",
        "split": "COUNT",
        "order": "as_is",
    },
    {
        "name": "gray_arity_desc_support",
        "method": "GRAY_MEIJER",
        "mode": "STEINER_MODE",
        "root": "ARITY",
        "split": "ARITY",
        "order": "desc_support",
    },
    {
        "name": "meijer_random_asc_support",
        "method": "MEIJER",
        "mode": "STEINER_MODE",
        "root": "RANDOM",
        "split": "RANDOM",
        "order": "asc_support",
    },
    {
        "name": "gray_gauss_count",
        "method": "GRAY_MEIJER",
        "mode": "GAUSS_MODE",
        "root": "RECURSIVE",
        "split": "COUNT",
        "order": "as_is",
    },
    {
        "name": "meijer_exhaustive_gray_code",
        "method": "MEIJER",
        "mode": "STEINER_MODE",
        "root": "EXHAUSTIVE",
        "split": "COUNT",
        "order": "gray",
    },
)


def make_architecture(name: str, n_qubits: int):
    """Build a routing architecture (``full`` = fully connected)."""
    if not name or str(name).lower() == "full":
        return create_architecture(FULLY_CONNECTED, n_qubits=n_qubits)
    # ``create_architecture`` matches lowercase register names ('line', 'square',
    # 'circle', 'ibm_qx2', ...) -- an upper-cased name is rejected.
    return create_architecture(str(name).lower(), n_qubits=n_qubits)


def policy_names() -> "list[str]":
    return [policy["name"] for policy in ROUTING_POLICIES]


def _policy_kwargs(policy: "dict[str, str]") -> "dict[str, Any]":
    """Map the string policy onto the router's enums (missing = router default)."""
    kwargs: "dict[str, Any]" = {}
    if policy.get("method"):
        kwargs["method"] = getattr(RoutingMethod, policy["method"])
    if policy.get("mode"):
        kwargs["mode"] = getattr(ElimMode, policy["mode"])
    if policy.get("root"):
        kwargs["root_heuristic"] = getattr(RootHeuristic, policy["root"])
    if policy.get("split"):
        kwargs["split_heuristic"] = getattr(SplitHeuristic, policy["split"])
    return kwargs


def _parity_value(parity_bits: "Sequence[int]") -> int:
    value = 0
    for bit in parity_bits:
        value = (value << 1) | int(bit)
    return value


def order_terms(terms: "dict[tuple[int, ...], Fraction]", mode: str):
    """Order the parity terms for the router (order matters for Gray synthesis)."""
    items = list(terms.items())
    if mode == "desc_support":
        items.sort(key=lambda item: (-sum(int(b) for b in item[0]), item[0]))
    elif mode == "asc_support":
        items.sort(key=lambda item: (sum(int(b) for b in item[0]), item[0]))
    elif mode == "gray":
        # Gray-code order over the parity integer: consecutive parities differ by
        # one bit, which is what the Gray/Meijer synthesis exploits.
        items.sort(key=lambda item: ((lambda v: v ^ (v >> 1))(_parity_value(item[0])),))
    return items


def route_with_policy(
    terms: "dict[tuple[int, ...], Fraction]",
    n_qubits: int,
    policy: "dict[str, str]",
    attempts: int = 1,
    architecture_name: str = "full",
):
    """Route one phase polynomial with one explicit synthesis policy.

    ``terms`` maps a parity bit-tuple (one bit per qubit, as returned by
    ``Linux-MCX.extract_phase_terms``) to a phase in units of pi.
    """
    if not terms:
        return LINUX_MCX.zx.Circuit(n_qubits)
    ordered = order_terms(terms, policy.get("order", "as_is"))
    identity_rows = [[int(i == q) for i in range(n_qubits)] for q in range(n_qubits)]
    out_par = [Parity(row) for row in identity_rows]
    zphases: "dict[Parity, Fraction]" = {}
    for parity_bits, phase in ordered:
        bits = [int(b) for b in parity_bits][:n_qubits]
        bits += [0] * (n_qubits - len(bits))
        zphases[Parity(bits)] = Fraction(phase)

    phase_poly = PhasePoly(zphases, out_par)
    architecture = make_architecture(architecture_name, n_qubits)
    kwargs = _policy_kwargs(policy)

    best = None
    best_score = None
    for _ in range(max(1, int(attempts))):
        try:
            candidate = route_phase_poly(phase_poly, architecture, **kwargs)
        except Exception:
            continue
        metrics = LINUX_MCX.compute_ftq_metrics(candidate)
        score = (metrics.t_depth, metrics.t_count, metrics.cx_depth)
        if best_score is None or score < best_score:
            best, best_score = candidate, score
    if best is None:
        raise RuntimeError(f"policy {policy.get('name')} produced no circuit")
    return best


def policy_cost_row(
    terms, n_qubits: int, attempts: int = 1, architecture_name: str = "full"
) -> "list[dict[str, Any]]":
    """Route ``terms`` with every policy and return the cost table."""
    rows: "list[dict[str, Any]]" = []
    for index, policy in enumerate(ROUTING_POLICIES):
        row: "dict[str, Any]" = {"index": index, "name": policy["name"]}
        try:
            metrics = LINUX_MCX.compute_ftq_metrics(
                route_with_policy(
                    terms,
                    n_qubits,
                    policy,
                    attempts=attempts,
                    architecture_name=architecture_name,
                )
            )
            row.update(
                {
                    "t_count": metrics.t_count,
                    "t_depth": metrics.t_depth,
                    "cx_depth": metrics.cx_depth,
                    "ok": True,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            row.update({"t_count": None, "t_depth": None, "cx_depth": None, "ok": False, "error": str(exc)[:80]})
        rows.append(row)
    return rows


def best_routing_policy(terms, n_qubits: int, attempts: int = 1, architecture_name: str = "full"):
    """Oracle: return ``(best_index, table)`` over the policy space.

    The ordering is ``(T-depth, T-count, CX-depth)``: T-depth is the primary
    FTQC objective of this project.
    """
    table = policy_cost_row(
        terms, n_qubits, attempts=attempts, architecture_name=architecture_name
    )
    scored = [
        (row["t_depth"], row["t_count"], row["cx_depth"], row["index"])
        for row in table
        if row.get("ok")
    ]
    if not scored:
        return 0, table
    scored.sort()
    return scored[0][3], table


# ---------------------------------------------------------------------------
# headroom measurement
# ---------------------------------------------------------------------------


def block_policy_report(
    circuit,
    attempts: int = 2,
    verify: bool = True,
    verbose: bool = True,
    architecture_name: str = "full",
) -> "dict[str, Any]":
    """Compare the default route against the policy oracle, block by block.

    Returns a report with, per block and in total: the default cost, the oracle
    cost, the per-policy table, and (when ``verify``) whether every policy's
    circuit is still unitarily equivalent to the block.
    """
    linux_mcx = LINUX_MCX
    n_qubits = circuit.qubits
    blocks: "list[dict[str, Any]]" = []
    totals = {"default": {"t_count": 0, "t_depth": 0, "cx_depth": 0},
              "oracle": {"t_count": 0, "t_depth": 0, "cx_depth": 0}}

    for index, (kind, gates) in enumerate(linux_mcx._phase_segments(circuit)):
        if kind != "seg":
            continue
        block_circuit = linux_mcx._circuit_block(n_qubits, gates)
        terms = linux_mcx._segment_terms(gates, n_qubits)
        if not terms:
            continue
        default_circuit = linux_mcx.route_phase_term_set(terms, n_qubits, attempts=attempts)
        default_metrics = linux_mcx.compute_ftq_metrics(default_circuit)
        best_index, table = best_routing_policy(
            terms, n_qubits, attempts=attempts, architecture_name=architecture_name
        )
        best_row = table[best_index]
        oracle_metrics = {
            "t_count": best_row["t_count"],
            "t_depth": best_row["t_depth"],
            "cx_depth": best_row["cx_depth"],
        }
        for key in ("t_count", "t_depth", "cx_depth"):
            totals["default"][key] += int(default_metrics.__dict__[key])
            totals["oracle"][key] += int(oracle_metrics[key] or 0)

        verified = None
        if verify:
            verified = []
            for row in table:
                if not row.get("ok"):
                    verified.append(False)
                    continue
                candidate = route_with_policy(
                    terms,
                    n_qubits,
                    ROUTING_POLICIES[row["index"]],
                    attempts=attempts,
                    architecture_name=architecture_name,
                )
                metrics = linux_mcx.small_unitary_metrics(block_circuit, candidate)
                verified.append(None if metrics is None else bool(metrics[0] < 1e-8 and metrics[1] < 1e-8))

        blocks.append(
            {
                "block": len(blocks),
                "terms": len(terms),
                "default": {
                    "t_count": default_metrics.t_count,
                    "t_depth": default_metrics.t_depth,
                    "cx_depth": default_metrics.cx_depth,
                },
                "oracle": oracle_metrics,
                "best_policy": best_row["name"],
                "delta_t_depth": int(default_metrics.t_depth) - int(oracle_metrics["t_depth"] or 0),
                "delta_t_count": int(default_metrics.t_count) - int(oracle_metrics["t_count"] or 0),
                "table": table,
                "verified": verified,
            }
        )
        if verbose:
            print(
                f"    [block {len(blocks) - 1:2d}] terms={len(terms):2d} | "
                f"default T-count={default_metrics.t_count}, T-depth={default_metrics.t_depth} | "
                f"oracle T-count={oracle_metrics['t_count']}, T-depth={oracle_metrics['t_depth']} "
                f"({best_row['name']}) | "
                f"delta T-count={int(default_metrics.t_count) - int(oracle_metrics['t_count'] or 0):+d}, "
                f"delta T-depth={int(default_metrics.t_depth) - int(oracle_metrics['t_depth'] or 0):+d}"
            )
    report = {
        "blocks": blocks,
        "totals": totals,
        "delta_t_count": totals["default"]["t_count"] - totals["oracle"]["t_count"],
        "delta_t_depth": totals["default"]["t_depth"] - totals["oracle"]["t_depth"],
    }
    if verbose:
        print(
            f"    [headroom] default total: T-count={totals['default']['t_count']}, "
            f"T-depth={totals['default']['t_depth']}, CX={totals['default']['cx_depth']} | "
            f"oracle total: T-count={totals['oracle']['t_count']}, "
            f"T-depth={totals['oracle']['t_depth']}, CX={totals['oracle']['cx_depth']} | "
            f"gain: -{report['delta_t_count']} T, -{report['delta_t_depth']} T-depth"
        )
    return report


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description="Per-block routing-policy headroom measurement.")
    parser.add_argument("--num-controls", type=int, default=4)
    parser.add_argument("--mcx-basis", choices=("clifford_t", "rz"), default="clifford_t")
    parser.add_argument("--attempts", type=int, default=2, help="routing attempts per policy")
    parser.add_argument(
        "--architecture",
        default="full",
        help="routing topology: full | line | square | circle | ibm_qx2 | ...",
    )
    parser.add_argument("--no-verify", action="store_true", help="skip the per-policy unitary check")
    parser.add_argument("--json", default=None, help="optional path to dump the report as JSON")
    args = parser.parse_args(argv)

    source = LINUX_MCX.make_mcx_source(args.num_controls, basis=args.mcx_basis)
    print(
        f"source: C^{args.num_controls}X ({args.mcx_basis}) | {source.qubits} qubits, "
        f"{len(source.gates)} gates, {len(LINUX_MCX._phase_blocks(source))} H-blocks",
        flush=True,
    )
    print(f"    architecture: {args.architecture}", flush=True)
    report = block_policy_report(
        source,
        attempts=args.attempts,
        verify=not args.no_verify,
        architecture_name=args.architecture,
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"    [json] report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
