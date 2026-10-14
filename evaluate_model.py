"""
Evaluate the trained phase-gadget GNN on the C^4X (5-qubit) stress test.

The file is a driver only: ``Linux-MCX`` is never modified and every
algorithmic step goes through one of its existing functions
(``PhaseGadgetGNN``, ``make_mcx_source``, ``report_cx_and_depth``,
``predict_phase_polynomial``, ``build_circuit_from_phase_terms``,
``small_unitary_metrics``, ``print_phase_terms``).

The loaded checkpoint is the *last-epoch* (robust) autoregressive model
``checkpoints/best_gnn_autoregressive_last.pt``.  Decoding runs through the
advanced inference path: ``soft_feedback=True`` feeds the decoder's own
probabilities back (mild exposure-bias mitigation) and
``return_diagnostics=True`` exposes the Stop/Count head consistency scoring
(``stop_len``, ``count_len``, ``agreed``, ``horizon``, ``pruned_tail``) that is
printed before the decoded phase polynomial.

Feature extraction is self-contained
-----------------------------------
``extract_inference_features(graph, n_qubits)`` builds the inference batch
straight from the *already simplified* ZX graph supplied by the caller.  It
only reads the diagram; it never simplifies, extracts, or rebuilds anything,
so the topology seen by the GNN is exactly the topology of the input graph.

The node layout reproduces the training-time layout of the dataset the
checkpoint was fitted on (``node_dim = 27``):

    [is_z_spider, is_x_spider, is_boundary, is_ground, phase_over_pi,
     phase_sin, phase_cos, phase_is_non_clifford, degree_over_n,
     support_size_over_n, qubit_over_n]            -> 11 columns
    + 16-column per-qubit boundary-support mask     -> 27 columns total

Edges carry a 2-way one-hot ``edge_attr`` ([1, 0] simple, [0, 1] Hadamard)
and are emitted in both directions, matching the training batches.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import math
import sys
from pathlib import Path

import torch
import pyzx as zx
from torch_geometric.data import Data
ROOT = Path(__file__).resolve().parent
LINUX_MCX_PATH = ROOT / "Linux-MCX"
DATA_GENERATOR_PATH = ROOT / "data_generator.py"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_gnn_v5_full.pt"

NUM_CONTROLS = 4                     # C^4X -> 5 qubits
SUPPORT_WIDTH = 16                   # = Linux-MCX MAX_QUBITS (support-mask padding)
NODE_FEATURE_WIDTH = 11 + SUPPORT_WIDTH   # 27, the checkpoint's node_dim
TOTAL_STEPS = 8

# Target synthesis for the C^nX source.
#   "clifford_t" - exact Toffoli AND-ladder, every rotation is a multiple of
#                  pi/4 (fine-angle rotations = 0) -> T-count is meaningful
#   "rz"         - Qiskit {H, CX, Rz} synthesis: exact, but its pi/8 / pi/16
#                  rotations have no Clifford+T realisation, so a T-count
#                  evaluation against it is structurally ~0
DEFAULT_MCX_BASIS = "clifford_t"
MCX_BASIS = DEFAULT_MCX_BASIS      # resolved from --mcx-basis in main()
HYBRID_BLOCKS = True               # resolved from --hybrid-blocks in main()
RELATIVE_PHASE = False             # resolved from --relative-phase in main()
VARIANT_POLICY = True              # resolved from --variant-policy in main()
VARIANT_CHECKPOINT = ROOT / "checkpoints" / "variant_policy_best.pt"
VARIANT_RL_CHECKPOINT = ROOT / "checkpoints" / "variant_policy_rl_best.pt"
NUM_LARGE_TESTS = 10            # mixed-circuit stress tests to run
LARGE_MIN_BLOCKS = 30
LARGE_MAX_BLOCKS = 60
LARGE_H_PROB = 0.5              # Hadamard density inside the mixed stress layers
VARIANT_BLOCKS = 3
VARIANT_INSTANCES = 8
VARIANT_SEED = 20261008
VARIANT_H_PROB = 0.7


def step(index: int, message: str) -> None:
    print(f"[{index}/{TOTAL_STEPS}] {message}", flush=True)


def load_data_generator():
    """Import ``data_generator.py`` so the evaluation uses the training pipeline."""
    loader = importlib.machinery.SourceFileLoader("data_generator_eval", str(DATA_GENERATOR_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {DATA_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def load_variant_runtime():
    """Import ``variant_policy_runtime.py`` (strategy-B deployment glue)."""
    loader = importlib.machinery.SourceFileLoader(
        "variant_policy_runtime_eval", str(ROOT / "variant_policy_runtime.py")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("cannot build an import spec for variant_policy_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def load_mixed_stress():
    """Import ``mixed_stress.py`` (heterogeneous large-circuit stress test)."""
    loader = importlib.machinery.SourceFileLoader(
        "mixed_stress_eval", str(ROOT / "mixed_stress.py")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError("cannot build an import spec for mixed_stress.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def training_target(linux_mcx, dg, circuit, max_terms: int):
    """Reference target in the dataset's own representation.

    Mirrors ``data_generator``: ``clifford_simp`` + ``full_reduce`` +
    re-extraction, block-wise phase extraction, then the canonical
    ``(block, descending |phase|, parity)`` ordering and the model's row budget.
    Returns ``(graph, simplified_circuit, rows)`` with one
    ``(parity_bits, phase_class, block_index)`` per row.
    """
    config = dg.GenConfig(samples=1, out_dir="/tmp/_eval_target")
    graph, simplified, _stats = dg.simplify_circuit(circuit, config)
    polynomial = dg.extract_phase_polynomial(simplified, config.phase_mode, source="simplified")
    flat = [
        (int(parity), int(units) % dg.PHASE_CLASSES, block_id)
        for block_id, block in enumerate(polynomial.blocks)
        for parity, units in block.items()
    ]
    flat.sort(key=lambda term: (term[2], -dg.phase_class_magnitude(term[1]), -term[0], term[1]))
    flat = flat[:max_terms]
    rows = [
        (tuple((parity >> q) & 1 for q in range(circuit.qubits)), cls, block_id)
        for parity, cls, block_id in flat
    ]
    return graph, simplified, rows


def load_linux_mcx():
    """Import ``Linux-MCX`` (the file has no ``.py`` suffix) without touching it."""
    loader = importlib.machinery.SourceFileLoader("linux_mcx", str(LINUX_MCX_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot build an import spec for {LINUX_MCX_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# inference features (no data_generator, no re-simplification, read-only)
# ---------------------------------------------------------------------------


def _boundary_support_masks(graph, vertices, n_qubits: int) -> dict:
    """Component-wise boundary qubit support bitmask for every vertex.

    Vertices that sit in the same connected component of the diagram share the
    union of the qubit indices of the boundary spiders in that component.  If a
    component has no boundary spider at all we fall back to the qubit indices
    of its vertices.  The traversal is purely read-only.
    """
    neighbours = {v: set(graph.neighbors(v)) for v in vertices}
    support: dict = {}
    seen: set = set()

    for start in vertices:
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbour in neighbours[vertex]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)

        mask = 0
        for vertex in component:
            if int(graph.type(vertex)) == int(zx.VertexType.BOUNDARY):
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        if mask == 0:
            for vertex in component:
                qubit = graph.qubit(vertex)
                if qubit is not None and 0 <= int(qubit) < n_qubits:
                    mask |= 1 << int(qubit)
        for vertex in component:
            support[vertex] = mask
    return support


def extract_inference_features(graph, n_qubits: int) -> Data:
    """Turn a simplified ZX graph into the ``[num_nodes, 27]`` GNN input.

    Read-only: the graph is traversed, never mutated, so ``data.x`` always has
    one row per input vertex and the diagram's topology is preserved exactly.
    """
    vertices = list(graph.vertices())
    index_of = {vertex: i for i, vertex in enumerate(vertices)}
    support = _boundary_support_masks(graph, vertices, n_qubits)

    node_features = []
    for vertex in vertices:
        vtype = int(graph.type(vertex))
        phase = graph.phase(vertex)
        phase_value = float(phase) if phase is not None else 0.0
        degree = int(graph.vertex_degree(vertex))
        mask = support.get(vertex, 0)
        size = bin(int(mask)).count("1") if mask else 0
        qubit = graph.qubit(vertex)
        is_non_clifford = abs((phase_value * 4.0) % 1.0) > 1e-9

        base = [
            1.0 if vtype == int(zx.VertexType.Z) else 0.0,          # is_z_spider
            1.0 if vtype == int(zx.VertexType.X) else 0.0,          # is_x_spider
            1.0 if vtype == int(zx.VertexType.BOUNDARY) else 0.0,   # is_boundary
            1.0 if graph.is_ground(vertex) else 0.0,                # is_ground
            (phase_value % 2.0) / 2.0,                              # phase_over_pi
            math.sin(math.pi * phase_value),                        # phase_sin
            math.cos(math.pi * phase_value),                        # phase_cos
            1.0 if is_non_clifford else 0.0,                        # phase_is_non_clifford
            degree / max(n_qubits, 1),                              # degree_over_n
            size / max(n_qubits, 1),                                # support_size_over_n
            float(qubit) / max(n_qubits - 1, 1) if qubit is not None else -1.0,
        ]
        support_row = [
            1.0 if (int(mask) >> q) & 1 else 0.0 for q in range(SUPPORT_WIDTH)
        ]
        node_features.append(base + support_row)

    x = torch.tensor(node_features, dtype=torch.float)

    edges = []
    edge_types = []
    for edge in graph.edges():
        a, b = int(edge[0]), int(edge[1])
        if a not in index_of or b not in index_of:
            continue
        etype = 1 if int(graph.edge_type(edge)) == int(zx.EdgeType.HADAMARD) else 0
        i, j = index_of[a], index_of[b]
        edges.append((i, j))
        edge_types.append(etype)
        edges.append((j, i))
        edge_types.append(etype)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    edge_attr = torch.zeros((len(edge_types), 2), dtype=torch.float)
    for row, etype in enumerate(edge_types):
        edge_attr[row, etype] = 1.0

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# main flow
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    """CLI: only the C^nX source synthesis is configurable."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained PhaseGadgetGNN on the C^4X stress test. "
            "Uses the checkpoint's own hyper-parameters; the only option is the "
            "synthesis basis of the source circuit."
        )
    )
    parser.add_argument(
        "--relative-phase",
        dest="relative_phase",
        action="store_true",
        default=False,
        help="build the source as the certified 4-T/RCCX relative-phase ladder "
        "(~43%% fewer T gates; equivalence is checked with the diagonal-correction test)",
    )
    parser.add_argument(
        "--hybrid-blocks",
        dest="hybrid_blocks",
        action="store_true",
        default=True,
        help="block-by-block hybrid solve: per-block GNN proposal, local verify, Steiner routing",
    )
    parser.add_argument(
        "--no-hybrid-blocks",
        dest="hybrid_blocks",
        action="store_false",
        help="disable the hybrid block solver (single-shot proposal only)",
    )
    parser.add_argument(
        "--mcx-basis",
        choices=("clifford_t", "rz"),
        default=DEFAULT_MCX_BASIS,
        help="source synthesis: 'clifford_t' = exact Toffoli AND-ladder with "
        "pi/4-only rotations (T-count meaningful, default); 'rz' = Qiskit "
        "{H,CX,Rz} (fine angles, T-count structurally ~0)",
    )
    parser.add_argument(
        "--variant-policy",
        dest="variant_policy",
        action="store_true",
        default=True,
        help="run the trained strategy-B per-block exact/relative variant policy",
    )
    parser.add_argument(
        "--no-variant-policy",
        dest="variant_policy",
        action="store_false",
        help="skip the strategy-B variant policy",
    )
    parser.add_argument(
        "--variant-checkpoint",
        default=str(ROOT / "checkpoints" / "variant_policy_best.pt"),
    )
    parser.add_argument("--variant-blocks", type=int, default=3)
    parser.add_argument("--variant-instances", type=int, default=8)
    parser.add_argument("--variant-seed", type=int, default=20261008)
    parser.add_argument("--variant-h-prob", type=float, default=0.7)
    parser.add_argument("--variant-large-tests", type=int, default=10,
                        help="number of mixed large-circuit stress tests (0 = off)")
    parser.add_argument("--variant-large-min-blocks", type=int, default=30)
    parser.add_argument("--variant-large-max-blocks", type=int, default=60)
    parser.add_argument("--variant-large-h-prob", type=float, default=0.5,
                        help="Hadamard probability inside the mixed stress-test layers")
    parser.add_argument(
        "--variant-rl-checkpoint",
        default=str(ROOT / "checkpoints" / "variant_policy_rl_best.pt"),
        help="if it exists, use the RL (part b) weights for the block ordering",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    global MCX_BASIS, HYBRID_BLOCKS, RELATIVE_PHASE
    global VARIANT_POLICY, VARIANT_CHECKPOINT, VARIANT_BLOCKS, VARIANT_INSTANCES, VARIANT_SEED, VARIANT_H_PROB
    global VARIANT_RL_CHECKPOINT
    global NUM_LARGE_TESTS, LARGE_MIN_BLOCKS, LARGE_MAX_BLOCKS, LARGE_H_PROB
    MCX_BASIS = args.mcx_basis
    HYBRID_BLOCKS = args.hybrid_blocks
    RELATIVE_PHASE = bool(args.relative_phase)
    VARIANT_POLICY = bool(args.variant_policy)
    VARIANT_CHECKPOINT = Path(args.variant_checkpoint)
    VARIANT_BLOCKS = int(args.variant_blocks)
    VARIANT_INSTANCES = int(args.variant_instances)
    VARIANT_SEED = int(args.variant_seed)
    VARIANT_H_PROB = float(args.variant_h_prob)
    VARIANT_RL_CHECKPOINT = Path(args.variant_rl_checkpoint)
    NUM_LARGE_TESTS = int(args.variant_large_tests)
    LARGE_MIN_BLOCKS = int(args.variant_large_min_blocks)
    LARGE_MAX_BLOCKS = int(args.variant_large_max_blocks)
    LARGE_H_PROB = float(args.variant_large_h_prob)
    # -- 1. imports --------------------------------------------------------
    step(1, "Import torch / pyzx as zx / Linux-MCX ...")
    linux_mcx = load_linux_mcx()
    dg = load_data_generator()
    print(f"    torch {torch.__version__}, pyzx {zx.__version__}", flush=True)
    print(f"    Linux-MCX loaded from {LINUX_MCX_PATH}", flush=True)

    # -- 2. checkpoint -----------------------------------------------------
    step(2, f"torch.load({CHECKPOINT_PATH.relative_to(ROOT)}, map_location='cpu') ...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    print(
        "    node_dim = {}, max_phase_terms = {}, hidden = {}, max_qubits = {}".format(
            checkpoint["node_dim"],
            checkpoint["max_phase_terms"],
            checkpoint["hidden"],
            checkpoint["max_qubits"],
        ),
        flush=True,
    )
    if int(checkpoint["node_dim"]) != NODE_FEATURE_WIDTH:
        raise ValueError(
            f"checkpoint node_dim={checkpoint['node_dim']} but "
            f"extract_inference_features emits {NODE_FEATURE_WIDTH}"
        )

    # -- 3. rebuild the GNN from the checkpoint hyper-parameters -----------
    step(3, "Instantiate Linux-MCX.PhaseGadgetGNN from the checkpoint hyper-parameters ...")
    model = linux_mcx.PhaseGadgetGNN(
        node_dim=checkpoint["node_dim"],
        max_phase_terms=checkpoint["max_phase_terms"],
        hidden=checkpoint["hidden"],
        max_qubits=checkpoint["max_qubits"],
    )

    # -- 4. weights + eval mode -------------------------------------------
    step(4, "load_state_dict(checkpoint['model_state']) + model.eval() ...")
    linux_mcx.load_compatible_state_dict(model, checkpoint["model_state"], quiet=False)
    model.eval()
    print("    weights restored, model in eval mode", flush=True)

    # -- 5. source circuit ------------------------------------------------
    step(
        5,
        f"Linux-MCX.make_mcx_source(num_controls={NUM_CONTROLS}, "
        f"basis='{MCX_BASIS}') ...",
    )
    target_basic = linux_mcx.make_mcx_source(
        num_controls=NUM_CONTROLS, basis=MCX_BASIS, relative_phase=RELATIVE_PHASE
    )
    qubit_count = target_basic.qubits
    print(f"    source circuit: C^{NUM_CONTROLS}X on {qubit_count} qubits", flush=True)
    print(
        f"    synthesis basis: {MCX_BASIS}"
        + (
            " (exact Toffoli AND-ladder, Clifford+T, T-count is meaningful)"
            if MCX_BASIS == "clifford_t"
            else " (Qiskit {H,CX,Rz}: fine angles -> T-count structurally ~0)"
        ),
        flush=True,
    )
    source_metrics = linux_mcx.compute_ftq_metrics(target_basic)
    print(
        f"    source FTQC: T-count={source_metrics.t_count}, "
        f"T-depth={source_metrics.t_depth}, CX-depth={source_metrics.cx_depth}, "
        f"fine-angle rotations={source_metrics.fine_count}",
        flush=True,
    )
    if source_metrics.fine_count:
        print(
            "    [warn] the source contains rotations that are not multiples of "
            "pi/4: no Clifford+T circuit can match it exactly, so the reported "
            "T-count is not meaningful. Re-run with --mcx-basis clifford_t.",
            flush=True,
        )
    linux_mcx.report_cx_and_depth(target_basic, "source C^4X")

    # -- 6. ZX simplification + inference features -------------------------
    step(6, "to_graph() -> zx.simplify.clifford_simp() -> extract_inference_features() ...")
    graph = target_basic.to_graph()
    vertices_before = len(list(graph.vertices()))
    zx.simplify.clifford_simp(graph)
    vertices_clifford = len(list(graph.vertices()))
    # The generator trains on the FULLY reduced diagram (clifford_simp +
    # full_reduce + re-extraction).  Stopping at clifford_simp feeds the GNN a
    # larger, out-of-distribution diagram (measured on the C^4X ladder: 79 vs 63
    # vertices) -- one of the reasons predictions looked poor.
    zx.simplify.full_reduce(graph)
    vertices_after = len(list(graph.vertices()))

    # the extraction is strictly read-only: snapshot the topology before/after
    def _topology_snapshot():
        return (
            tuple(sorted(graph.vertices())),
            tuple(sorted(tuple(sorted(edge)) for edge in graph.edges())),
        )

    topology_before = _topology_snapshot()
    data = extract_inference_features(graph, qubit_count)
    if _topology_snapshot() != topology_before:
        raise RuntimeError("extract_inference_features mutated the ZX graph")
    print(
        f"    ZX graph: {vertices_before} -> {vertices_clifford} (clifford_simp) -> "
        f"{vertices_after} (full_reduce: the training-time diagram used by the GNN)",
        flush=True,
    )
    if data.x.shape != (vertices_after, NODE_FEATURE_WIDTH):
        raise ValueError(f"unexpected feature shape {tuple(data.x.shape)}")
    print(
        f"    Data: x = {tuple(data.x.shape)}, edge_index = "
        f"{tuple(data.edge_index.shape)}, edge_attr = {tuple(data.edge_attr.shape)}",
        flush=True,
    )
    print(
        "    topology check: vertices and edges unchanged by the feature pass",
        flush=True,
    )

    # -- 7. GNN inference -------------------------------------------------
    step(
        7,
        "Linux-MCX.predict_phase_polynomial(model, data, qubit_count, "
        "soft_feedback=False, return_diagnostics=True) ...",
    )
    pred_terms, diag = linux_mcx.predict_phase_polynomial(
        model,
        data,
        qubit_count,
        soft_feedback=True,
        return_diagnostics=True,
    )
    print(f"    GNN proposed {len(pred_terms)} phase terms", flush=True)
    print(
        f"    [Diagnostics] phase blocks predicted: {diag.get('n_blocks', 1)}",
        flush=True,
    )
    print(
        f"    [Diagnostics] Stop Head: {diag['stop_len']} steps, "
        f"Count Head: {diag['count_len']} steps",
        flush=True,
    )
    print(
        f"    [Diagnostics] Agreed: {diag['agreed']}, "
        f"Horizon used: {diag['horizon']}, Tail pruned: {diag['pruned_tail']}",
        flush=True,
    )
    linux_mcx.print_phase_terms("GNN phase polynomial", pred_terms)

    # -- 8. rebuild + verify ----------------------------------------------
    step(8, "build_circuit_from_phase_terms() -> small_unitary_metrics() -> report_cx_and_depth() ...")
    try:
        rebuilt_circuit = linux_mcx.build_circuit_from_phase_terms(pred_terms, qubit_count)
    except Exception as exc:  # e.g. an empty phase polynomial
        print(f"    [failure] the GNN produced no routable phase polynomial: {exc}", flush=True)
        return 1

    metrics = linux_mcx.small_unitary_metrics(target_basic, rebuilt_circuit)
    if metrics is None:
        print("    [skip] equivalence check unavailable for this circuit size", flush=True)
    else:
        relative_penalty, coherent_error = metrics
        print(
            f"    relative penalty = {relative_penalty:.3e}, "
            f"coherent error = {coherent_error:.3e}",
            flush=True,
        )
        if coherent_error < 1e-8 and relative_penalty < 1e-8:
            print("    [ok] rebuilt circuit is exactly equivalent to the source", flush=True)
        else:
            print("    [warn] rebuilt circuit is not exactly equivalent to the source", flush=True)

    print("    optimized (GNN phase polynomial + Steiner routing):", flush=True)
    linux_mcx.report_cx_and_depth(rebuilt_circuit, "GNN optimized")

    # -- hybrid block solver: per-block GNN proposal -> verify -> route -------
    if HYBRID_BLOCKS:
        print(
            "\n    [hybrid] block-by-block solve: GNN proposal per phase block, local "
            "verification, Steiner routing, exact fallback",
            flush=True,
        )
        hybrid_circuit, block_report = linux_mcx.hybrid_block_solve(
            model,
            target_basic,
            featurize=extract_inference_features,
            decode=lambda net, batch, qubits: linux_mcx.predict_phase_polynomial(
                net, batch, qubits, soft_feedback=True
            ),
            use_gnn=True,
            verify=True,
            attempts=3,
            log=lambda message: print(message, flush=True),
        )
        hybrid_check = linux_mcx.small_unitary_metrics(target_basic, hybrid_circuit)
        if hybrid_check is not None:
            rel_h, coh_h = hybrid_check
            print(
                f"    [hybrid] final circuit vs source: relative={rel_h:.3e}, "
                f"coherent={coh_h:.3e} -> "
                f"{'OK' if rel_h < 1e-8 and coh_h < 1e-8 else 'FAIL'}",
                flush=True,
            )
        accepted = sum(1 for info in block_report if info["accepted"])
        print(
            f"    [hybrid] blocks={len(block_report)}, GNN-accepted={accepted}, "
            f"exact-fallback={len(block_report) - accepted}",
            flush=True,
        )
        linux_mcx.report_ftq_metrics(hybrid_circuit, "hybrid blocks")

    # -- strategy-B per-block variant policy --------------------------------
    # The trained policy decides, block by block, between the exact Clifford+T
    # ladder and the cheaper relative-phase RCCX ladder.  It is deployed with a
    # verifier-in-the-loop greedy decoder: blocks are accepted in order of the
    # model's confidence, each only if the *simulated* measurement statistics of
    # the whole circuit still match the all-exact circuit.  Every accepted mask is
    # therefore safe by construction, and greedy decoding captures far more of the
    # oracle headroom than a single threshold.
    if VARIANT_POLICY:
        print(
            "\n    [variant-policy] strategy-B per-block exact/relative choice",
            flush=True,
        )
        if not Path(VARIANT_CHECKPOINT).exists():
            print(
                "    [variant-policy] no policy checkpoint found -> skipped",
                flush=True,
            )
        else:
            runtime = load_variant_runtime()
            gen = runtime.load_generator()
            device = torch.device("cpu")
            weights = VARIANT_CHECKPOINT
            source = "supervised"
            if VARIANT_RL_CHECKPOINT.exists():
                weights = VARIANT_RL_CHECKPOINT
                source = "RL (part b)"
            print(f"    [variant-policy] using {source} weights: {weights.name}", flush=True)
            policy, ckpt = runtime.load_policy(str(weights), device)
            summary = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in ckpt.items()
                if key in ("epoch", "val_loss", "val_auc", "running_reward")
            )
            print(f"    [variant-policy] weights: {summary}", flush=True)
            rows = [
                runtime.evaluate_instance(
                    linux_mcx, gen, policy, NUM_CONTROLS, VARIANT_BLOCKS,
                    VARIANT_SEED + offset, VARIANT_H_PROB, device, decoder="greedy",
                )
                for offset in range(VARIANT_INSTANCES)
            ]

            def mean(key):
                return sum(row[key] for row in rows) / len(rows)

            thr_safe = sum(
                row["t_threshold"] if row["threshold_usable"] else row["t_all_exact"]
                for row in rows
            ) / len(rows)
            oracle_gain = mean("t_all_exact") - mean("t_oracle")
            greedy_gain = mean("t_all_exact") - mean("t_greedy")
            print(
                f"    [variant-policy] {len(rows)} instances of C^{NUM_CONTROLS}X x "
                f"{VARIANT_BLOCKS} blocks (h_prob={VARIANT_H_PROB:g}), "
                f"fragmented={sum(1 for r in rows if r['fragmented'])}/{len(rows)}",
                flush=True,
            )
            print(
                f"    [variant-policy] T-count per instance: "
                f"all-exact={mean('t_all_exact'):.2f} | "
                f"oracle={mean('t_oracle'):.2f} (saves {oracle_gain:.2f}) | "
                f"policy(greedy+safe)={mean('t_greedy'):.2f} (saves {greedy_gain:.2f}, "
                f"{greedy_gain / max(oracle_gain, 1e-9):.0%} of headroom) | "
                f"threshold+fallback={thr_safe:.2f} | "
                f"all-relative(illegal)={mean('t_all_relative'):.2f}",
                flush=True,
            )
            print(
                f"    [variant-policy] deployed mask measurement-safe="
                f"{sum(1 for r in rows if r['policy_usable']) / len(rows):.0%} | "
                f"raw threshold mask safe="
                f"{sum(1 for r in rows if r['threshold_usable']) / len(rows):.0%} | "
                f"all-relative usable="
                f"{sum(1 for r in rows if r['all_relative_usable']) / len(rows):.0%}",
                flush=True,
            )
            for row in rows[:3]:
                bits = "".join(
                    "R" if (row["mask"] >> i) & 1 else "E" for i in range(row["blocks"])
                )
                print(
                    f"      seed {row['seed']}: probs=["
                    f"{', '.join(f'{p:.2f}' for p in row['probs'])}] -> {bits} "
                    f"(E=exact, R=relative) | T greedy={row['t_greedy']} "
                    f"vs exact={row['t_all_exact']} oracle={row['t_oracle']}",
                    flush=True,
                )
            # -- large-scale MIXED-control-count stress test -----------------
            # Each of the NUM_LARGE_TESTS circuits has 30..60 blocks whose control
            # counts are drawn independently from {2..8}: a genuinely
            # heterogeneous instance.  Every block is embedded into one shared
            # data register (controls + target) with a reused ancilla pool, so
            # the reduced verifier still applies exactly -- no 2^k enumeration.
            if NUM_LARGE_TESTS > 0:
                mixed = load_mixed_stress()
                controls = list(getattr(mixed, "CONTROL_CHOICES", (2, 3, 4, 5, 6, 7, 8)))
                print(
                    f"\n    [variant-policy] mixed stress config: tests={NUM_LARGE_TESTS}, "
                    f"blocks={LARGE_MIN_BLOCKS}..{LARGE_MAX_BLOCKS}, controls={controls}, "
                    f"h_prob={LARGE_H_PROB:g}",
                    flush=True,
                )
                mixed.run_stress(
                    policy,
                    device,
                    num_tests=NUM_LARGE_TESTS,
                    min_blocks=LARGE_MIN_BLOCKS,
                    max_blocks=LARGE_MAX_BLOCKS,
                    seed=VARIANT_SEED,
                    h_prob=LARGE_H_PROB,
                    log=lambda message: print(message, flush=True),
                )

    # -- reference proposal: Steiner-route the source itself ------------------
    # This is the achievable baseline for the *same* router.  Note it must be a
    # block-preserving route (``steiner_route_phase_layers``): flattening every
    # Hadamard-separated phase block into one dictionary (``extract_phase_terms``
    # + ``build_circuit_from_phase_terms``) is only valid for single-block
    # circuits -- a Clifford+T synthesis like the Toffoli ladder contains inner
    # Hadamards, so the flattened reconstruction is not equivalent (measured
    # relative penalty ~0.35, which is exactly how this driver surfaces it).
    reference_circuit = linux_mcx.steiner_route_phase_layers(target_basic)
    reference_terms = linux_mcx.extract_phase_terms(target_basic)
    predicted_keys = {
        tuple(int(b) for b in parity) for parity in pred_terms
    }
    reference_keys = {tuple(int(b) for b in parity) for parity in reference_terms}
    overlap = predicted_keys & reference_keys
    print(
        f"    [terms] GNN={len(predicted_keys)}, reference(block-wise)="
        f"{len(reference_keys)}, parity overlap={len(overlap)} "
        f"({100.0 * len(overlap) / max(1, len(predicted_keys)):.0f}% of the proposal, "
        f"{100.0 * len(overlap) / max(1, len(reference_keys)):.0f}% of the reference)",
        flush=True,
    )
    # -- training-objective metrics ------------------------------------------
    # The dataset never asks for the *source* circuit: it asks for the phase
    # polynomial of the SIMPLIFIED circuit in the canonical (block, |phase|,
    # parity) order.  Compare the decoded rows against exactly that target.
    _, simplified_circuit, target_rows = training_target(
        linux_mcx, dg, target_basic, model.max_phase_terms
    )
    pred_rows = diag.get("rows") or []
    n_cmp = min(len(pred_rows), len(target_rows))
    bit_total = bit_hit = cls_hit = row_hit = 0
    for i in range(n_cmp):
        pred_parity, pred_cls, _pb = pred_rows[i]
        true_parity, true_cls, _tb = target_rows[i]
        bit_total += len(true_parity)
        bit_hit += sum(1 for a, b in zip(pred_parity, true_parity) if a == b)
        cls_hit += int(pred_cls == true_cls)
        row_hit += int(tuple(pred_parity) == tuple(true_parity) and pred_cls == true_cls)
    target_blocks = len({row[2] for row in target_rows})
    print(
        f"    [training-objective] rows: predicted {len(pred_rows)} vs target "
        f"{len(target_rows)} (simplified circuit) | blocks: predicted "
        f"{diag.get('n_blocks', 0)} vs target {target_blocks} | "
        f"row-exact {row_hit / max(1, n_cmp):.1%} | parity-bit "
        f"{bit_hit / max(1, bit_total):.1%} | phase-class {cls_hit / max(1, n_cmp):.1%}",
        flush=True,
    )

    # -- block-aware rebuild with the TRAINING-time scaffolding --------------
    # The proposal is a *sequence of phase blocks* of the simplified circuit, so
    # the rebuild must reuse that circuit's Clifford scaffolding (not the raw
    # source's 5 Hadamard segments).
    block_terms = diag.get("block_terms") or []
    if block_terms:
        blocked_circuit, covered, total_segments = linux_mcx.rebuild_from_block_terms(
            simplified_circuit, block_terms
        )
        blocked_check = linux_mcx.small_unitary_metrics(target_basic, blocked_circuit)
        if blocked_check is not None:
            rel_b, coh_b = blocked_check
            print(
                f"    [blocked] rebuilt {covered}/{total_segments} phase blocks from "
                f"the proposal: relative={rel_b:.3e}, coherent={coh_b:.3e} -> "
                f"{'OK' if rel_b < 1e-8 and coh_b < 1e-8 else 'FAIL'}",
                flush=True,
            )
        linux_mcx.report_ftq_metrics(blocked_circuit, "blocked rebuild")
        print(
            "    [blocked] scaffolding = simplified circuit (training-time block "
            f"structure, {total_segments} segments); compared against the source",
            flush=True,
        )
    # -- reachability of the single-block class ------------------------------
    flat_reference = linux_mcx.build_circuit_from_phase_terms(reference_terms, qubit_count)
    flat_check = linux_mcx.small_unitary_metrics(target_basic, flat_reference)
    if flat_check is not None:
        print(
            f"    [reachability] the TRUE flattened phase polynomial rebuilt in the "
            f"single-block class scores relative={flat_check[0]:.3e}: a single-block "
            "proposal can never match a multi-block source",
            flush=True,
        )
    if RELATIVE_PHASE:
        n_ancillas = max(0, NUM_CONTROLS - 2)
        rp = linux_mcx.relative_phase_metrics(
            target_basic,
            reference_circuit,
            ancillas=list(range(NUM_CONTROLS, NUM_CONTROLS + n_ancillas)),
            num_controls=NUM_CONTROLS,
        )
        if rp is None:
            print("    [relative-phase] dense check unavailable for this size", flush=True)
        else:
            print(
                f"    [relative-phase] certified diagonal-correction test on the "
                f"reference route: offdiag={rp[0]:.2e}, ||diag|-1|={rp[1]:.2e} -> "
                f"{'OK' if max(rp) <= 1e-8 else 'FAIL'}",
                flush=True,
            )
    reference_check = linux_mcx.small_unitary_metrics(target_basic, reference_circuit)
    if reference_check is not None:
        rel, coh = reference_check
        print(
            f"    [reference] routed true phase polynomial: "
            f"relative={rel:.3e}, coherent={coh:.3e} -> "
            f"{'OK' if rel < 1e-8 and coh < 1e-8 else 'FAIL'}",
            flush=True,
        )
    linux_mcx.report_ftq_metrics(reference_circuit, "reference optimized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
