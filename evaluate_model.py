"""
Evaluate the trained phase-gadget GNN on the C^4X (5-qubit) stress test.

The file is a driver only: ``Linux-MCX`` is never modified and every
algorithmic step goes through one of its existing functions
(``PhaseGadgetGNN``, ``make_mcx_source``, ``report_cx_and_depth``,
``predict_phase_polynomial``, ``build_circuit_from_phase_terms``,
``small_unitary_metrics``, ``print_phase_terms``).

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
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_gnn_large.pt"

NUM_CONTROLS = 4                     # C^4X -> 5 qubits
SUPPORT_WIDTH = 16                   # = Linux-MCX MAX_QUBITS (support-mask padding)
NODE_FEATURE_WIDTH = 11 + SUPPORT_WIDTH   # 27, the checkpoint's node_dim
TOTAL_STEPS = 8


def step(index: int, message: str) -> None:
    print(f"[{index}/{TOTAL_STEPS}] {message}", flush=True)


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


def main() -> int:
    # -- 1. imports --------------------------------------------------------
    step(1, "Import torch / pyzx as zx / Linux-MCX ...")
    linux_mcx = load_linux_mcx()
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
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("    weights restored, model in eval mode", flush=True)

    # -- 5. source circuit ------------------------------------------------
    step(5, f"Linux-MCX.make_mcx_source(num_controls={NUM_CONTROLS}) ...")
    target_basic = linux_mcx.make_mcx_source(num_controls=NUM_CONTROLS)
    qubit_count = target_basic.qubits
    print(f"    source circuit: C^{NUM_CONTROLS}X on {qubit_count} qubits", flush=True)
    print("    baseline (Qiskit no-ancilla synthesis):", flush=True)
    linux_mcx.report_cx_and_depth(target_basic, "source C^4X")

    # -- 6. ZX simplification + inference features -------------------------
    step(6, "to_graph() -> zx.simplify.clifford_simp() -> extract_inference_features() ...")
    graph = target_basic.to_graph()
    vertices_before = len(list(graph.vertices()))
    zx.simplify.clifford_simp(graph)
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
        f"    ZX graph: {vertices_before} vertices before clifford_simp -> "
        f"{vertices_after} after (topology used by the GNN, no re-extraction)",
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
    step(7, "Linux-MCX.predict_phase_polynomial(model, data, qubit_count) ...")
    pred_terms = linux_mcx.predict_phase_polynomial(model, data, qubit_count)
    print(f"    GNN proposed {len(pred_terms)} phase terms", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
