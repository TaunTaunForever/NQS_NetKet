from __future__ import annotations

import netket as nk


SIZE_TO_EXTENT = {
    4: [2, 1],
    8: [2, 2],
    12: [2, 3],
    18: [3, 3],
    24: [3, 4],
    32: [4, 4],
    50: [5, 5],
    72: [6, 6],
    98: [7, 7],
    128: [8, 8],
}


def build_honeycomb_graph(num_sites: int, *, max_neighbor_order: int = 2):
    if num_sites not in SIZE_TO_EXTENT:
        raise ValueError(f"Unsupported NUM_SITES={num_sites}. Supported sizes: {sorted(SIZE_TO_EXTENT)}")
    if max_neighbor_order > 1 and num_sites == 4:
        raise ValueError(
            "J1-J2 Honeycomb runs do not support the 4-site cluster in this setup because "
            "this very small Honeycomb cluster with second-neighbor edges can generate "
            "self-referential lattice edges in NetKet."
        )
    extent = SIZE_TO_EXTENT[num_sites]
    graph = nk.graph.Honeycomb(extent=extent, max_neighbor_order=max_neighbor_order, pbc=True)
    return graph, extent


def make_heisenberg_hamiltonian(num_sites: int, *, j1: float, j2: float):
    graph, extent = build_honeycomb_graph(num_sites, max_neighbor_order=2)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
    sign_rule = [True, False] if abs(j2) < 1e-12 else [False, False]
    ha = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[j1, j2], sign_rule=sign_rule)
    return graph, extent, hi, ha
