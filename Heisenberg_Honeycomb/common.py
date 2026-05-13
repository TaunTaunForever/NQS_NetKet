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


def build_honeycomb_graph(num_sites: int):
    if num_sites not in SIZE_TO_EXTENT:
        raise ValueError(f"Unsupported NUM_SITES={num_sites}. Supported sizes: {sorted(SIZE_TO_EXTENT)}")
    extent = SIZE_TO_EXTENT[num_sites]
    graph = nk.graph.Honeycomb(extent=extent, pbc=True)
    return graph, extent


def make_heisenberg_hamiltonian(num_sites: int):
    graph, extent = build_honeycomb_graph(num_sites)
    hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=graph, sign_rule=True)
    return graph, extent, hi, ha
