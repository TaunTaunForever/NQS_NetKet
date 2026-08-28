from __future__ import annotations

from collections.abc import Sequence

import netket as nk


def make_tfi(extent: Sequence[int], g: float):
    """Build the paper's ferromagnetic 2D transverse-field Ising Hamiltonian.

    NetKet's Ising convention is `-h sum X + J sum ZZ`, so the paper's
    `-sum ZZ - g sum X` convention corresponds to `h=g, J=-1`.
    """

    graph = nk.graph.Grid(extent=tuple(extent), pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    hamiltonian = nk.operator.Ising(
        hilbert=hilbert,
        graph=graph,
        h=float(g),
        J=-1.0,
        dtype=complex,
    )
    return hilbert, graph, hamiltonian


def make_square_j1j2(length: int, j1: float = 1.0, j2: float = 0.5):
    """Build the square-lattice J1-J2 Heisenberg benchmark from the paper."""

    def site(x: int, y: int) -> int:
        return (x % length) * length + (y % length)

    edges: list[tuple[int, int, int]] = []
    for x in range(length):
        for y in range(length):
            i = site(x, y)
            edges.append((i, site(x + 1, y), 0))
            edges.append((i, site(x, y + 1), 0))
            edges.append((i, site(x + 1, y + 1), 1))
            edges.append((i, site(x + 1, y - 1), 1))

    graph = nk.graph.Graph(edges=edges, n_nodes=length * length)
    hilbert = nk.hilbert.Spin(s=0.5, N=length * length)
    hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, graph=graph, J=[j1, j2])
    return hilbert, graph, hamiltonian
