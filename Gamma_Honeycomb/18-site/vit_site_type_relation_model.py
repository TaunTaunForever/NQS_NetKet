from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import normal

from vit_model import BondAwareTransformerBlock, OutputHead


def _adjacency_and_edge_color(graph):
    n_sites = graph.n_nodes
    adjacency = [[] for _ in range(n_sites)]
    edge_color = {}

    for i, j, color in graph.edges(return_color=True):
        i = int(i)
        j = int(j)
        color = int(color)
        adjacency[i].append(j)
        adjacency[j].append(i)
        edge_color[(i, j)] = color
        edge_color[(j, i)] = color

    return adjacency, edge_color


def build_bipartite_site_type_ids(graph, permutation=None):
    """
    Build fixed site-type ids from the honeycomb bipartite structure.

    The default output is two site types:
      0 = one sublattice
      1 = the other sublattice
    """
    adjacency, _edge_color = _adjacency_and_edge_color(graph)
    n_sites = graph.n_nodes
    site_types = np.full(n_sites, -1, dtype=np.int32)

    for root in range(n_sites):
        if site_types[root] != -1:
            continue
        site_types[root] = 0
        queue = deque([root])

        while queue:
            node = queue.popleft()
            for nbr in adjacency[node]:
                expected = 1 - int(site_types[node])
                if site_types[nbr] == -1:
                    site_types[nbr] = expected
                    queue.append(nbr)
                elif int(site_types[nbr]) != expected:
                    raise ValueError("Graph is not bipartite; cannot build site-type ids.")

    if permutation is not None:
        perm = np.asarray(permutation, dtype=np.int32)
        site_types = site_types[perm]

    return tuple(int(x) for x in site_types.tolist())


def _all_pairs_shortest_path_distances(adjacency):
    n_sites = len(adjacency)
    dist = np.full((n_sites, n_sites), n_sites + 1, dtype=np.int32)

    for src in range(n_sites):
        dist[src, src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            next_dist = int(dist[src, node]) + 1
            for nbr in adjacency[node]:
                if next_dist < dist[src, nbr]:
                    dist[src, nbr] = next_dist
                    queue.append(nbr)

    return dist


def build_extended_kitaev_relation_matrix(graph, permutation=None, color_to_relation=None):
    """
    Build a richer relation matrix than the original bond-aware model.

    Relation ids:
      0 = self
      1 = x-colored nearest-neighbor bond
      2 = y-colored nearest-neighbor bond
      3 = z-colored nearest-neighbor bond
      4 = graph-distance-2 relation
      5 = graph-distance-3 relation
      6 = farther same-sublattice relation
      7 = farther opposite-sublattice relation
    """
    adjacency, edge_color = _adjacency_and_edge_color(graph)
    n_sites = graph.n_nodes
    dist = _all_pairs_shortest_path_distances(adjacency)

    if color_to_relation is None:
        color_to_relation = {
            0: 1,
            1: 2,
            2: 3,
        }

    site_types = np.asarray(build_bipartite_site_type_ids(graph), dtype=np.int32)
    relation = np.full((n_sites, n_sites), 7, dtype=np.int32)
    np.fill_diagonal(relation, 0)

    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                relation[i, j] = 0
                continue

            distance = int(dist[i, j])
            if distance == 1:
                relation[i, j] = int(color_to_relation[edge_color[(i, j)]])
            elif distance == 2:
                relation[i, j] = 4
            elif distance == 3:
                relation[i, j] = 5
            else:
                relation[i, j] = 6 if int(site_types[i]) == int(site_types[j]) else 7

    if permutation is not None:
        perm = np.asarray(permutation, dtype=np.int32)
        relation = relation[np.ix_(perm, perm)]

    return tuple(tuple(int(x) for x in row) for row in relation)


def site_relation_to_patch_relation_expanded(site_relation, patch_size):
    """
    Convert a site-level extended relation matrix into a patch-level matrix.

    For mixed pairs inside a patch-patch block, prefer the most local relation:
      x/y/z bond > distance-2 > distance-3 > far same/opposite-sublattice
    """
    site_relation = np.asarray(site_relation, dtype=np.int32)
    n_sites = site_relation.shape[0]

    if n_sites % patch_size != 0:
        raise ValueError(
            f"n_sites={n_sites} must be divisible by patch_size={patch_size}"
        )

    n_patches = n_sites // patch_size
    patch_relation = np.full((n_patches, n_patches), 7, dtype=np.int32)
    np.fill_diagonal(patch_relation, 0)

    priority = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 0: 7}

    for a in range(n_patches):
        sites_a = range(a * patch_size, (a + 1) * patch_size)
        for b in range(n_patches):
            if a == b:
                patch_relation[a, b] = 0
                continue
            sites_b = range(b * patch_size, (b + 1) * patch_size)
            rels = [int(site_relation[i, j]) for i in sites_a for j in sites_b]
            best_rel = min(rels, key=lambda rel: priority.get(rel, rel + 100))
            patch_relation[a, b] = best_rel

    return tuple(tuple(int(x) for x in row) for row in patch_relation)


def site_type_ids_to_patch_type_ids(site_type_ids, patch_size):
    """
    Collapse site-type ids into patch-token type ids.

    For PATCH_SIZE=1 this returns the original sublattice ids.
    For larger patches, each unique within-patch site-type pattern gets its own
    fixed token type id.
    """
    site_type_ids = list(int(x) for x in site_type_ids)
    n_sites = len(site_type_ids)

    if n_sites % patch_size != 0:
        raise ValueError(
            f"len(site_type_ids)={n_sites} must be divisible by patch_size={patch_size}"
        )

    patch_type_lookup = {}
    patch_type_ids = []
    next_id = 0

    for patch_start in range(0, n_sites, patch_size):
        pattern = tuple(site_type_ids[patch_start : patch_start + patch_size])
        if pattern not in patch_type_lookup:
            patch_type_lookup[pattern] = next_id
            next_id += 1
        patch_type_ids.append(patch_type_lookup[pattern])

    return tuple(int(x) for x in patch_type_ids)


class KitaevSiteTypeRelationHoneycombViT(nn.Module):
    """
    Honeycomb ViT with:
      - fixed site-type embeddings
      - expanded relation classes in bond-aware attention
    """
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    relation_matrix: Tuple[Tuple[int, ...], ...]
    site_type_ids: Tuple[int, ...]
    zero_imag_branch: bool = False

    data_type: jnp.dtype = jnp.float64
    permutation: Optional[Tuple[int, ...]] = None
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        B, N = x.shape

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        if N % self.patch_size != 0:
            raise ValueError(
                f"Number of sites N={N} must be divisible by patch_size={self.patch_size}"
            )

        n_patches = N // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size).astype(self.data_type)

        x = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.xavier_uniform(),
            name="patch_embed",
        )(x)

        pos_emb = self.param(
            "pos_emb",
            normal(stddev=0.02),
            (n_patches, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb[None, :, :]

        site_type_ids = self.site_type_ids
        if len(site_type_ids) != n_patches:
            raise ValueError(
                f"site_type_ids length {len(site_type_ids)} does not match number of tokens {n_patches}."
            )
        num_site_types = max(int(x) for x in site_type_ids) + 1
        site_type_emb = nn.Embed(
            num_embeddings=num_site_types,
            features=self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            embedding_init=normal(stddev=0.02),
            name="site_type_embedding",
        )(jnp.asarray(site_type_ids, dtype=jnp.int32))
        x = x + site_type_emb[None, :, :]

        relation_matrix = self.relation_matrix
        if self.patch_size > 1:
            rel_arr = jnp.asarray(relation_matrix, dtype=jnp.int32)
            if rel_arr.shape == (N, N):
                relation_matrix = site_relation_to_patch_relation_expanded(
                    relation_matrix,
                    self.patch_size,
                )

        for layer in range(self.num_layers):
            x = BondAwareTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                relation_matrix=relation_matrix,
                data_type=self.data_type,
                residual_scale=self.residual_scale,
                name=f"BondAwareTransformerBlock_{layer}",
            )(x)

        log_psi = OutputHead(
            d_model=self.embed_dim,
            data_type=self.data_type,
            zero_imag_branch=self.zero_imag_branch,
            name="OutputHead",
        )(x)

        if self.learn_phase:
            return log_psi
        return jnp.real(log_psi)
