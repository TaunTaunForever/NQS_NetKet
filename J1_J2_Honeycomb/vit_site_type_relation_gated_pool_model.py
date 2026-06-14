from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import normal

from vit_site_type_relation_model import (
    MLP,
    log_cosh,
    site_relation_to_patch_relation_expanded,
    site_type_ids_to_patch_type_ids,
)


def _edge_list_with_colors(graph):
    edges = []
    for edge in graph.edges(return_color=True):
        if len(edge) == 3:
            i, j, color = edge
        else:
            i, j = edge
            color = 0
        edges.append((int(i), int(j), int(color)))
    return edges


def _nearest_neighbor_adjacency(graph):
    n_sites = graph.n_nodes
    adjacency = [[] for _ in range(n_sites)]
    for i, j, color in _edge_list_with_colors(graph):
        if color != 0:
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def build_bipartite_site_type_ids(graph, permutation=None):
    adjacency = _nearest_neighbor_adjacency(graph)
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
                    raise ValueError("Nearest-neighbor honeycomb backbone is not bipartite.")

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


def _graph_metadata(graph):
    basis_coords = np.asarray(graph.basis_coords, dtype=int)
    basis_vectors = np.asarray(graph.basis_vectors, dtype=float)
    site_offsets = np.asarray(graph.site_offsets, dtype=float)
    extent = np.asarray(graph.extent, dtype=int)
    positions = np.asarray(graph.positions, dtype=float)
    sublattice = basis_coords[:, 2]
    return {
        "basis_coords": basis_coords,
        "basis_vectors": basis_vectors,
        "site_offsets": site_offsets,
        "extent": extent,
        "positions": positions,
        "sublattice": sublattice,
    }


def _minimal_image_displacement(meta, i: int, j: int) -> np.ndarray:
    basis_coords = meta["basis_coords"]
    basis_vectors = meta["basis_vectors"]
    site_offsets = meta["site_offsets"]
    extent = meta["extent"]

    cell_delta = basis_coords[j, :2] - basis_coords[i, :2]
    offset_delta = site_offsets[basis_coords[j, 2]] - site_offsets[basis_coords[i, 2]]

    best_vec = None
    best_norm = None
    for s1 in (-extent[0], 0, extent[0]):
        for s2 in (-extent[1], 0, extent[1]):
            shifted = cell_delta + np.array([s1, s2], dtype=int)
            vec = shifted @ basis_vectors + offset_delta
            norm = float(np.dot(vec, vec))
            if best_norm is None or norm < best_norm - 1e-12:
                best_norm = norm
                best_vec = vec
    return best_vec


def _canonical_bond_vector(vec: np.ndarray) -> tuple[float, float]:
    if vec[1] < -1e-10 or (abs(vec[1]) < 1e-10 and vec[0] < 0):
        vec = -vec
    return (float(np.round(vec[0], 8)), float(np.round(vec[1], 8)))


def build_honeycomb_bond_oriented_relation_matrix(graph, permutation=None):
    """
    Richer relation ids for the honeycomb J1-J2 model.

    Relation ids are assigned dynamically as:
      0                  = self
      1..n_j1_orient     = J1 bond orientation classes
      next block         = J2 bond orientation classes
      next block         = graph-distance-3 on the J1 backbone
      next block         = farther same-sublattice relation
      final block        = farther opposite-sublattice relation
    """
    meta = _graph_metadata(graph)
    nn_adjacency = _nearest_neighbor_adjacency(graph)
    nn_dist = _all_pairs_shortest_path_distances(nn_adjacency)
    site_types = np.asarray(build_bipartite_site_type_ids(graph), dtype=np.int32)

    edge_buckets = {0: [], 1: []}
    for i, j, color in _edge_list_with_colors(graph):
        if color in edge_buckets:
            edge_buckets[color].append((i, j))
    edge_sets = {
        color: {tuple(sorted(bond)) for bond in bonds}
        for color, bonds in edge_buckets.items()
    }

    orientation_maps = {}
    next_relation_id = 1
    for color in (0, 1):
        unique_vectors = sorted(
            {
                _canonical_bond_vector(_minimal_image_displacement(meta, i, j))
                for i, j in edge_buckets[color]
            }
        )
        orientation_maps[color] = {
            vec: next_relation_id + idx for idx, vec in enumerate(unique_vectors)
        }
        next_relation_id += len(unique_vectors)

    relation_distance_3 = next_relation_id
    relation_far_same = next_relation_id + 1
    relation_far_opposite = next_relation_id + 2

    n_sites = graph.n_nodes
    relation = np.full((n_sites, n_sites), relation_far_opposite, dtype=np.int32)
    np.fill_diagonal(relation, 0)

    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                relation[i, j] = 0
                continue

            vec = _canonical_bond_vector(_minimal_image_displacement(meta, i, j))
            bond = tuple(sorted((i, j)))
            if bond in edge_sets[0]:
                relation[i, j] = orientation_maps[0][vec]
                continue
            if bond in edge_sets[1]:
                relation[i, j] = orientation_maps[1][vec]
                continue

            if int(nn_dist[i, j]) == 3:
                relation[i, j] = relation_distance_3
            else:
                relation[i, j] = (
                    relation_far_same
                    if int(site_types[i]) == int(site_types[j])
                    else relation_far_opposite
                )

    if permutation is not None:
        perm = np.asarray(permutation, dtype=np.int32)
        relation = relation[np.ix_(perm, perm)]

    return tuple(tuple(int(x) for x in row) for row in relation)


class AttentionPooling(nn.Module):
    embed_dim: int
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        query = self.param(
            "pool_query",
            normal(stddev=0.02),
            (self.embed_dim,),
            self.data_type,
        )
        scores = jnp.einsum("btd,d->bt", x, query)
        scores = scores / jnp.sqrt(jnp.asarray(self.embed_dim, dtype=self.data_type))
        weights = nn.softmax(scores, axis=1)
        pooled = jnp.einsum("bt,btd->bd", weights, x)
        return pooled


class AttentionPooledOutputHead(nn.Module):
    d_model: int
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, x):
        z = AttentionPooling(
            embed_dim=self.d_model,
            data_type=self.data_type,
            name="AttentionPooling",
        )(x)

        z = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="out_layer_norm",
        )(z)

        out_real = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_real",
        )(z)
        out_real = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_real",
        )(out_real)

        out_imag = nn.Dense(
            self.d_model,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="output_layer_imag",
        )(z)
        out_imag = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="norm_imag",
        )(out_imag)

        if self.zero_imag_branch:
            out_imag = jnp.zeros_like(out_imag)

        out = out_real + 1j * out_imag
        return jnp.sum(log_cosh(out), axis=-1)


class RelationAwareSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int
    relation_matrix: Tuple[Tuple[int, ...], ...]
    num_relation_types: int
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        batch_size, n_tokens, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )

        relation = jnp.asarray(self.relation_matrix, dtype=jnp.int32)
        if relation.shape != (n_tokens, n_tokens):
            raise ValueError(
                f"relation_matrix shape {relation.shape} does not match token shape {(n_tokens, n_tokens)}."
            )

        head_dim = self.embed_dim // self.num_heads
        qkv = nn.Dense(
            3 * self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(batch_size, n_tokens, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, n_tokens, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, n_tokens, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        logits = jnp.einsum("bhid,bhjd->bhij", q, k)
        logits = logits / jnp.sqrt(jnp.asarray(head_dim, dtype=self.data_type))

        relation_bias_table = self.param(
            "relation_attention_bias",
            nn.initializers.zeros,
            (self.num_heads, self.num_relation_types),
            self.data_type,
        )
        relation_bias = relation_bias_table[:, relation]
        logits = logits + relation_bias[None, :, :, :]

        attn = nn.softmax(logits, axis=-1)
        y = jnp.einsum("bhij,bhjd->bhid", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, n_tokens, embed_dim)

        y = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="out",
        )(y)
        return y


class GatedSiteTypeRelationTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    relation_matrix: Tuple[Tuple[int, ...], ...]
    num_relation_types: int
    data_type: jnp.dtype = jnp.float64
    gate_init: float = 0.8

    @nn.compact
    def __call__(self, x):
        gate_init = float(np.log(self.gate_init / (1.0 - self.gate_init)))

        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = RelationAwareSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            relation_matrix=self.relation_matrix,
            num_relation_types=self.num_relation_types,
            data_type=self.data_type,
            name="RelationAwareSelfAttention",
        )(y)
        attn_gate = jax.nn.sigmoid(
            self.param(
                "attn_gate_logit",
                lambda rng, shape, dtype: jnp.full(shape, gate_init, dtype=dtype),
                (1,),
                self.data_type,
            )
        )[0]
        x = x + attn_gate * y

        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)
        mlp_gate = jax.nn.sigmoid(
            self.param(
                "mlp_gate_logit",
                lambda rng, shape, dtype: jnp.full(shape, gate_init, dtype=dtype),
                (1,),
                self.data_type,
            )
        )[0]
        x = x + mlp_gate * y
        return x


class HoneycombSiteTypeRelationGatedPoolViT(nn.Module):
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
    gate_init: float = 0.8

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        batch_size, n_sites = x.shape

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        if n_sites % self.patch_size != 0:
            raise ValueError(
                f"Number of sites N={n_sites} must be divisible by patch_size={self.patch_size}"
            )

        n_patches = n_sites // self.patch_size
        x = x.reshape(batch_size, n_patches, self.patch_size).astype(self.data_type)

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
        relation_matrix = self.relation_matrix
        if self.patch_size > 1:
            if len(site_type_ids) == n_sites:
                site_type_ids = site_type_ids_to_patch_type_ids(site_type_ids, self.patch_size)
            rel_arr = jnp.asarray(relation_matrix, dtype=jnp.int32)
            if rel_arr.shape == (n_sites, n_sites):
                relation_matrix = site_relation_to_patch_relation_expanded(
                    relation_matrix, self.patch_size
                )

        if len(site_type_ids) != n_patches:
            raise ValueError(
                f"site_type_ids length {len(site_type_ids)} does not match number of tokens {n_patches}."
            )

        num_site_types = max(int(v) for v in site_type_ids) + 1
        site_type_emb = nn.Embed(
            num_embeddings=num_site_types,
            features=self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            embedding_init=normal(stddev=0.02),
            name="site_type_embedding",
        )(jnp.asarray(site_type_ids, dtype=jnp.int32))
        x = x + site_type_emb[None, :, :]

        num_relation_types = max(max(row) for row in relation_matrix) + 1
        for layer in range(self.num_layers):
            x = GatedSiteTypeRelationTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                relation_matrix=relation_matrix,
                num_relation_types=num_relation_types,
                data_type=self.data_type,
                gate_init=self.gate_init,
                name=f"GatedSiteTypeRelationTransformerBlock_{layer}",
            )(x)

        log_psi = AttentionPooledOutputHead(
            d_model=self.embed_dim,
            data_type=self.data_type,
            zero_imag_branch=self.zero_imag_branch,
            name="AttentionPooledOutputHead",
        )(x)

        if self.learn_phase:
            return log_psi
        return jnp.real(log_psi)
