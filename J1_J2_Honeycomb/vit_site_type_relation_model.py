from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import normal


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


def build_honeycomb_relation_matrix(graph, permutation=None):
    """
    Build relation ids tailored to SU(2)-symmetric honeycomb models.

    Relation ids:
      0 = self
      1 = nearest-neighbor (J1) bond
      2 = next-nearest-neighbor (J2) bond
      3 = graph-distance-3 on the nearest-neighbor honeycomb backbone
      4 = farther same-sublattice relation
      5 = farther opposite-sublattice relation
    """
    nn_adjacency = _nearest_neighbor_adjacency(graph)
    n_sites = graph.n_nodes
    nn_dist = _all_pairs_shortest_path_distances(nn_adjacency)
    site_types = np.asarray(build_bipartite_site_type_ids(graph), dtype=np.int32)

    edge_relation = {}
    for i, j, color in _edge_list_with_colors(graph):
        rel = 1 if color == 0 else 2
        edge_relation[(i, j)] = rel
        edge_relation[(j, i)] = rel

    relation = np.full((n_sites, n_sites), 5, dtype=np.int32)
    np.fill_diagonal(relation, 0)

    for i in range(n_sites):
        for j in range(n_sites):
            if i == j:
                relation[i, j] = 0
                continue
            if (i, j) in edge_relation:
                relation[i, j] = edge_relation[(i, j)]
                continue

            distance = int(nn_dist[i, j])
            if distance == 3:
                relation[i, j] = 3
            else:
                relation[i, j] = 4 if int(site_types[i]) == int(site_types[j]) else 5

    if permutation is not None:
        perm = np.asarray(permutation, dtype=np.int32)
        relation = relation[np.ix_(perm, perm)]

    return tuple(tuple(int(x) for x in row) for row in relation)


def site_relation_to_patch_relation_expanded(site_relation, patch_size):
    site_relation = np.asarray(site_relation, dtype=np.int32)
    n_sites = site_relation.shape[0]

    if n_sites % patch_size != 0:
        raise ValueError(
            f"n_sites={n_sites} must be divisible by patch_size={patch_size}"
        )

    n_patches = n_sites // patch_size
    patch_relation = np.full((n_patches, n_patches), 5, dtype=np.int32)
    np.fill_diagonal(patch_relation, 0)

    priority = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 0: 5}

    for a in range(n_patches):
        sites_a = range(a * patch_size, (a + 1) * patch_size)
        for b in range(n_patches):
            if a == b:
                patch_relation[a, b] = 0
                continue
            sites_b = range(b * patch_size, (b + 1) * patch_size)
            rels = [int(site_relation[i, j]) for i in sites_a for j in sites_b]
            patch_relation[a, b] = min(rels, key=lambda rel: priority.get(rel, rel + 100))

    return tuple(tuple(int(x) for x in row) for row in patch_relation)


def site_type_ids_to_patch_type_ids(site_type_ids, patch_size):
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


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.out_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        return x


def log_cosh(z):
    z_abs = jnp.where(jnp.real(z) >= 0, z, -z)
    return z_abs + jnp.log1p(jnp.exp(-2 * z_abs)) - jnp.log(2.0)


class OutputHead(nn.Module):
    d_model: int
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False

    @nn.compact
    def __call__(self, x):
        z = x.sum(axis=1)

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


class SiteTypeRelationTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    relation_matrix: Tuple[Tuple[int, ...], ...]
    num_relation_types: int
    data_type: jnp.dtype = jnp.float64
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = RelationAwareSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            relation_matrix=self.relation_matrix,
            num_relation_types=self.num_relation_types,
            data_type=self.data_type,
            name="RelationAwareSelfAttention",
        )(y)
        x = x + self.residual_scale * y

        y = nn.LayerNorm(dtype=self.data_type, param_dtype=self.data_type)(x)
        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)
        x = x + self.residual_scale * y
        return x


class HoneycombSiteTypeRelationViT(nn.Module):
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
                relation_matrix = site_relation_to_patch_relation_expanded(relation_matrix, self.patch_size)

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

        num_relation_types = max(max(row) for row in relation_matrix) + 1
        for layer in range(self.num_layers):
            x = SiteTypeRelationTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                relation_matrix=relation_matrix,
                num_relation_types=num_relation_types,
                data_type=self.data_type,
                residual_scale=self.residual_scale,
                name=f"SiteTypeRelationTransformerBlock_{layer}",
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
