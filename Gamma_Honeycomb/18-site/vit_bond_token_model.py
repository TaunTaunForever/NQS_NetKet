from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import normal

from vit_model import (
    OutputHead,
    BondAwareTransformerBlock,
    build_kitaev_relation_matrix,
)


def build_edge_index_and_color(graph, permutation=None, color_to_relation=None):
    """
    Build edge endpoint and edge-relation tuples for the bond-token readout.

    The returned edge indices are in the model token/site ordering. This first
    version assumes PATCH_SIZE=1, so each token is one physical site.
    """
    import numpy as np

    n_sites = graph.n_nodes

    if color_to_relation is None:
        colors = sorted({int(c) for *_ij, c in graph.edges(return_color=True)})
        color_to_relation = {color: idx + 1 for idx, color in enumerate(colors)}

    if permutation is None:
        original_site_to_token = np.arange(n_sites, dtype=np.int32)
    else:
        perm = np.asarray(permutation, dtype=np.int32)
        original_site_to_token = np.empty_like(perm)
        for token_idx, site_idx in enumerate(perm):
            original_site_to_token[int(site_idx)] = int(token_idx)

    edges = []
    edge_relations = []
    for i, j, color in graph.edges(return_color=True):
        ti = int(original_site_to_token[int(i)])
        tj = int(original_site_to_token[int(j)])
        edges.append((ti, tj))
        edge_relations.append(int(color_to_relation[int(color)]))

    return tuple(tuple(int(x) for x in edge) for edge in edges), tuple(edge_relations)


class BondTokenReadoutHead(nn.Module):
    """
    Global OutputHead plus an explicit learned sum over physical graph edges.

    logψ = global_score(tokens) + scale * sum_edges MLP([h_i, h_j, h_i+h_j, h_i*h_j, edge_emb])
    """
    embed_dim: int
    edge_indices: Tuple[Tuple[int, int], ...]
    edge_relations: Tuple[int, ...]
    num_relation_types: int
    learn_phase: bool
    hidden_dim: int
    data_type: jnp.dtype = jnp.float64
    zero_imag_branch: bool = False
    use_global_head: bool = True
    bond_scale_init: float = 0.1

    @nn.compact
    def __call__(self, tokens):
        B, T, D = tokens.shape

        edge_idx = jnp.asarray(self.edge_indices, dtype=jnp.int32)
        edge_rel = jnp.asarray(self.edge_relations, dtype=jnp.int32)

        token_i = tokens[:, edge_idx[:, 0], :]
        token_j = tokens[:, edge_idx[:, 1], :]

        edge_emb = nn.Embed(
            num_embeddings=self.num_relation_types,
            features=self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            embedding_init=normal(stddev=0.02),
            name="edge_relation_embedding",
        )(edge_rel)
        edge_emb = jnp.broadcast_to(edge_emb[None, :, :], (B, edge_idx.shape[0], self.embed_dim))

        pair_sum = token_i + token_j
        pair_prod = token_i * token_j
        features = jnp.concatenate([token_i, token_j, pair_sum, pair_prod, edge_emb], axis=-1)

        h = nn.Dense(
            self.hidden_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.xavier_uniform(),
            name="bond_readout_dense",
        )(features)
        h = nn.gelu(h)

        bond_real = nn.Dense(
            1,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.zeros,
            name="bond_readout_real",
        )(h).squeeze(-1)

        bond_imag = nn.Dense(
            1,
            dtype=self.data_type,
            param_dtype=self.data_type,
            kernel_init=nn.initializers.zeros,
            name="bond_readout_imag",
        )(h).squeeze(-1)

        if self.zero_imag_branch:
            bond_imag = jnp.zeros_like(bond_imag)

        scale = self.param(
            "bond_readout_scale",
            lambda key, shape, dtype: jnp.asarray(self.bond_scale_init, dtype=dtype),
            (),
            self.data_type,
        )
        bond_score = scale * jnp.sum(bond_real + 1j * bond_imag, axis=-1)

        if self.use_global_head:
            global_score = OutputHead(
                d_model=self.embed_dim,
                data_type=self.data_type,
                zero_imag_branch=self.zero_imag_branch,
                name="global_OutputHead",
            )(tokens)
            log_psi = global_score + bond_score
        else:
            log_psi = bond_score

        return log_psi if self.learn_phase else jnp.real(log_psi)


class BondAwareBondTokenReadoutHoneycombViT(nn.Module):
    """
    Bond-aware Honeycomb ViT with explicit bond-token readout.

    This model is designed as a direct extension of the bond-aware ViT:
      - site tokens
      - bond-aware attention using relation_matrix
      - final global head plus sum over physical graph-edge scores

    First implementation assumes PATCH_SIZE=1.
    """
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    relation_matrix: Tuple[Tuple[int, ...], ...]
    edge_indices: Tuple[Tuple[int, int], ...]
    edge_relations: Tuple[int, ...]
    num_relation_types: int
    zero_imag_branch: bool = False
    data_type: jnp.dtype = jnp.float64
    permutation: Optional[Tuple[int, ...]] = None
    residual_scale: float = 0.8
    use_global_head: bool = True
    bond_readout_hidden_dim: Optional[int] = None
    bond_scale_init: float = 0.1

    @nn.compact
    def __call__(self, sigma):
        x = jnp.atleast_2d(sigma)
        B, N = x.shape

        if self.patch_size != 1:
            raise ValueError(
                "BondAwareBondTokenReadoutHoneycombViT currently requires PATCH_SIZE=1."
            )

        if self.permutation is not None:
            perm_arr = jnp.asarray(self.permutation, dtype=jnp.int32)
            x = x[:, perm_arr]

        x = x.reshape(B, N, 1).astype(self.data_type)

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
            (N, self.embed_dim),
            self.data_type,
        )
        x = x + pos_emb[None, :, :]

        for layer in range(self.num_layers):
            x = BondAwareTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                relation_matrix=self.relation_matrix,
                data_type=self.data_type,
                residual_scale=self.residual_scale,
                name=f"BondAwareTransformerBlock_{layer}",
            )(x)

        hidden_dim = self.bond_readout_hidden_dim or self.mlp_hidden_dim
        return BondTokenReadoutHead(
            embed_dim=self.embed_dim,
            edge_indices=self.edge_indices,
            edge_relations=self.edge_relations,
            num_relation_types=self.num_relation_types,
            learn_phase=self.learn_phase,
            hidden_dim=hidden_dim,
            data_type=self.data_type,
            zero_imag_branch=self.zero_imag_branch,
            use_global_head=self.use_global_head,
            bond_scale_init=self.bond_scale_init,
            name="BondTokenReadoutHead",
        )(x)
