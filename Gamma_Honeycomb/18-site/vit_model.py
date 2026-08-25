from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import normal


class MLP(nn.Module):
    hidden_dim: int
    out_dim: int
    data_type: jnp.dtype

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


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    data_type: jnp.dtype
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            deterministic=True,
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(y)
        x = x + self.residual_scale * y

        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)
        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)
        x = x + self.residual_scale * y
        return x


def log_cosh(z):
    z_abs = jnp.where(jnp.real(z) >= 0, z, -z)
    return z_abs + jnp.log1p(jnp.exp(-2 * z_abs)) - jnp.log(2.0)


class OutputHead(nn.Module):
    d_model: int
    data_type: jnp.dtype
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


class HoneycombPatchViT(nn.Module):
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    zero_imag_branch: bool = False

    data_type: jnp.dtype = jnp.float64
    permutation: Optional[Tuple[int, ...]] = None

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

        for layer in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                data_type=self.data_type,
                name=f"TransformerBlock_{layer}",
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


# ============================================================
# Kitaev bond-aware attention extensions
# ============================================================

def build_kitaev_relation_matrix(graph, permutation=None, color_to_relation=None):
    """
    Build an integer relation matrix for Kitaev bond-aware attention.

    Relation ids by default:
      0 = self
      1 = x-colored bond / color 0
      2 = y-colored bond / color 1
      3 = z-colored bond / color 2
      4 = non-neighbor

    Parameters
    ----------
    graph:
        NetKet graph object. Must support graph.edges(return_color=True).
    permutation:
        Optional site ordering applied before the model sees the input.
        If provided, the relation matrix is permuted into the model token order.
    color_to_relation:
        Optional dict mapping graph edge colors to relation ids.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Static relation matrix suitable for passing into
        KitaevBondAwareHoneycombViT.
    """
    import numpy as np

    n_sites = graph.n_nodes
    relation = np.full((n_sites, n_sites), 4, dtype=np.int32)
    np.fill_diagonal(relation, 0)

    if color_to_relation is None:
        color_to_relation = {
            0: 1,
            1: 2,
            2: 3,
        }

    for i, j, color in graph.edges(return_color=True):
        rel = color_to_relation[int(color)]
        relation[int(i), int(j)] = rel
        relation[int(j), int(i)] = rel

    if permutation is not None:
        perm = np.asarray(permutation, dtype=np.int32)
        relation = relation[np.ix_(perm, perm)]

    return tuple(tuple(int(x) for x in row) for row in relation)


def site_relation_to_patch_relation(site_relation, patch_size):
    """
    Convert a site-level relation matrix into a patch-level relation matrix.

    This is useful if you later want to use PATCH_SIZE > 1. For the first
    bond-aware Kitaev tests, PATCH_SIZE=1 is recommended, because each token
    then corresponds directly to one physical site.
    """
    import numpy as np

    site_relation = np.asarray(site_relation, dtype=np.int32)
    n_sites = site_relation.shape[0]

    if n_sites % patch_size != 0:
        raise ValueError(
            f"n_sites={n_sites} must be divisible by patch_size={patch_size}"
        )

    n_patches = n_sites // patch_size
    patch_relation = np.full((n_patches, n_patches), 4, dtype=np.int32)
    np.fill_diagonal(patch_relation, 0)

    # Prefer physical Kitaev bonds over self/non-neighbor relations.
    priority = [1, 2, 3, 0, 4]

    for a in range(n_patches):
        sites_a = range(a * patch_size, (a + 1) * patch_size)
        for b in range(n_patches):
            sites_b = range(b * patch_size, (b + 1) * patch_size)
            rels = []
            for i in sites_a:
                for j in sites_b:
                    rels.append(site_relation[i, j])
            rels = np.asarray(rels, dtype=np.int32)

            for rel in priority:
                if np.any(rels == rel):
                    patch_relation[a, b] = rel
                    break

    return tuple(tuple(int(x) for x in row) for row in patch_relation)


class BondAwareSelfAttention(nn.Module):
    """
    Multi-head self-attention with a learned Kitaev bond-type bias.

    The attention logits are modified as:

        logits[b, h, i, j] += bond_bias[h, relation_matrix[i, j]]

    where relation ids are typically:
      0 = self
      1 = x bond
      2 = y bond
      3 = z bond
      4 = non-neighbor

    The bias is initialized at zero, so the model starts equivalent to
    ordinary self-attention and learns whether the bond structure is useful.
    """
    embed_dim: int
    num_heads: int
    relation_matrix: Tuple[Tuple[int, ...], ...]
    # ``None`` means infer the required cardinality from ``relation_matrix``.
    # This is the safe default because the extended site-relation builder uses
    # relation IDs beyond the original five bond/non-neighbor classes.
    num_relation_types: int | None = None
    data_type: jnp.dtype = jnp.float64

    @nn.compact
    def __call__(self, x):
        B, T, D = x.shape

        relation_ids = tuple(
            int(relation_id)
            for row in self.relation_matrix
            for relation_id in row
        )
        if not relation_ids:
            raise ValueError("relation_matrix must contain at least one relation id")
        if min(relation_ids) < 0:
            raise ValueError("relation_matrix relation ids must be non-negative")
        required_relation_types = max(relation_ids) + 1
        if self.num_relation_types is None:
            num_relation_types = required_relation_types
        else:
            num_relation_types = int(self.num_relation_types)
            if num_relation_types < required_relation_types:
                raise ValueError(
                    "relation_matrix requires "
                    f"{required_relation_types} relation types (ids 0 through "
                    f"{required_relation_types - 1}), but num_relation_types="
                    f"{num_relation_types}."
                )

        if D != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {D}")

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )

        relation = jnp.asarray(self.relation_matrix, dtype=jnp.int32)
        if relation.shape != (T, T):
            raise ValueError(
                f"relation_matrix shape {relation.shape} does not match token shape {(T, T)}. "
                "For PATCH_SIZE=1 this should be (NUM_SITES, NUM_SITES). For larger "
                "patches, pass a patch-level relation matrix."
            )

        head_dim = self.embed_dim // self.num_heads

        qkv = nn.Dense(
            3 * self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="qkv",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        logits = jnp.einsum("bhid,bhjd->bhij", q, k)
        logits = logits / jnp.sqrt(jnp.asarray(head_dim, dtype=self.data_type))

        bond_bias_table = self.param(
            "bond_attention_bias",
            nn.initializers.zeros,
            (self.num_heads, num_relation_types),
            self.data_type,
        )

        # bond_bias has shape (num_heads, T, T).
        bond_bias = bond_bias_table[:, relation]
        logits = logits + bond_bias[None, :, :, :]

        attn = nn.softmax(logits, axis=-1)

        y = jnp.einsum("bhij,bhjd->bhid", attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)

        y = nn.Dense(
            self.embed_dim,
            dtype=self.data_type,
            param_dtype=self.data_type,
            name="out",
        )(y)

        return y


class BondAwareTransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    relation_matrix: Tuple[Tuple[int, ...], ...]
    data_type: jnp.dtype
    num_relation_types: int | None = None
    residual_scale: float = 0.8

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)

        y = BondAwareSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            relation_matrix=self.relation_matrix,
            num_relation_types=self.num_relation_types,
            data_type=self.data_type,
            name="BondAwareSelfAttention",
        )(y)

        x = x + self.residual_scale * y

        y = nn.LayerNorm(
            dtype=self.data_type,
            param_dtype=self.data_type,
        )(x)

        y = MLP(
            hidden_dim=self.mlp_hidden_dim,
            out_dim=self.embed_dim,
            data_type=self.data_type,
        )(y)

        x = x + self.residual_scale * y
        return x


class KitaevBondAwareHoneycombViT(nn.Module):
    """
    Honeycomb ViT with Kitaev bond-aware attention.

    This is intended as a minimally changed version of HoneycombPatchViT:
      - same patch embedding style
      - same learned positional embeddings
      - same OutputHead
      - transformer blocks replaced by BondAwareTransformerBlock

    Recommended first test:
      PATCH_SIZE = 1

    For PATCH_SIZE > 1, pass a patch-level relation matrix created with
    site_relation_to_patch_relation(...).
    """
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_hidden_dim: int
    patch_size: int
    learn_phase: bool
    relation_matrix: Tuple[Tuple[int, ...], ...]
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

        relation_matrix = self.relation_matrix
        if self.patch_size > 1:
            # The caller may pass a patch-level relation matrix directly. If a
            # site-level matrix is passed instead, convert it here.
            rel_arr = jnp.asarray(relation_matrix, dtype=jnp.int32)
            if rel_arr.shape == (N, N):
                relation_matrix = site_relation_to_patch_relation(
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
