# Gamma 8-site weighted NIS

Every maintained model-type launcher is a direct, editable entry point. Edit
its top-level `CONFIG` and run it directly:

- `vit_nis.py` or `vit_nis_site_relation.py`: site-relation target;
- `vit_nis_base.py`: base ViT target;
- `vit_nis_inputproj.py`: input-projection target;
- `vit_nis_symmproj.py`: symmetry-projected target.

The site-relation target uses the same site-type/relation encoding as the
current Gamma SRT model.

## Target-model variants

| Script / variant | Architectural change | Symmetry behaviour | Main trade-off |
| --- | --- | --- | --- |
| `vit_nis_base.py` / `plain` | Patch embedding, learned positional embeddings, ordinary self-attention, and the shared complex output head. | No built-in point-group constraint. | Fastest and most flexible baseline, but it must learn lattice structure and symmetry from data. |
| `vit_nis.py` or `vit_nis_site_relation.py` / `site_relation` | Adds bipartite site-type embeddings and a six-class extended Kitaev relation bias to attention. | Encodes local honeycomb/bond structure; it does not explicitly average over the full point group. | The recommended physics-informed default; slightly more parameters/attention bookkeeping. |
| `vit_nis_inputproj.py` / `inputproj` | Maps every spin configuration to the lexicographically smallest configuration in its point-group orbit before the shared base ViT. | Exactly invariant under the supplied point group at the input level. | Evaluates one base network per sample, but canonicalization is discrete and may make the input map less smooth at orbit boundaries. |
| `vit_nis_symmproj.py` / `symmproj` | Evaluates the shared base ViT on every point-group image and combines log-amplitudes with a stable complex log-mean-exp. | Explicit symmetry projection of the wavefunction. | Strongest direct symmetry enforcement, but costs one base-model evaluation per symmetry operation. |

All variants use the same autoregressive proposal, weighted NIS estimator,
weighted matrix-free SR update, and NetKet-native multi-GPU sharding. This
makes their energy, ESS, and SR diagnostics directly comparable.

Its default `execution_backend="netket"` uses `WeightedNISState` and
`WeightedNISVMC`: energies are NetKet `Stats`, target parameters are updated by
a NetKet-compatible driver, and proposal fitting remains an explicit weighted
NIS update. Resampled configurations are saved only as compatibility artifacts;
they are never injected into `MCState` or treated as iid target samples.

With `use_multi_gpu=True`, the NetKet backend natively shards one global
proposal pool over all visible GPUs, including the weighted matrix-free SR
operator. Keep `n_proposals` divisible by the GPU count. The retained
`execution_backend="explicit_jax"` option is the legacy pmap path and does not
support weighted SR. Exact diagonalization and full-Hilbert-space diagnostics
remain opt-in 8-site validation tools.
