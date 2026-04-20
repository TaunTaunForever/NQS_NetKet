# 8-Site NIR Experiments

This directory mirrors the 18-site NIR sandbox but is tuned for fast local
experiments on the 8-site Kitaev model.

Current starter experiment:

- `kitaev_honeycomb_ViT_8_nir.py`
  A lightweight plain-ViT NIR trainer using:
  - `NUM_LAYERS = 2`
  - `EMBED_DIM = 8`
  - `NUM_HEADS = 4`
  - `PATCH_SIZE = 2`

The script currently implements an amplitude-only NIR training stage:

1. Sample proposals from a separate autoregressive proposal network.
2. Compute target/proposal log-probabilities.
3. Form importance weights and monitor ESS / efficiency.
4. Resample configurations.
5. Train the proposal network with a forward-KL objective.
6. Inject the resampled batch into a NetKet `MCState`.
7. Use `expect_and_grad` to update the ViT.
