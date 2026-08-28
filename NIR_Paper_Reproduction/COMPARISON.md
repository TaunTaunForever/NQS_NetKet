# Method Comparison

This file tracks the gap between the paper's NIR setup and our current NIR experiments.

## Paper Method

- Proposal distribution: separate autoregressive Transformer sampling network.
- Proposal tokenization: spin/site values are embedded, shifted by a learned beginning token, and passed through causal Transformer blocks.
- Target distribution: the NQS probability distribution `p(s) = |psi(s)|^2`.
- Importance weights: `w(s) = p(s) / q(s)`.
- Target samples: multinomial-style importance resampling from proposal samples.
- Proposal loss: forward KL, estimated by maximizing proposal log-probability on samples drawn from the target via resampling.
- Probability support: a small single-site probability floor is added so that `q(s) > 0` wherever `p(s) > 0`.
- Adaptive retraining: collect proposal samples until ESS is sufficient; update the NQS only if sampling efficiency is above threshold.
- Optimizer stack: Adam for NQS and proposal; MinSR used for NQS convergence.

## Paper Numerical Targets

- TFI:
  - Target NQS: multi-state MLP.
  - MLP hidden layers: 4.
  - MLP hidden width ambiguity: the text says 256, the appendix table says 128.
  - Proposal: Transformer with embedding dimension 32, 4 heads, 4 layers.
  - Learning rates: NQS 1e-3, proposal 1e-3.
  - Steps: 10000.
  - Batch size: 512 for both NQS and proposal network.
  - Adaptive thresholds: ESS threshold listed as 2.0 and efficiency threshold as 0.1.

- J1-J2 square:
  - System: 10x10 square lattice, J2/J1 = 0.5.
  - Target NQS: residual CNN with 16 channels, 4 residual blocks, 3x3 circular convolutions, layer norm, GELU, global average pooling.
  - Symmetries: translation from circular convolution/pooling, spin inversion by canonicalizing first spin, rotations/reflections by averaging transformed outputs.
  - Magnetization handling: target penalizes nonzero magnetization by subtracting 30 from real log psi; proposal does not enforce magnetization.
  - Reported proposal table:
    - 16 dim, 2 layers: not completed, efficiency 0.01.
    - 16 dim, 3 layers: not completed, efficiency 0.03.
    - 16 dim, 4 layers: E/N = -0.4969, efficiency 0.10.
    - 32 dim, 2 layers: E/N = -0.4966, efficiency 0.15.
    - 32 dim, 3 layers: E/N = -0.4968, efficiency 0.21.
    - 32 dim, 4 layers: E/N = -0.4969, efficiency 0.28.

## Our Current NIR Method

- Proposal network: autoregressive Transformer in Flax, now with float32 proposal parameters and a JAX `lax.scan`/decode-cache sampling path.
- Proposal training: forward-KL on resampled configurations by default, matching the paper's preferred mode.
- Importance weights: computed from log target/proposal probabilities, matching the paper.
- Adaptive trigger: defaults to efficiency thresholding, with optional log-ratio standard deviation support.
- Resampling: supports multinomial, systematic, and stratified; current paper-faithful baseline should use multinomial.
- Target update: usually NetKet MCState injection plus SR/MinSR direction.
- Added variants not in the paper: weighted target updates, unique-weighted updates, exact/full-sum refinement, sampler refinement, staged sample/LR schedules, fixed-magnetization proposal support.

## Major Differences To Remove For Reproduction

- Use paper systems first: TFI and square J1-J2, not honeycomb Gamma/J1-J2.
- Use paper target models first: multi-state MLP and residual CNN, not site-aware ViT.
- Use paper optimizer defaults first: Adam plus MinSR, not our late-stage SGD/SRt tuning.
- Use paper proposal dimensions first: 32 dim, 4 heads, 4 layers for TFI; table values for square J1-J2.
- Use paper batch size first: 512 target samples and 512 proposal training samples.
- Do not use exact/full-sum refinement as part of NIR reproduction.
- Do not constrain the square J1-J2 proposal to fixed magnetization in the baseline, because the paper explicitly does not.

## Implemented In This Reproduction Workspace

- `scripts/run_tfi_single_state_nir.py`: runnable single-state TFI NIR harness.
- `scripts/run_j1j2_square_resnet_nir.py`: runnable square-lattice J1-J2 ResNet NIR harness.
- `src/nir_paper_reproduction/proposal.py`: paper-style autoregressive Transformer proposal.
- `src/nir_paper_reproduction/nir_core.py`: paper-style NIR loop with ESS accumulation, efficiency gate, resampling, proposal forward-KL, Adam target/proposal updates, and optional MinSR.

The full paper TFI benchmark still needs the multi-state/determinant NQS layer. Until that is implemented, the TFI harness should be treated as a sampler sanity check rather than a reproduction of the exact TFI figures.

## Open Paper Ambiguities

- The TFI MLP width differs between the main text and appendix table: 256 versus 128.
- The ESS threshold is listed as 2.0, while the text describes an ESS threshold without fully specifying whether it is absolute or a multiple of target batch size.
- The arXiv source did not include official code, so implementation details such as the precise MinSR schedule, random seeds, and proposal probability floor must be inferred or swept.
