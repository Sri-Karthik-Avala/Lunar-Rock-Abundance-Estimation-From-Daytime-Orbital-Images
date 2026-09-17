# Lunar Rock-Abundance — Experiment Report

Metric: `0.25*pooled + 0.75*worst_two_regions`, where prediction & truth ranks are
orthogonalized against a cubic of a brightness/albedo-rank baseline (brightness-only ≈ 0).
Target: beat ~0.42, reach 0.58+.

## Submissions & leaderboard (ground truth)

| Ver | Model / representation | random-fold OOF | GroupKFold pseudo-region | **LB** |
|-----|------------------------|-----------------|--------------------------|--------|
| v1  | from-scratch ResNet-GN, raw global-norm | 0.625 | 0.236 | 0.4279 |
| v2  | + high-pass channel, strong aug | 0.647 | 0.309 | 0.4279 |
| v3  | FFT log-mag + high-pass, continuous-rotation aug | 0.559 | **0.353** | **0.3352** |
| v5  | **ConvNeXt-Tiny (ImageNet)**, 3 texture channels | (pending) | — | (pending) |

## Key findings (what worked / failed / why)

1. **Validation: random-fold OOF predicts the LB; cluster-GroupKFold does NOT.**
   v3 *improved* the GroupKFold-over-image-clusters score (0.236→0.353) yet the LB *fell*
   (0.428→0.335). v1/v2 random-fold OOF tracked the LB positively. Reason: clustering on
   image descriptors and holding clusters out removes an entire texture-statistic regime —
   far harsher and unlike the real held-out provinces. **Decision: select on random-fold OOF.**
   Empirical relation ≈ `LB ≈ 1.4*OOF − 0.45` → reaching 0.58 needs OOF ≈ 0.73.

2. **Spectral-only under-performs spatial.** The FFT/rotation-invariant branch (v3) was the
   single worst submission despite the strongest brightness-orthogonality in offline analysis.
   The real test rewards general predictive accuracy from spatial morphology, not pure frequency
   statistics. (Frequency info is still useful as an auxiliary channel, not as the core.)

3. **From-scratch CNNs cap near OOF 0.63–0.65 → LB ~0.43** on this faint, information-limited
   signal. The realistic lever for a step change is a **stronger pretrained backbone** (allowed
   by the rules), which is what v5 introduces.

4. **Engineering bugs found & fixed along the way:** giant single-batch eval caused Windows
   GPU-memory spill (100× slowdown) → batched inference; dead code in `rankdata`; brightness
   aug corrupting zero-mean high-pass channels; one-shot 8000-patch GPU allocation → batched.

## v5 design (current solution.py)

- **Backbone:** torchvision **ConvNeXt-Tiny** (ImageNet), avg+max-pooled regression head.
  Layered fallback: ConvNeXt → ResNet18 → from-scratch (sandbox-safe; always runs).
- **Input:** 3 brightness-invariant multi-scale texture channels `[raw, high-pass 9×9, high-pass 3×3]`,
  bilinearly resized to 128. All stats regenerated at runtime (no hardcoding, no leakage).
- **Augmentation (brightness-shortcut destruction, texture-preserving):** D4; per-image contrast;
  a large additive offset on the raw channel only (high-pass channels are immune → forces the
  model onto texture); mild noise.
- **Loss:** SmoothL1 + 0.6·(1−Pearson) on standardized log-abundance (rank-aligned).
- **Ensemble/inference:** 5-fold, D4×8 TTA, rank→train-quantile mapping (monotone, valid [0,1]).

## Next directions (gated on v5 LB)

- If v5 helps but short of 0.58: **diversity ensemble** — ConvNeXt + a texture-descriptor
  GBM (LBP/Haralick/GLCM/FFT-stats) + a second backbone; rank-average. Diversity, not depth.
- ConvNeXt-Small / partial vs full fine-tune sweep, multi-scale (100+64) fusion.
- Heavier photometric invariance if OOF↔LB gap suggests residual brightness reliance.

What we deliberately de-prioritized (evidence-based): cluster-GroupKFold model selection,
FFT-centric models, and SSL pretraining — none are supported by the LB results so far.
