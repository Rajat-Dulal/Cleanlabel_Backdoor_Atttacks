# Research design (DRAFT extracted from paper/IEEE_Final.pdf - review before marking Final)

**Title:** Clean-Label Poisoning Attacks on Convolutional Neural Networks: From Feature Interpolation to Guided Diffusion Sampling

## Aim
Determine whether image generative models can produce clean-label poisons for CNN image classifiers that are
(a) more effective and (b) harder to remove with diffusion-based purification (DiffPure-style) than
PGD-bounded pixel-space poisons.

## Research questions
> TODO: replace with the exact three RQs from your SIT746 submission. Suggested wording derived from the paper:
1. Can latent-space (feature) interpolation toward real target-class embeddings give a more effective/semantically grounded clean-label poison than random-noise initialisation? (Phase 1)
2. Can single-stage guided reverse diffusion with a feature-collision objective produce effective clean-label poisons at very low poison rates, using only a frozen pretrained DDPM? (Phase 2)
3. Are diffusion-native poisons more resistant to diffusion-based purification than PGD-based poisons? (Phase 2 - **not yet evaluated**)

## Threat model
Attacker injects a bounded number of **correctly labelled** poison samples into the victim's training set; cannot change
labels, training procedure, or see the victim's final weights. A surrogate CNN approximates the victim's feature space
(black-box transfer assumption). In Phase 2 the diffusion model is only a poison *generator* - neither the attack
target nor the defence. *(Threat-model diagram: add `docs/threat_model.png`.)*

## Methods
* **Phase 1 / SIT723** - `src/sit723_feature_interpolation`: z~ = (1-lam) z_b + lam p_t, inverted to pixels by PGD under an L-inf budget. lam=0.35, eps=8/255, 5% of base class.
* **Phase 2 / SIT746** - `src/sit746_guided_diffusion`: SDEdit partial forward diffusion to t_start, then guided reverse DDPM
  (`google/ddpm-cifar10-32`), loss = feature-collision (surrogate embedding) + LPIPS(x0_hat, x_base), step scaled by g * t/T.

## Metrics
CDA (clean test accuracy), ASR (definitions differ per phase - see `reproducibility_notes.md` section D), SSIM / LPIPS (stealth), confusion matrices.

## Compute assumptions
Free-tier Google Colab / Kaggle, NVIDIA T4. No diffusion model trained from scratch. Surrogate ~2-3 min (15 epochs); runtime of poison generation / victim training was not recorded - fill in after next run.

## Evaluation still to do (from paper Sec. V-C / VII)
Poison-budget study (1-20%), purification-robustness vs PGD baseline across t_purify, more class pairs, Neural Cleanse / STRIP / ANP, LPIPS stealth, ablations (lambda, t_start, g, LPIPS weight, surrogate data size), pretrained-victim failure analysis.
