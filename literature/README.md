# Literature review & gap analysis

Condensed from Sections I-II of `paper/IEEE_Final.pdf`. Full source PDFs are **not** committed (copyright) - use the links/IDs in `bibliography.bib`.

## Taxonomy of clean-label attacks used in the review
| Category | Methods | Key point |
|---|---|---|
| Feature collision | Poison Frogs [1], Convex Polytope [2], Bullseye Polytope [3], Hidden Trigger [5] | PGD pixel perturbation so poison features collide with a target in a surrogate's embedding space. ~100% ASR in transfer learning, drops to ~60-70% under end-to-end retraining; white-box needs. |
| Adversarial / generative perturbation | Label-Consistent [4], CleanSheet | Adversarial/GAN-interpolated perturbations keep labels clean in end-to-end training. |
| Surrogate-based inward-trigger | Narcissus | Surrogate trained on target-class + public OOD data; ~99% ASR at ~0.05% poison rate. |

## Diffusion models in the arms race
* **Defence:** DiffPure [6], ECLIPSE [7], Hong et al. [8] - forward-diffuse then denoise to strip high-frequency perturbations; drives classical clean-label ASR to near zero.
* **Diffusion as poisoning *target*:** Pan et al. [11] (BadNets-style trigger in a diffusion model's own training data).
* **Diffusion as poison *generator*:** Souri et al. [9] - classifier-guided diffusion makes base images, then a *separate* gradient-matching attack (Witches' Brew) does the poisoning (two-stage). CBV [10] - modifies the score function to embed backdoor semantics, but for VLMs with multimodal guidance.

## Identified gaps
1. Trigger/perturbation design is arbitrary (random-noise start) rather than semantically grounded.
2. High attacker-knowledge requirements (white-box / strong surrogate).
3. Generative models are under-explored for *synthesising* poison images.

## This work's (narrow) claim
Single-stage guided reverse diffusion with a feature-collision objective embedded in the denoising trajectory, frozen pretrained DDPM, for clean-label poisoning of **CNN classifiers**, to be benchmarked against purification defences vs a PGD baseline (benchmark **not yet run**). Not claimed: first use of diffusion for poisoning.

## TODO
Add per-paper annotations (1-2 lines each), fill the TODO entries in `bibliography.bib`.
