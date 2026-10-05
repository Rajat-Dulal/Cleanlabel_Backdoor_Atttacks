# Project plan, milestone log, risks (SIT723 -> SIT746)

> TODO: not recoverable from the code or the paper - fill in from your own records.

## Phase plan
| Phase | Unit | Deliverable | Status |
|---|---|---|---|
| 1 | SIT723 | Literature review, gap analysis, feature-interpolation baseline | Done |
| 2 | SIT746 | Guided-diffusion pipeline, t_start sweep, cross-architecture test | Preliminary |
| 3 | - | Purification-robustness comparison (DiffPure vs PGD baseline) | Not run |
| 4 | - | Paper / workshop submission | Draft |

## Milestone log
_(date - milestone - note)_

## Risks / challenges log (seed items observed in the code & paper)
* Free-tier GPU limits -> small surrogate CNN, 8-epoch victims, single-seed results.
* Phase 1 ASR (14.5%) well below literature; needs lambda / eps / schedule tuning.
* Attack fails against ImageNet-pretrained victim (negative result).
* Purification-robustness comparison - the central claim - not yet run.
