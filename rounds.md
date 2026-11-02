# Training and evaluation rounds

## Round 0 — plain pretrained YAMNet, no training

Run date: 2026-10-31T00:41:50.784733-07:00. Scope: four frozen Valorant tests (01, 02, 05, 10) and the unseen CS test (16). The final synthetic lab clip was not opened.

This is the untouched-model before baseline. No add-on was trained, no voice removal or side-channel recognition was used, and no settings were changed after seeing test results. Future improvement rounds must use development validation; these test clips are closed again until the final after comparison.

| Round | What changed | Set | Caught % | False/min | Arrow % | Talking: caught % | Music: caught % |
|---|---|---|---|---|---|---|---|
| 0 | Plain YAMNet; no training | Valorant (4 clips) | 20.0% (4/20) | 1.31 | 100.0% (3/3) | N/A | N/A |
| 0 | Same fixed model/settings | Unseen CS (1 clip) | N/A (0/0) | 4.00 | N/A (0 arrows scored) | N/A | N/A |

### Per-label caught rate

Primary targets exclude sounds manually labeled center, matching the chosen default to hide centered captions. Unknown-direction targets remain eligible for sound recognition. Percentages use matched event counts, not frame accuracy.

| Label | Valorant caught / eligible | Caught % | CS caught / eligible | Caught % |
|---|---|---|---|---|
| footsteps | 0/1 | 0.0% | 0/0 | N/A |
| gunshot | 4/19 | 21.1% | 0/0 | N/A |
| reload | 0/0 | N/A | 0/0 | N/A |
| explosion | 0/0 | N/A | 0/0 | N/A |

Footsteps have only one eligible Valorant example, so the recall estimate is especially uncertain. Reload and explosion have no eligible labeled examples; their scores are N/A, not perfect or zero accuracy. YAMNet also has no explicit reload class, so round 0 cannot emit reload captions.

### Caption errors and arrows

Valorant: 14 captions over 2.286 minutes. 3 had no annotated game-sound onset of any label within one second (1.31/min). 10 captions were unmatched under one-to-one same-label scoring (4.37/min); this broader count also includes wrong labels, duplicate captions, and captions matching only excluded centered targets.

CS: 2 captions over 0.500 minutes; 2 false by the same rule (4.00/min), and 2 unmatched (4.00/min).

Valorant arrow accuracy is 3/3 (100.0%), with arrows on 3/4 matched left/right targets (75.0% coverage). The 100% figure is only three arrows, not evidence of reliable direction performance. Unknown directions do not receive arrow credit or count as correct arrows.

### Centered-event accounting

| Set | Centered targets excluded from primary recall | Caught / all labeled sounds, including center |
|---|---|---|
| Valorant | 19 | 6/39 (15.4%) |
| Unseen CS | 1 | 0/1 (0.0%) |

The CS CSV currently contains only one game-sound event, labeled center. There are therefore zero eligible CS targets under the default center-hidden policy, and primary CS recall and arrow accuracy cannot be measured. The all-sounds view is provided to make this exclusion visible.

### Quiet / talking / music

All five files currently have zero talking/music annotation spans. The user has explicitly said those spans are not complete. Quiet, talking, and music metrics are all N/A: missing spans must not be treated as quiet audio. Model speech guesses were not used to fill the answer keys. Cached predictions can later be rescored against completed context labels without rerunning or tuning the model. Preserve this original round-0 report and annotation hashes when adding that supplemental report.

### Confusion table

Valorant eligible target events. Correct same-label matches are assigned first. Remaining nearby wrong-label predictions are paired one-to-one with missed targets only for this descriptive table; temporal coincidence is not proof of what caused an error. Predictions left over after this table are still included in unmatched captions/min.

| True sound | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 0 | 0 | 0 | 0 | 1 |
| gunshot | 2 | 4 | 0 | 0 | 13 |
| reload | 0 | 0 | 0 | 0 | 0 |
| explosion | 0 | 0 | 0 | 0 | 0 |

CS has no eligible primary targets, so its primary confusion table has no entries. The original all-direction counts remain in the machine-readable summary.

### Frozen method and reproducibility

- Recognition: original mixed audio, averaged to 16 kHz mono, 15,600 samples per window (0.975 s), stride 7,680 samples (0.48 s). No model fitting or source separation.
- Classes: footsteps 48; gunshot 421/422/423/424; explosion 420; reload unsupported. Thresholds 0.06 / 0.08 / 0.12 reuse the step-1 labeling-proposal thresholds unchanged. These were not optimized for live captions.
- Direction: original stereo energy balance above 0.18 for left/right; balanced near-identical stereo (side energy ratio below 0.03) classified center and suppressed. Otherwise no arrow.
- Caption policy: max two simultaneous captions, two-second lifetime, one attempt per consecutive positive label/direction run. Capacity-blocked attempts are discarded. This deterministic policy is a scoring adapter; it is not an implemented live overlay.
- Timing: caption timestamp is the window end, not its beginning. One tail window is padded at the media endpoint. Wall-clock playback latency is not measured by this offline run.
- Matching: same label, onset difference at most one second, maximum-cardinality one-to-one matching. Every caption and every target can match at most once.
- False/min follows the original brief literally: no nearby game-sound onset of any class. Unmatched/min is reported too, so wrong labels and repeats are not hidden.
- Aggregate metrics pool event counts and duration; they do not average clip percentages.
- No test-error-driven tuning was performed. No training clips or voice-reduced copies were scored in this run.

Frozen implementation commit before evaluation: `307853c350ad68b0721e10acb90c0c1e2ce8ba22`.
Model SHA-256: `10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de`.
Configuration SHA-256: `06e81d72c41cf5b9cbca9cdc21e2e5fc7be56fda62c0527e169c6b1cb35fe2bc`.

Code: `run_baseline.py` and `scoring.py`. Protocol and aggregate/per-clip metrics: `evaluations/round-0/protocol.json` and `evaluations/round-0/summary.json`. The summary fingerprints each original media file and answer CSV. Local immutable prediction lists and class-score caches: `.local/evaluations/round-0/`. The runner refuses to overwrite an existing round-0 protocol or result.

Validation: all 24 tests passed before the one baseline run, including matching boundaries, maximum matching, duplicates/wrong classes, centered targets, arrow coverage, weighted aggregation, two-caption capacity, deduplication, and center suppression. Test CSVs and original audio were not changed by evaluation.

No accuracy-improvement claim is made: this is the starting point. Continue labeling the eleven development clips; future changes are selected on development validation before the final sealed comparison.

## Development reference — Plain pretrained YAMNet on development clips

**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.

| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |
|---|---|---|---|---|---|---|---|---|
| Reference | 12.5% (5/40) | 1.13 | 6.20 | 66.7% (2/3) | 60.0% | 20.0% | 8.0% | N/A |

| Label | Caught / target | Recall |
|---|---|---|
| footsteps | 1/5 | 20.0% |
| gunshot | 4/29 | 13.8% |
| reload | 0/6 | 0.0% |
| explosion | 0/0 | N/A |

Confusion counts (rows true, columns predicted; missed is no matched caption):

| True | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 1 | 0 | 0 | 0 | 4 |
| gunshot | 1 | 4 | 0 | 0 | 24 |
| reload | 0 | 0 | 0 | 0 | 6 |
| explosion | 0 | 0 | 0 | 0 | 0 |

All-direction recall, including intentionally hidden centered events: 18/111 (16.2%). Primary scoring excluded 71 centered targets.

Machine-readable metrics: `training/round-0/metrics.json`. No test-based tuning.

## Training round 1 — Original-only control to measure whether voice-reduced training augmentation helps

**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.

| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |
|---|---|---|---|---|---|---|---|---|
| Pretrained development reference | 12.5% (5/40) | 1.13 | 6.20 | 66.7% | 60.0% | — | — | — |
| 1 | 47.5% (19/40) | 5.45 | 15.02 | 81.8% (9/11) | 57.9% | 53.3% | 44.0% | N/A |

| Label | Caught / target | Recall |
|---|---|---|
| footsteps | 3/5 | 60.0% |
| gunshot | 14/29 | 48.3% |
| reload | 2/6 | 33.3% |
| explosion | 0/0 | N/A |

Confusion counts (rows true, columns predicted; missed is no matched caption):

| True | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 3 | 1 | 0 | 0 | 1 |
| gunshot | 3 | 14 | 1 | 0 | 11 |
| reload | 0 | 1 | 2 | 0 | 3 |
| explosion | 0 | 0 | 0 | 0 | 0 |

All-direction recall, including intentionally hidden centered events: 45/111 (40.5%). Primary scoring excluded 71 centered targets.

Each outer fold selected its threshold using only inner validation on the other ten clips. Thresholds: 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9.

Machine-readable metrics: `training/round-1/metrics.json`. No test-based tuning.

## Training round 2 — Same head and encoder, training on original plus aligned voice-reduced copies

**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.

| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |
|---|---|---|---|---|---|---|---|---|
| Pretrained development reference | 12.5% (5/40) | 1.13 | 6.20 | 66.7% | 60.0% | — | — | — |
| 2 | 47.5% (19/40) | 5.45 | 15.96 | 72.7% (8/11) | 57.9% | 66.7% | 36.0% | N/A |

| Label | Caught / target | Recall |
|---|---|---|
| footsteps | 4/5 | 80.0% |
| gunshot | 13/29 | 44.8% |
| reload | 2/6 | 33.3% |
| explosion | 0/0 | N/A |

Confusion counts (rows true, columns predicted; missed is no matched caption):

| True | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 4 | 1 | 0 | 0 | 0 |
| gunshot | 7 | 13 | 1 | 0 | 8 |
| reload | 0 | 2 | 2 | 0 | 2 |
| explosion | 0 | 0 | 0 | 0 | 0 |

All-direction recall, including intentionally hidden centered events: 42/111 (37.8%). Primary scoring excluded 71 centered targets.

Each outer fold selected its threshold using only inner validation on the other ten clips. Thresholds: 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9.

Machine-readable metrics: `training/round-2/metrics.json`. No test-based tuning.

## Training round 3 — Stronger mn10_as encoder plus mono/side features, retaining original plus cleaned training pairs

**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.

| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |
|---|---|---|---|---|---|---|---|---|
| Pretrained development reference | 12.5% (5/40) | 1.13 | 6.20 | 66.7% | 60.0% | — | — | — |
| 3 | 32.5% (13/40) | 6.01 | 17.28 | 100.0% (6/6) | 46.2% | 46.7% | 24.0% | N/A |

| Label | Caught / target | Recall |
|---|---|---|
| footsteps | 0/5 | 0.0% |
| gunshot | 13/29 | 44.8% |
| reload | 0/6 | 0.0% |
| explosion | 0/0 | N/A |

Confusion counts (rows true, columns predicted; missed is no matched caption):

| True | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 0 | 3 | 1 | 0 | 1 |
| gunshot | 6 | 13 | 1 | 0 | 9 |
| reload | 2 | 0 | 0 | 0 | 4 |
| explosion | 0 | 0 | 0 | 0 | 0 |

All-direction recall, including intentionally hidden centered events: 40/111 (36.0%). Primary scoring excluded 71 centered targets.

Each outer fold selected its threshold using only inner validation on the other ten clips. Thresholds: 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9.

Machine-readable metrics: `training/round-3/metrics.json`. No test-based tuning.

## Step 3 checkpoint — stopped below 50%

All eleven finished Valorant development clips have aligned, finite 44.1 kHz stereo voice-reduced copies. Each output sample count matches the original media timeline exactly; labels, source identity, and copy hashes were checked before training. They were already complete when this run started, so no missing copies needed regeneration. Audit: `training/augmentation-audit.json`.

| Development experiment | Input / training data | Caught | False/min | Unmatched/min | Arrows correct |
|---|---|---|---|---|---|
| Pretrained development reference | yamnet, mono, original only | 5/40 (12.5%) | 1.13 | 6.20 | 2/3 (66.7%) |
| Round 1: original-only control | yamnet, mono, original only | 19/40 (47.5%) | 5.45 | 15.02 | 9/11 (81.8%) |
| Round 2: paired YAMNet | yamnet, mono, original + voice-reduced | 19/40 (47.5%) | 5.45 | 15.96 | 8/11 (72.7%) |
| Round 3: paired EfficientAT | efficientat, mono+side, original + voice-reduced | 13/40 (32.5%) | 6.01 | 17.28 | 6/6 (100.0%) |

**Stop decision:** the best real-development score is 19/40 = **47.5%**, below the user's 50% checkpoint. The required paired-data model also scores 47.5%. Training stopped after the three predeclared rounds. No final after test evaluation, new-game adaptation, live helper, overlay, or chat-bot work was started. Saved heads are experimental checkpoints, not approved release models. `training/decision.json` records this stop.

The original-only round is a control for the augmentation experiment. Round 2 trains on all originals and voice-reduced siblings within each training fold. Adding cleaned copies changed footsteps from 3/5 to 4/5 and gunshots from 14/29 to 13/29, leaving total recall unchanged and worsening arrow accuracy and unmatched captions. We cannot claim that voice removal helped overall. Round 3 changes both the encoder and the mono/side representation; its worse score cannot be attributed to either change alone.

False captions/min follows the original brief (no nearby onset of any sound class); unmatched/min additionally includes wrong labels, repeats, and excluded centered targets. Both are high. An inner threshold that met the false-caption objective did not generalize reliably to the outer clip. We did not tune thresholds on the outer clip to conceal this.

Data limits: 111 game-sound events are labeled in the training clips, of which 71 are centered and intentionally excluded from primary recall. The 40 eligible targets comprise only 5 footsteps, 29 gunshots, and 6 reloads. There are no explosion examples. Footstep percentages are consequently very unstable; explosion recall and music scores are N/A. Original-only control arrow accuracy is 9/11, paired YAMNet 8/11, and EfficientAT 6/6, with limited arrow coverage. Six correct arrows are not evidence of dependable direction performance.

These development results use nested leave-one-clip-out: the outer clip and its cleaned sibling are never in training or threshold calibration. Thresholds come from inner leave-one-clip-out on the other ten clips; final checkpoint calibration uses development out-of-fold predictions. Each round fits feature scaling on its training portion only. Matching, center suppression, capacity limit, caption timing, and false-caption definitions use the existing round-0 scoring functions unchanged. Targets are multi-label window-overlap sound labels; talking/music are context only. Context recall assigns events by their onset and preserves the global one-to-one matches; false/min exposure uses the union of annotated intervals. An event spanning a context boundary is not split into extra targets.

The four Valorant test clips, CS test, final lab clip, and their current answer keys/prediction caches were **not opened in step 3**. The historical round-0 test score remains a separate earlier run. The user revised test annotations after that baseline: any future final comparison must rescore the immutable baseline prediction cache against the same final answer key as the new model, preserving the original report. That rescore has not been performed here.

EfficientAT used the official `mn10_as` weights and source commit `a425fdce92572e602a1d5634799bd9f1f2efa806`, with MIT license copied to `third_party/EfficientAT-LICENSE`. It was used as a frozen feature extractor on causal 0.975-second windows, shorter than its usual full-clip AudioSet inference context. This is a tested configuration for this experiment, not a claim about the encoder's best possible accuracy. All model heads use fixed C=0.1 balanced L2 logistic regressions. Adding the second variant doubles the training rows; the C setting was kept unchanged. CPU feature-extraction timings in metric JSONs measure cached batch throughput, not end-to-end live latency. The live pipeline has not been benchmarked.

Reproducibility: `training-plan.json`, `training_core.py`, `run_training.py`, `training/environment.json`, and each `training/round-N/{protocol,metrics, deployment}.json` (deployment only for trained rounds). Local features, per-fold outer captions, and fitted classifier heads are in `.local/training-features/` and `.local/training-runs/`; audio and model binaries remain out of Git. The 28 tests passed before running the rounds, covering scorer behavior, clip exclusion, shared label timing, unsupported classes, and manual context exposure. After training, all eleven annotation/copy hashes and each outer training-ID list were verified unchanged and disjoint from its validation clip.

Before further model work, reconsider the data/model approach rather than continuing to optimize this score. More independently recorded, clearly off-center examples and game-only source audio would be useful next evidence, but no collection, relabeling, or scope change has been started without discussion.

### Context detail for all development rounds

These slices use human annotations, original mixed validation audio, and the same global one-to-one event matches. Quiet is the complement of talking/music in these fully reviewed development clips. Music has no annotated intervals and is N/A, rather than assumed successful.

| Round | Context | Caught / targets | Recall | False/min | Unmatched/min | Arrows correct | Coverage | Footsteps recall | Gunshots recall | Reload recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | quiet | 3/15 | 20.0% | 2.02 | 12.12 | 2/2 (100.0%) | 66.7% | 33.3% | 18.2% | 0.0% |
| 0 | talking | 2/25 | 8.0% | 0.78 | 3.91 | 0/1 (0.0%) | 50.0% | 0.0% | 11.1% | 0.0% |
| 0 | music | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 1 | quiet | 8/15 | 53.3% | 5.39 | 18.85 | 3/3 (100.0%) | 37.5% | 66.7% | 54.5% | 0.0% |
| 1 | talking | 11/25 | 44.0% | 5.47 | 13.54 | 6/8 (75.0%) | 72.7% | 50.0% | 44.4% | 40.0% |
| 1 | music | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2 | quiet | 10/15 | 66.7% | 6.06 | 18.18 | 5/5 (100.0%) | 50.0% | 100.0% | 63.6% | 0.0% |
| 2 | talking | 9/25 | 36.0% | 5.21 | 15.10 | 3/6 (50.0%) | 66.7% | 50.0% | 33.3% | 40.0% |
| 2 | music | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 3 | quiet | 7/15 | 46.7% | 9.43 | 24.24 | 3/3 (100.0%) | 42.9% | 0.0% | 63.6% | 0.0% |
| 3 | talking | 6/25 | 24.0% | 4.69 | 14.58 | 3/3 (100.0%) | 50.0% | 0.0% | 33.3% | 0.0% |
| 3 | music | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Training round 4 — Fold-local cut/paste augmentation, gains and L/R panning; strict inner false-caption limit

**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.

| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |
|---|---|---|---|---|---|---|---|---|
| Pretrained development reference | 12.5% (5/40) | 1.13 | 6.20 | 66.7% | 60.0% | — | — | — |
| 4 | 7.5% (3/40) | 1.13 | 3.57 | 0.0% (0/1) | 33.3% | 13.3% | 4.0% | N/A |

| Label | Caught / target | Recall |
|---|---|---|
| footsteps | 0/5 | 0.0% |
| gunshot | 3/29 | 10.3% |
| reload | 0/6 | 0.0% |
| explosion | 0/0 | N/A |

Confusion counts (rows true, columns predicted; missed is no matched caption):

| True | Footsteps | Gunshot | Reload | Explosion | Missed |
|---|---|---|---|---|---|
| footsteps | 0 | 0 | 0 | 0 | 5 |
| gunshot | 0 | 3 | 0 | 0 | 26 |
| reload | 0 | 0 | 0 | 0 | 6 |
| explosion | 0 | 0 | 0 | 0 | 0 |

All-direction recall, including intentionally hidden centered events: 9/111 (8.1%). Primary scoring excluded 71 centered targets.

Machine-readable metrics: `training/round-4/metrics.json`. No test-based tuning.

Round 4 strictly enforces the 2/min limit on **inner calibration**; it never falls back to an infeasible threshold. The threshold grid extends above 0.9 up to an explicit abstain-all setting. Held-out outer performance, rather than the calibration score, determines the user's 35% / 2-per-minute step-4 gate.

Exact outer-fold thresholds (selected on inner folds): 0.999500, 0.999500, 0.999500, 0.997500, 0.999500, 0.997500, 0.997500, 0.999900, 0.999000, 0.999900, 0.997500.

Step-4 gate: **NOT PASSED**. Nested real-clip validation: 7.5% recall, 1.13 literal false/min, 3.57 unmatched/min. Sealed tests remain closed.

### Round 4 augmentation, calibration and gate audit

The user authorized one more round and replaced the previous 50% checkpoint with **at least 35% nested-validation recall and at most 2 false captions/minute**. This gate was **not met**. Step 4 was not started. The historical round-3 stop record remains preserved; the current decision is `training/decision-round-4.json`.

Created 666 distinct cut-and-paste mixes across the development corpus: three gains (0.35, 0.65, 1.0) × two pans per labeled game event. These supply 3996 new feature windows, including paired background-only controls. With the original and voice-reduced windows, the full-data experimental checkpoint uses 5316 windows. Each cross-validation model uses only the subset of recipes whose **donor and recipient** are both in that model's training fold. The full-data checkpoint is fitted only after the outer validation predictions are fixed; its fit is not a validation score.

Sources are human-labeled sound intervals from the aligned voice-reduced training copies. Recipients are other training clips, with original or cleaned backgrounds. Five-millisecond edge fades reduce splice clicks; gain and left/right panning vary the foreground. Recipient sound labels are retained, and every overlapping donor game-sound label is shifted/clipped along with the audio. Counterfactual background windows carry only their own recipient labels. No human CSV, validation audio, or test audio was edited. Recognition remains the same mono YAMNet embedding/head as round 2; panning creates stereo mixtures, but the classifier itself is not a learned direction model.

All 67 fitted training subsets passed donor/recipient membership checks. Checks run **before cache access and before waveform access**, not just when saving a mix. Every outer held-out clip is excluded in both original and cleaned forms, including from its inner calibration models. The eleven original training annotation hashes were verified unchanged after the run. Provenance is recorded in `training/round-4/augmentation-audit.json` and local per-recipe JSONs under `.local/training-runs/round-4/augmentation/`.

| Outer fold | Selected threshold | Inner recall | Inner false/min |
|---|---|---|---|
| 1 | 0.999500 | 13.2% | 1.45 |
| 2 | 0.999500 | 13.3% | 1.26 |
| 3 | 0.999500 | 7.9% | 1.93 |
| 4 | 0.997500 | 15.4% | 1.66 |
| 5 | 0.999500 | 8.8% | 1.64 |
| 6 | 0.997500 | 20.6% | 1.83 |
| 7 | 0.997500 | 20.0% | 1.87 |
| 8 | 0.999900 | 5.0% | 1.25 |
| 9 | 0.999000 | 9.7% | 1.44 |
| 10 | 0.999900 | 7.9% | 1.23 |
| 11 | 0.997500 | 15.8% | 1.98 |

**Held-out outer validation:** 3/40 = 7.5% recall, 6 literal false captions over 5.325 minutes (1.13/min). 19 captions were unmatched (3.57/min), including wrong labels, repeats and captions matching only excluded centered targets. The narrow false/min figure must not be presented as the total wrong-caption rate. Arrows: 0/1 correct; the sample is too small for a reliability claim.

The full-data checkpoint's single threshold is 0.999000, selected separately from development out-of-fold probabilities. Its **calibration** recall is 10.0% at 1.69 literal false/min. That reused calibration score is not the independent nested score and does not rescue the failed gate. Checkpoint files remain experimental and `ready_for_step4` is false.

| Context | Caught / targets | Recall | False/min | Unmatched/min | Arrow correct |
|---|---|---|---|---|---|
| quiet | 2/15 | 13.3% | 0.67 | 2.69 | 0/0 |
| talking | 1/25 | 4.0% | 1.30 | 3.91 | 0/1 |
| music | N/A | N/A | N/A | N/A | N/A |

The combined change reduced emitted false captions while sacrificing most recall. Because augmentation and stricter threshold selection changed together, this round does not isolate the effect of augmentation alone. Synthetic examples are reused recordings rather than independent new gameplay, and copied mixed snippets may retain other sounds or residual speech. No additional rounds, outer-threshold retuning, final after tests, live helper, overlay, or other new feature work was performed after the gate failed.

Validation: all 32 tests passed before the run, including both donor/recipient fold exclusion before cache reads, pan/gain behavior without array mutation, overlapping annotation shifts, and strict rejection of infeasible thresholds. Existing event-scoring functions and caption scheduling remained unchanged.

## Policy v2 — round-2 development replay; no new training

The user reviewed false-caption candidates and explicitly authorized new scoring/repeat rules, removal of reload captions, and proceeding to step 4. The former 35% / 2-per-minute gate is superseded by that instruction; no new model was fitted and no sealed tests were opened.

A caption is supported only by the **same label** over the labeled span, through at least 2 seconds after that sound starts. The existing 1-second early allowance is retained so an otherwise-correct early caption is not counted as false. Same-label/side candidates less than 2 seconds apart merge into one chain; suppressed candidates extend the quiet-gap timer. Different sides remain independent. Reload captions are suppressed and reload targets are excluded from active-class recall, with the six formerly eligible reload targets reported separately. Original annotations were not changed.

Caught recall remains maximum-cardinality one-to-one same-label onset matching within 1 second. A later supported caption can therefore be neither false nor a newly caught onset. Unmatched/min is retained as an extra diagnostic and includes those later supported captions; it is not equivalent to a wrong-caption rate. Arrow correctness remains independent of sound-label support.

| Comparison | Caught / targets | False/min | Unmatched/min | Arrow correct |
|---|---|---|---|---|
| Historical round 2, old rules incl. reload (not directly comparable) | 19/40 (47.5%) | 5.45 | 15.96 | 8/11 |
| Pretrained development reference, policy v2 | 5/34 (14.7%) | 3.38 | 6.01 | 2/3 |
| Round 2, reload hidden + span scoring, before repeat merge | 17/34 (50.0%) | 4.51 | 13.15 | 7/9 |
| Round 2, complete policy v2 | 16/34 (47.1%) | 4.32 | 12.02 | 7/9 |

Both the reference and replay use the same v2 postprocessor/scorer. The old 47.5% result included 40 targets; the active v2 denominator is 34, so do not present any change in percentage as a training improvement.

| Active class | Caught / targets | Recall |
|---|---|---|
| footsteps | 4/5 | 80.0% |
| gunshot | 12/29 | 41.4% |
| reload | 0/0 | N/A (disabled or no targets) |
| explosion | 0/0 | N/A (disabled or no targets) |

| Context | Caught / targets | Recall | False/min | Arrow correct |
|---|---|---|---|---|
| quiet | 10/14 | 71.4% | 5.39 | 5/5 |
| talking | 6/20 | 30.0% | 3.91 | 2/4 |
| music | N/A | N/A | N/A | N/A |

Replay removed 17 reload captions and merged 7 repeat candidates from 104 original emissions. Only cached outer held-out round-2 emissions were replayed: ten folds used 0.9 and clip 03 used 0.8, exactly as originally inner-calibrated. The live full-data round-2 checkpoint keeps its saved deployment threshold **0.9**. Do not describe the replay as fixed-0.9 cross-validation or as inference from the full-data model on its own training clips.

The exact legacy scheduler is retained before v2 postprocessing for replay/live/final consistency. Consequently a hidden reload candidate can still consume a raw internal slot; this conservative limitation cannot be removed from cached OOF captions without unavailable per-window predictions. The final scorer must apply the same versioned rules to **both** the cached pretrained baseline captions and the learned-model captions and report active-class denominators. Old-policy final scores must not be compared against new-policy scores.

Frozen policy: `caption-policy-v2.json`. Replay protocol/results: `training/policy-v2-round-2/`. This is a user-authorized scoring/presentation revision, **not training round 5**.

### Policy-v2 interpretation and step-4 verification

The 23 remaining span-unsupported captions include 10 surviving old false flags and 13 captions that the old scorer excused because another label had a nearby onset. The new rule requires the same label; a footsteps prediction is not justified by a gunshot annotation. The user’s listening observations were recorded as feedback, not converted into guessed timestamp edits. Previously unmarked footsteps remain unmarked until their annotations are explicitly corrected.

Step 4 is implemented: round-2 checkpoint at 0.9, 16-kHz mono embedding from original stereo, stereo direction, shared legacy scheduler and policy-v2 repeat/reload filter, and a loopback-only receive-only WebSocket on port 8770. Source adapters support paced local development files, read-only OBS Media Source playhead following, and an explicitly chosen stereo game-only input bus. There is no default microphone selection, live vocal separation, overlay UI, bot, or Twitch stream control.

A 12-second paced development-clip smoke test processed 23 windows; median inference/postprocessing was 5.735 ms and p95 was 8.809 ms, in addition to the 975-ms observation window and input/polling overhead. A real WebSocket client received both the current snapshot and a subsequent right-gunshot event. These are plumbing/performance checks on development audio, not accuracy scores. All 45 tests passed, including streaming/batch scheduler parity, span/label/recall boundaries, chained burst merging, reload exclusion, live WebSocket reconnect snapshots, OBS v5 authentication and simulated pause/seek/disconnect/reconnect, stereo 48-kHz-to-16-kHz window resampling without repeated chunk-edge transients, rejection of mono inputs, sealed-path/symlink/hardlink blocking on dummy fixtures, and the final entrypoint guard before any test access.

Native OBS integration and a routed game-only hardware input still need a local check: OBS was not running and no suitable routed game bus was available during implementation. The adapter tests used simulated OBS/device inputs; the model and outbound WebSocket used real local development audio. The helper does not account for OBS mixer volume/pan/filter changes in file-follow mode. Instructions: `LIVE_HELPER.md`; launcher: `Try helper.command`. Source hashes and check results are in `training/policy-v2-round-2/helper-checks.json`.

The final-only runner is prepared but **was not run**. Both before and after will use the frozen policy-v2 postprocessor/scorer; historical old-policy scores remain preserved. All sealed real/CS and lab-final clips stay closed. Step 4 stops here for the user to try; step 5 still requires mockup approval.

## Training round 5 — add labeled skirmish part 01; unchanged round-2 setup

**Verdict: basically no useful overall improvement.** Gunshots improved, footsteps worsened. Across the same 12 held-out development clips, the new model caught **one more sound** and emitted **one more false caption**. Given the user’s preference against false captions, this is not a clear win. No further tuning or training rounds were attempted.

Training-data change only: include the 34 reviewed human labels from `17 skirmish game audio - part 01.csv` (14 footsteps, 12 gunshots, 1 reload, 7 talking spans). Its 14 active non-centered targets are 9 footsteps and 5 gunshots. Parts 02–06 are **explicitly excluded**: their audio, features and answer rows were not loaded, and their header-only CSVs were never treated as negative examples. Only part 01 of the new recording participates. No sealed media, answer keys, caches or metrics were opened.

Frozen before fitting: the identical YAMNet mono embedding, 0.975-second window/0.48-second hop, original multi-label window targets, four independent balanced logistic heads (`C=0.1`, fold-fitted standardizer, liblinear, same seed), original stereo direction, raw two-slot scheduler and policy-v2 postprocessor/scorer. Every comparison uses the user-specified fixed **0.9 threshold**, without an inner threshold search. The original eleven retain original + aligned voice-reduced training variants. The game-audio-only minute uses **original audio only**, as previously instructed. No new separator copies or cut/paste augmentation were made. The previously applied +20 dB preparation gain on that minute was not changed.

### Fair comparison — same 12 development clips

Both systems score the same held-out originals with identical target sets and timing. For each original clip, the old control trains on the other ten original clips; the new control trains on those same ten plus part 01. On the held-out part-01 fold, **both** train on only the original eleven: the new minute cannot train on its own scoring fold. Whole recording groups and their voice-reduced siblings stay outside validation. The old control is reconstructed from the unchanged round-2 procedure, rather than evaluated on clips its full-data model has seen.

| Model | Caught / targets | Recall | False captions | False/min | Arrows correct when shown | Arrow coverage | Unmatched/min |
|---|---|---|---|---|---|---|---|
| Old round-2 control | 20/48 | 41.7% | 31 | 4.90 | 8/10 (80.0%) | 50.0% | 13.12 |
| Round 5 (+part 01) | 21/48 | 43.8% | 32 | 5.06 | 9/11 (81.8%) | 52.4% | 12.96 |

Per-label rates use the same total 6.325 minutes, grouped by predicted label for false captions. Arrows are scored only for caught events with known left/right truth and a shown left/right arrow. No-arrow cases are reported through coverage, not silently counted as correct.

| Sound | Model | Caught / targets | Recall | False/min | Arrows correct when shown |
|---|---|---|---|---|---|
| footsteps | Old round 2 | 4/14 | 28.6% | 2.06 | 2/2 (100.0%) |
| footsteps | Round 5 | 3/14 | 21.4% | 2.53 | 2/2 (100.0%) |
| gunshot | Old round 2 | 16/34 | 47.1% | 2.85 | 6/8 (75.0%) |
| gunshot | Round 5 | 18/34 | 52.9% | 2.53 | 7/9 (77.8%) |

The class trade-off is explicit: **gunshots +2 caught and −2 false captions; footsteps −1 caught and +3 false captions**. Arrow changes are based on only 10 versus 11 scored arrows overall, so they are not evidence of a large direction improvement. Reload output remains disabled and explosion recall is unavailable because there are no explosion targets.

### Original eleven only — isolates the effect of adding training data

| Model | Caught / targets | Recall | False/min | Footsteps caught | Gunshots caught | Arrow correct |
|---|---|---|---|---|---|---|
| Old round-2 control | 15/34 | 44.1% | 4.13 | 4/5 (80.0%) | 11/29 (37.9%) | 7/9 (77.8%) |
| Round 5 | 16/34 | 47.1% | 4.32 | 3/5 (60.0%) | 13/29 (44.8%) | 8/10 (80.0%) |

Do not compare this old-control 44.1% directly to the earlier policy-v2 replay’s 47.1%: that earlier replay preserved the historical 0.8 threshold on clip 03 and 0.9 on ten others. This requested comparison fixes **both models to 0.9 on every fold**. The reconstructed original full-data head reproduces all stored round-2 head/scaler parameters **bit for bit**, and its original-11 outer predictions reproduce the saved round-2 fixed-0.9 legacy calibration curve exactly. This verifies the old control rather than silently changing its setup.

### New minute held out — not a training-set score

Both models give exactly the same held-out result on part 01, as expected because both use the original eleven on that fold: **5/14 caught (35.7%), 9.00 false captions/min, arrows 1/1 with only 20% arrow coverage**. Footsteps are 0/9, gunshots 5/5. This is not an after-training score on the minute itself. The identical result prevents the added clip from inflating its own measured benefit.

### Context breakdown, same twelve clips

| Context | Model | Caught / targets | Recall | False/min | Arrow correct |
|---|---|---|---|---|---|
| quiet | Old round 2 | 14/26 | 53.8% | 6.40 | 6/6 |
| quiet | Round 5 | 14/26 | 53.8% | 6.83 | 6/6 |
| talking | Old round 2 | 6/22 | 27.3% | 4.02 | 2/4 |
| talking | Round 5 | 7/22 | 31.8% | 4.02 | 3/5 |
| music | Old round 2 | N/A | N/A | N/A | N/A |
| music | Round 5 | N/A | N/A | N/A | N/A |

Quiet means outside the supplied talking/music spans; agent voice lines remain talking context. False captions require same-label span/onset support under policy v2. Unmatched counts additionally include later valid burst captions, so unmatched/min is not a wrong-caption rate. Onset recall stays at the original 1-second tolerance, with centered targets excluded from primary scoring. Full all-direction and confusion/context details remain in the machine-readable metrics.

Validation: **49 tests passed**, including explicit exclusion of unlabeled sibling parts and held-out recording groups. All 12 input annotation hashes, frozen source hashes and the old checkpoint hash were verified unchanged after the run. The current live helper still uses the unchanged original round-2 checkpoint. The newly fitted full-data round-5 checkpoint is saved separately under `.local/training-runs/round-5/model.joblib`; it was not deployed or used to score its own training clips.

Recorded artifacts: `training/round-5/protocol.json`, `metrics.json`, `fold-audit.json`, `verification.json`, `checkpoint.json`, and `conclusion.json`. Both systems’ complete outer probability arrays and caption lists were saved locally, so future threshold reviews do not require refitting. **No sealed tests opened. No new threshold, policy, encoder, direction, live-helper or overlay changes.**

## Training round 6 — add finished skirmish parts 02–04; fixed 0.9 and policy v2

**Footsteps helped a little:** on the same 12 clips used for the round-5 comparison, catches rose **3/14 → 4/14**, while false footstep captions fell **16 → 13**. This is one extra footstep and three fewer false footstep captions, not a large accuracy breakthrough. Overall catches stayed at 21/48: one extra footstep replaced one gunshot. Total false captions decreased, but arrow correctness worsened slightly.

Finished status was checked before any training: part 01 remains complete with 34 labels; parts 02, 03 and 04 are complete with **29, 33 and 35 labels**, respectively, with zero pending guesses. Their labels contain 41 footsteps, 38 gunshots, 13 talking spans and 5 reloads in total. Only these three reviewed clips were added to the round-5 training corpus. Parts **05 and 06 were explicitly excluded**: no audio, labels, feature windows or negative examples were loaded from them. No finish clicks were needed.

Setup unchanged: exact frozen YAMNet mono embedding, window/hop, ClassifierHead/scaler/seed/class weights, 0.9 threshold, original stereo direction and policy-v2 scheduler/scorer. The original eleven retain their existing original+voice-reduced training pairs. All four game-only skirmish parts use original audio only, retaining the preparation gain already applied. No new normalization, separator, cut/paste augmentation, threshold calibration, label edits, or other tuning. The full-data round-6 model is saved separately; the live helper remains unchanged.

### Same twelve clips — direct matched comparison against round 5

Primary comparison uses exactly the original 11 clips plus skirmish part 01, with the same truth labels/exposure for both versions. For each original held-out clip, round 5 trains on the other originals plus part 01; round 6 adds parts 02–04. When any skirmish minute is validated, **all four parts of the same recording are held out together**. Training on part 02 while validating adjacent part 03 would leak recording-specific audio and is forbidden. These are recording-group-out development predictions, not scores from either full-data model on its training clips.

| Measure | Round 5 | Round 6 |
|---|---|---|
| Caught sounds | 21 | 21 |
| False captions | 32 | 29 |
| Overall recall | 43.75% | 43.75% |
| False captions/min | 5.06 | 4.59 |
| Unmatched/min | 12.96 | 12.49 |
| Arrow correct % | 81.82% | 72.73% |
| Arrow coverage % | 52.38% | 52.38% |
| Arrows correct when shown | 9/11 | 8/11 |

| Label | Model | Caught / targets | Recall | False captions/min | Arrows correct when shown |
|---|---|---|---|---|---|
| footsteps | Round 5 | 3/14 | 21.4% | 2.53 | 2/2 |
| footsteps | Round 6 | 4/14 | 28.6% | 2.06 | 3/3 |
| gunshot | Round 5 | 18/34 | 52.9% | 2.53 | 7/9 |
| gunshot | Round 6 | 17/34 | 50.0% | 2.53 | 5/8 |

Per-label false/min is grouped by the predicted class and divided by the same full exposure. Footstep false/min improved **2.53 → 2.06**; gunshot false/min stayed **2.53**. The added correct footstep arrow changes footstep arrows from 2/2 to 3/3, while gunshot arrows change 7/9 to 5/8. These small samples do not justify a large direction reliability claim. Reload remains disabled and there are no explosion targets.

### Broader checks — no cherry-picking new scoring clips

| Shared evaluation scope | Model | Caught / targets | Footsteps caught | Gunshots caught | False/min | Arrows right |
|---|---|---|---|---|---|---|
| Original 11 only | Round 5 | 16/34 (47.1%) | 3/5 (60.0%) | 13/29 (44.8%) | 4.32 | 8/10 |
| Original 11 only | Round 6 | 16/34 (47.1%) | 4/5 (80.0%) | 12/29 (41.4%) | 3.76 | 7/10 |
| All 15 reviewed dev clips | Round 5 | 42/95 (44.2%) | 5/38 (13.2%) | 37/57 (64.9%) | 6.11 | 15/21 |
| All 15 reviewed dev clips | Round 6 | 42/95 (44.2%) | 6/38 (15.8%) | 36/57 (63.2%) | 5.79 | 14/21 |
| Entire skirmish recording held out | Round 5 | 26/61 (42.6%) | 2/33 (6.1%) | 24/28 (85.7%) | 8.50 | 7/11 |
| Entire skirmish recording held out | Round 6 | 26/61 (42.6%) | 2/33 (6.1%) | 24/28 (85.7%) | 8.50 | 7/11 |

On the larger matched 15-clip set, footsteps improve **5/38 → 6/38 (13.2% → 15.8%)**, so detection remains weak overall. Original-11 footstep recall improves 3/5 → 4/5, but that is still just one extra event. The skirmish-only held-out predictions are identical between versions: both train only on the original eleven when the entire new recording is held out. The skirmish score therefore cannot demonstrate transfer between adjacent pieces of the same recording; it deliberately avoids that leakage.

### Context on the primary same-12 comparison

| Context | Model | Caught / targets | Recall | False/min | Arrows right |
|---|---|---|---|---|---|
| quiet | Round 5 | 14/26 | 53.8% | 6.83 | 6/6 |
| quiet | Round 6 | 13/26 | 50.0% | 6.40 | 5/5 |
| talking | Round 5 | 7/22 | 31.8% | 4.02 | 3/5 |
| talking | Round 6 | 8/22 | 36.4% | 3.52 | 3/6 |
| music | Round 5 | N/A | N/A | N/A | N/A |
| music | Round 6 | N/A | N/A | N/A | N/A |

Policy-v2 definitions are unchanged: onset recall within 1 second, centered targets excluded from primary recall, same-label span/onset support for false captions, two-second same-sound/side quiet-gap merging, and no reload output. Later supported captions can be unmatched without being false. Quiet is the complement of the supplied talking/music spans; agent speech remains talking context. Full per-label, all-direction, confusion and context tables are in the machine-readable metrics.

Verification: **51 tests passed**. The reconstructed round-5 control reproduces its saved outer probability arrays, full head/scaler parameters, and primary same-12 scores **bit for bit**. Frozen code/label hashes and both saved round-5/live checkpoints remained unchanged. Every validation fold excludes the held-out recording group, including all four skirmish siblings where relevant. Parts 05–06 and all sealed tests remain closed.

Artifacts: `training/round-6/protocol.json`, `metrics.json`, `fold-audit.json`, `checkpoint.json`, `verification.json`, and `conclusion.json`. Both controls’ outer probabilities/captions and the full-data new model are saved under `.local/training-runs/round-6/` for reproducibility. No live-model or overlay switch was made and no additional training/tuning was attempted.

## Round-6 probability review — separate footstep thresholds; no new training

The user requested lower footstep thresholds using **saved round-6 whole-recording-out probabilities**, with gunshots kept at 0.9. The sweep reads cached features/probabilities only: no encoder inference, fitting, voice separation, new labels, or test access. It reuses policy v2 and the frozen raw two-caption scheduler. Original/new skirmish recording grouping remains as frozen in round 6; parts 05–06 are excluded.

| Footstep threshold | Footsteps caught | Recall | False footstep captions/min | Gunshots caught (threshold 0.9) | Total false/min |
|---|---|---|---|---|---|
| 0.90 | 6/38 | 15.8% | 1.61 | 36/57 | 5.79 |
| 0.80 | 7/38 | 18.4% | 2.79 | 33/57 | 6.86 |
| 0.70 | 9/38 | 23.7% | 3.97 | 32/57 | 8.04 |
| 0.60 | 8/38 | 21.1% | 4.72 | 32/57 | 8.58 |
| 0.50 | 10/38 | 26.3% | 5.04 | 32/57 | 8.69 |
| 0.40 | 10/38 | 26.3% | 5.68 | 31/57 | 9.22 |
| 0.30 | 10/38 | 26.3% | 5.68 | 30/57 | 9.44 |
| 0.20 | 10/38 | 26.3% | 6.43 | 29/57 | 10.30 |
| 0.10 | 14/38 | 36.8% | 7.19 | 27/57 | 11.05 |
| 0.05 | 14/38 | 36.8% | 6.97 | 28/57 | 10.83 |

**Decision: no lower threshold adopted.** At 0.8, one additional caught footstep costs eleven additional false footstep captions over the same development exposure; at 0.7, three extra catches cost twenty-two extra false captions. At 0.1, eight extra catches cost fifty-two extra false captions. This does not satisfy the requested large recall improvement without excessive false captions. The user then explicitly chose **“Keep round 2 at 0.9”** rather than switching to round 6 at 0.9. The live helper therefore retains round 2 with both footsteps and gunshots at 0.9. No checkpoint bytes were changed and training is finished for this phase.

These are exploratory development-calibration curves, not an unbiased score after selecting a threshold. The fixed-0.9 row reproduces the recorded round-6 15-clip primary metrics exactly. Both the all-15 and original-comparison-12 breakdowns are saved in `training/round-6-threshold-review/metrics.json`. A lower footstep threshold can alter gunshot emissions through the existing two-slot scheduler even though the gunshot threshold stays 0.9; de-duplication/capacity also explain non-monotonic rows. No cherry-picked test result was used.

## Step 5 — approved purple option-1 overlay, three-second display

Built the user-approved style and placement: 52 px bold white Arial at 1920×1080, opaque black outline/halo, and **#9146FF behind only the sound name**. The stack is centered, with its lower edge 17% above the bottom (option 1, “Near speech captions”), leaving the usual bottom speech-caption area free and keeping the minimap/crosshair clear. The production page has a transparent background and no speech-spacing placeholders. The number, brackets, arrow and direction words remain visible. White/purple contrast is 4.6419:1 and white/black is 21:1 at full opacity.

The overlay shows at most two captions for **up to 3 seconds**, fading over the last 250 ms. A third replaces the oldest, so the two-caption limit wins if needed. Recognition retains policy v2’s frozen 2-second raw scheduler and repeat-merging rule; display lifetime is separate. Disconnects, pauses, seeks and resets clear stale captions. Reconnect snapshots contain only the newest two unexpired captions and preserve their remaining lifetime. Reduced-motion settings turn off fade animation.

Implementation: `overlay/` production HTML/CSS/JS, local whitelist server `overlay_server.py`, and verified active settings in `live-config.json`/`deployment.py`. The helper automatically starts or reuses its own overlay HTTP server. OBS Browser Source URL: **http://127.0.0.1:8771/overlay**, 1920×1080, above the media/game source. Model events still use the local receive-only WebSocket on port 8770. No stream is started and no OBS scene is modified automatically. Instructions are in `OVERLAY.md`; the existing `Try helper.command` launcher starts the helper plus overlay.

Validation: **55 tests passed**. Real-browser checks used the actual production source to verify transparent compositing over an untouched development frame, exact approved placement, 52 px font, white/#9146FF colors, max-two replacement, three-second expiry with final fade, unknown directions without arrows, disabled reload/invalid-label rejection, no HTML injection, duplicate suppression, pause/reset clearing, and reconnect expiry preservation. There were no JavaScript errors. A separate end-to-end run used the real saved round-2 model on development clip 03 through the WebSocket into the production browser overlay; it displayed actual model captions, verified three-second event lifetimes, and cleared when the helper stopped. Native OBS scene compositing still needs the user’s local check; no Twitch broadcast was run.

The final-only evaluator now reads the frozen active deployment profile rather than hardcoded model metadata. It still refuses to open tests without explicit final authorization. Both the cached pretrained baseline and the selected learned model will use the same v2 postprocessor/scorer. **No final evaluation was run; all sealed test clips remain closed.**

## Final test — original-audio Valorant, round 2 at 0.9 (2026-11-02)

Final evaluation was explicitly authorized. This selected report uses the **same four held-out Valorant clips with their original audio** for both models. The before model is plain pretrained YAMNet with its frozen baseline settings; the after model is the selected **round-2 checkpoint at 0.9**. Both use the same policy-v2 postprocessor and scorer. No fitting, threshold changes, or test-based tuning were performed. These are actual model predictions, not the hand-label demo replay.

| Metric | Before: pretrained YAMNet | After: round 2 at 0.9 |
|---|---|---|
| All enabled sounds caught | 4/20 (20.0%) | 13/20 (65.0%) |
| All enabled sounds false captions/min | 2.62 (6 total) | 10.94 (25 total) |
| Gunshots caught | 4/19 (21.1%) | 13/19 (68.4%) |
| Gunshots false captions/min | 0.00 (0 total) | 3.50 (8 total) |
| Footsteps caught | 0/1 (0.0%) | 0/1 (0.0%) |
| Footsteps false captions/min | 1.75 (4 total) | 7.44 (17 total) |
| Arrow accuracy | 3/3 (100.0%) | 5/6 (83.3%) |

**68.4% is gunshot recall; overall recall is 65.0%.** The model catches more sounds but produces more false captions: **2.62 → 10.94 per minute**, so it does not meet the two-false-captions-per-minute goal. There is only **one non-centered footstep target**; 0/1 is not a reliable general estimate of footstep performance. After-model arrow accuracy is 5/6 among emitted arrows, with arrows supplied for 6 of 13 matched directional targets.

| Original-audio context | Before caught | After caught | Before false/min | After false/min |
|---|---|---|---|---|
| quiet | 2/7 (28.6%) | 5/7 (71.4%) | 4.48 | 17.92 |
| talking | 2/13 (15.4%) | 8/13 (61.5%) | 1.86 | 8.04 |

Primary recall excludes centered targets, and reload captions remain disabled. “Quiet” means outside the supplied talking/music spans, not verified silence. There are no music spans in these four test clips. The selected final report contains only original-audio Valorant results; the later voice-removal experiment is not the chosen result.

Artifacts: `evaluations/final-valorant-selected.json` (selected report), with frozen predictions and protocol in `evaluations/final-policy-v2/`. Historical entries above describe their state at the time; this final evaluation supersedes their “tests remain closed” status.
