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
