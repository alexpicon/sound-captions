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
