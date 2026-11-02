# Development guide

Commands run from the repository root. Start with the [runtime setup](../README.md#quick-start). Training and voice-reduction tools additionally require:

```sh
pip install -r requirements-model.txt
```

## Project structure

| Area | Files |
| --- | --- |
| Annotation UI and HTTP server | `labeler.py`, `labeler-ui/` |
| Annotation review / voice-reduced training pairs | `review.py`, `model_tools.py` |
| Audio features and classifier training | `training_core.py`, `run_training.py` |
| Additional development experiments | `cutpaste.py`, `run_round4.py`, `run_round5.py`, `run_round6.py` |
| Frozen deployment and streaming inference | `deployment.py`, `live_engine.py`, `live-config.json` |
| Caption policy and scoring | `caption_policy.py`, `policy_scoring.py`, `scoring.py` |
| OBS connection and hand-label replay | `obs_source.py`, `label_replay.py`, `live_helper.py` |
| Browser display | `overlay/`, `overlay_server.py` |
| Before/final evaluation | `run_baseline.py`, `run_final.py` |
| Recorded evidence | `training/`, `evaluations/`, `rounds.md` |

## Labeling

```sh
python labeler.py
```

Open `http://127.0.0.1:8765`. Media files are not bundled; place your recordings at the paths in `splits.json` to work with the supplied corpus annotations.

Each CSV uses `start_s,end_s,label,direction`. Sound labels are `footsteps`, `gunshot`, `reload`, and `explosion`; `talking` and `music` describe evaluation context. Directions are `left`, `right`, and `center`. Unlabeled time is not verified silence until the full clip has been reviewed.

The UI supports hand labeling, reviewing model suggestions, undo, and saving corrections. Suggestions are labeling aids, not evaluation results. Blind test-labeling mode does not expose model suggestions.

## Data separation

[`splits.json`](../splits.json) records whole-clip roles. Model choices use development validation; held-out test clips never enter training or threshold selection. Voice-reduced training copies stay in the same fold as their originals. Validation and the selected final evaluation use original mixed audio.

The initial Valorant development set contains eleven clips. Later experiments add reviewed skirmish recordings using recording-group isolation. The four final Valorant holdouts are clips 01, 02, 05, and 10. Xonotic and Counter-Strike experiments are outside the selected submission result.

Source links and media notes are in [`test-audio/`](../test-audio/). Full training reproduction requires the source recordings, reviewed completion state, and aligned augmentation caches. These large local inputs are not included in Git.

## Selected model and recorded experiments

The submitted model is the round-2 YAMNet classifier, with the final v2 caption policy and 0.9 threshold. Run `python scripts/setup_models.py` to verify and install its bundled checkpoint. [`models/README.md`](../models/README.md) records the hashes and encoder attribution.

Training rounds, threshold searches, and historical evaluation artifacts remain available for audit. They record the configuration and data available at the time; they are not all directly comparable. The selected result is [`evaluations/final-valorant-selected.json`](../evaluations/final-valorant-selected.json). See [the results guide](RESULTS.md) before interpreting other reports.

Experiment runners refuse to overwrite recorded outputs. `run_final.py` requires an explicit final-evaluation flag. Existing test results should be inspected without rerunning or tuning against their errors. Recognition and scoring source hashes are checked by the active deployment profile, so changing those files requires deliberate revalidation.

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests use generated audio and mock OBS/WebSocket peers to cover CSV validation and saving, fold isolation, event scoring, caption filtering, source lifecycle, replay, and overlay behavior. They do not require the original gameplay recordings. Optional dependency availability affects which checks can run; use the documented environment for the complete suite.

For a short inference check using your own stereo recording:

```sh
python live_helper.py --file /absolute/path/to/gameplay.mp4 --seconds 10
```

This checks the pipeline, not model accuracy. Local state and credentials belong in ignored `.local/` and `.env` paths.
