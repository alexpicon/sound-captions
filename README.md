# Sound Captions

Important game sounds, captioned for deaf and hard-of-hearing viewers and people watching muted.

**Labeling and the round-0 baseline are complete. Step 3 ran three training rounds and stopped at the agreed checkpoint:** best development validation recall is 47.5%, below 50%. All eleven stereo voice-reduced copies are verified. Sealed tests remain closed; no final after score, live helper, overlay, or Twitch bot has been run. Details are in `rounds.md` and `training/decision.json`. No files are uploaded and nothing connects to Twitch.

## Try the labeling tool

Python 3.10+ and FFmpeg are required. Both are already available on the development Mac. The web server needs no extra packages; the model tools use a separate Python 3.11 `.venv-model` environment, already installed on this Mac.

```sh
cd /Users/alexnpc/hackathons/sound-captions
python3 labeler.py
```

Open **http://127.0.0.1:8765** in Chrome or Safari. Keep the terminal open; Ctrl+C stops the server. Use `python3 labeler.py --port 8766` if the port is busy. On another machine, install Python and FFmpeg first and put the untracked media back at the paths in `splits.json`.

1. Pick **Training clips · review model guesses**, then a clip. Wear headphones to hear left versus right. Existing Xonotic CSV labels are preserved; the eleven unlabeled Valorant development videos have pretrained suggestions.
2. Play/pause and find the beginning of a sound. Click **Set start here**.
3. Find its end and click **Set end here**. You can also type precise times.
4. Choose the sound and direction, then **Replay selection** to check it.
5. Click **Add annotation**, then **Save CSV**. Saving writes a four-column CSV beside the original media file. Existing labels reopen automatically. Model guesses stay separate until explicitly confirmed.

Unconfirmed suggestions have their own status and model score (not an accuracy percentage). Use **Play**, **Confirm**, **Edit** then **Confirm corrected annotation**, or **Delete**. Changes to suggestion decisions have Undo and persist when you save. No bulk-confirm action is provided. Add events the model missed, including reloads: the supplied YAMNet class list has no reload class.

**Save CSV** keeps partial progress without approving the whole clip for training. Once every guess is resolved and you have listened through the entire clip, click **Finish reviewing clip**. This saves confirmed labels, marks the full clip reviewed, and queues an aligned stereo voice-reduced copy in the background. The status appears beneath the buttons. Any later label edit invalidates completion until you finish again. Completion applies to the whole clip, including background intervals, so missing annotations do not silently become training negatives.

To label the answer keys, choose **Label test clips · no model guesses**. This mode lists only Valorant 01, 02, 05, 10 and CS 16, plays the original mixed audio, and allows manual CSV labels. It never reads model suggestion caches, shows model scores, runs voice removal, or supplies training examples. Preparing answer labels is distinct from evaluating a model: baseline/final model scoring remains sealed. `clip_voice_final` already has labels and stays inaccessible in both modes.

Use row buttons to play, edit, or delete an annotation. **Undo** restores annotation changes during this clip session. Saving an existing CSV first copies its exact previous contents into `.local/backups/<clip-id>/`. Multiple browser windows cannot silently overwrite each other's saves. Switching clips or closing the page warns about unsaved list changes. Editing fields is a draft: click Add/Update before saving the CSV.

**Choose a layout:** click **Review** (full-height video, editor, and a wider guesses panel), **Side by side** (three columns), or **Big video** (large playback beside the other two panels). Each box scrolls independently and the layout adapts to smaller windows. The selected preset is remembered in this browser; old floating-panel positions are ignored. Clicking Edit keeps the layout in place and focuses the timing fields. Switching presets preserves playback, draft edits, and unsaved annotations. There are no move grips or overlapping floating windows.

Keyboard shortcuts when focus is outside a form, button, or video control: **Space** play/pause; **I** start; **O** end; **R** replay selection; **A** add/update; **←/→** seek two seconds. **Cmd/Ctrl+S** saves from anywhere. Playback supports 0.5×, 0.75×, and 1× and preserves stereo channels.

## Labeling conventions

CSV header stays `start_s,end_s,label,direction`. Times are seconds, up to three decimals.

- Game labels: `footsteps`, `gunshot`, `reload`, `explosion`.
- Directions: `left`, `right`, `center`, `unknown`. Use `unknown` when you cannot reliably hear a direction; do not guess. Later arrow metrics must report coverage and exclude unknown ground truth from directional accuracy.
- Label centered sounds too, including your own sounds. Later caption output will hide centered sounds by default, with a setting to show them without arrows. Keeping them labeled lets us measure suppression honestly.
- Treat one short, continuous burst of footsteps or gunfire as an event rather than annotating every footstep or bullet. Mark its actual start and end. Overlapping game sounds can have separate rows.
- Add `talking` and `music` intervals, with direction `unknown`. They can overlap game sounds and each other and are context for scoring, never caption classes. Quiet means neither talking nor music. Later metrics should include an event in both context slices if both apply.
- Leave silence/background without a target sound unmarked; the scorer/trainer will derive `nothing` intervals. Unlabeled does not mean verified silence until the full clip has been reviewed.

The original synthetic CSVs do not include talking/music intervals. Add these during review; synthetic speech is not automatically labeled. Two `clip_train0.csv` event end times were clipped from 120.26 and 120.64 to the actual 120.00-second duration, with their original start times retained.

## Frozen data roles

`splits.json` records whole-clip roles. Development mode cannot access test media or labels. The separate test-labeling mode exposes the five unlabeled real answer keys only, with no model assistance. The final lab clip remains inaccessible. No model evaluation has occurred.

| Role | Data | Use |
|---|---|---|
| Valorant development | 11 videos: 03, 04, 06–09, 11–15 | Label now; later leave-one-clip-out validation for model choices |
| Sealed Valorant test | 01, 02, 05, 10 | Pretrained before baseline and final selected after comparison only |
| Xonotic second-game development | 9 synthetic clips, including `clip_voice_heldout` despite its historical filename | Reserve for step 7 adaptation; do not train the initial Valorant model on these |
| Sealed lab test | `clip_voice_final` | Before/final comparison only; never training or tuning |
| Sealed unseen-game test | 16, CS | Never training or tuning; report unseen-game transfer |

The four Valorant holdouts were selected using the supplied descriptions, before evaluation: 01 is a hype/talking candidate, 05 a quiet clutch candidate, 02 a long busy fight, and 10 a defuse scenario. Their audio conditions need independent human annotation in test-labeling mode; those descriptions are not verified context labels. The selection stays frozen. Model scoring is still limited to the before baseline and final comparison.

Xonotic provides nine development clips, close to the requested ten. Its synthetic mixes reuse the same 19 source sounds, so its eventual score demonstrates adaptation to that controlled library, not proven generalization to fresh gameplay. Keep source-family overlap in mind even with whole-clip splitting.

## Local model setup and training copies

Already installed here; to reproduce on another machine with [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.11 .venv-model
uv pip install --python .venv-model/bin/python -r requirements-model.txt
.venv-model/bin/python model_tools.py suggest
```

On Windows, use `.venv-model/Scripts/python.exe` instead. `suggest` only accepts development IDs, defaults to the eleven Valorant development clips, and skips media with existing human CSV rows or cached suggestions. It uses the supplied YAMNet TFLite through [Google's LiteRT interpreter](https://developers.google.com/edge/litert/migration). Fixed proposal thresholds and model fingerprints are recorded in `.local/suggestions/`; these are labeling aids, not tuned caption settings or reported accuracy. It checks mono and side inputs for game sounds, original mono for talking/music, and original stereo energy for tentative directions.

After you finish a development clip, a single background worker uses [Demucs htdemucs](https://github.com/facebookresearch/demucs) to estimate its vocals and subtract them from the original stereo audio. Weights download into `.local/models` on first use. This is **voice reduction**, not guaranteed speech removal: separation may also alter game sounds, and matching timestamps does not prove every target sound survived. The first labeled training round should compare original-only against original-plus-cleaned on original mixed validation audio and log the result honestly.

Generated copies live under `.local/augmentation/<clip-id>/voice_removed.wav`. They retain both channels and the exact decoded timeline sample count at 44.1 kHz, with no leading trim or intentional offset. Only tail padding/truncation aligns audio length to the media timeline. Original CSVs are shared by reference; there is no second answer key to drift. Talking/music rows remain original context, and are not reused as sound targets for cleaned copies. Worker failures appear in the UI and `.local/augmentation/<clip-id>/worker.log`; click Finish again to retry.

To export reviewed paired training examples for a chosen training fold:

```sh
.venv-model/bin/python model_tools.py training-manifest --validation-id CLIP_ID
```

The output is `.local/training-manifest.json`. `review.training_pairs` excludes validation IDs before adding either copy, excludes every sealed test, and keeps Xonotic in its separate game group. Both variants share `clip_id` / `fold_group` and target timestamps. Validation/scoring uses the original mixed clip, never a cleaned training sibling. A clip enters the pool only when fully reviewed, with a matching CSV revision and ready augmentation. External CSV edits invalidate that eligibility. No classifier is trained automatically during labeling: training and validation rounds remain step 2 onward, after human labels are ready.

Review decisions, suggestions, separation files, and model environments are local and ignored by Git. Back them up with the media if moving machines; CSV labels alone remain readable, but review completion must be re-established if the local review state is absent.

## Agreed implementation direction

- Aim to finish implementation by **November 1, 2026**, leaving demo preparation time. The submission deadline is believed to be November 2 and still needs confirmation.
- First live demo: this Mac plays the user's Twitch clips as an OBS Media Source, and the helper captions that source in real time. Capture only that source's audio; avoid the microphone and audio feedback. A Windows game-audio capture adapter follows later. The audio/model pipeline should stay independent of its capture adapter.
- Preserve stereo for direction. Compare mono, side (`L−R`), and combined recognition inputs in development validation. Side audio can suppress identical centered speech; it is not itself a left/right direction detector.
- Keep centered captions hidden by default, with an eventual setting to show them without arrows. Unknown direction is not proof of a centered sound.
- Log development validation scores in `rounds.md` at step 2 onward. Open sealed sets only for the pretrained before baseline and final selected after comparison, with no tuning from their errors.
- YAMNet has no explicit reload class in the supplied map. Report that limitation rather than inventing baseline performance.
- Stop after each build step. Before implementing any stream overlay UI, show a mockup and wait for approval. The labeling workspace is not that overlay.

See `plan.md` for the full original brief. These agreements reflect the follow-up instructions and take precedence where the original plan differs.

## Round 0 baseline

The user-authorized plain-YAMNet before baseline was run once on the four held-out Valorant clips and unseen CS clip on October 31, 2026. Results and exact caveats are in `rounds.md`; frozen configuration and file/code fingerprints are in `baseline-config.json` and `evaluations/round-0/`. No training, voice separation, side-channel recognition, or test-based tuning was performed. The final lab clip remains closed.

`scoring.py` is the reusable event scorer for subsequent rounds. It uses one-to-one same-label onset matching within one second and reports both the brief's literal false-captions/min and the broader unmatched-captions/min, plus arrow coverage. Center-hidden primary recall is accompanied by all-direction recall. Quiet/talking/music slices are unavailable until manual context labels are complete; missing context spans are not assumed quiet. Score caches under `.local/evaluations/round-0/` allow later context-only rescoring without changing baseline predictions.

The baseline runner is `.venv-model/bin/python run_baseline.py`. It **refuses to overwrite round 0**. Do not remove that guard, delete the outputs to rerun, or use test results to select thresholds. Future model choices and rounds use development validation, with test model evaluation next allowed for the final comparison.

Run the complete test suite, including the NumPy-dependent caption-policy checks, with:

```sh
.venv-model/bin/python -m unittest discover -s tests -v
```

There are 24 tests after step 2. Training clips can continue to be labeled independently; this evaluation did not alter their workflow.

## Training rounds and checkpoint

The predeclared experiment is in `training-plan.json`. It uses nested leave-one-clip-out on the eleven finished Valorant development clips, keeping each original and voice-reduced sibling together. Every outer validation clip is excluded from head fitting, feature scaling, and threshold selection. Thresholds are calibrated by inner leave-one-clip-out on the other ten clips, using the existing round-0 event scorer and caption scheduler.

The original-only YAMNet control scored 47.5%, paired YAMNet 47.5%, and paired EfficientAT with mono/side features 32.5%. The comparable untrained **development** reference was 12.5%; the earlier held-out test baseline is a different corpus. False-caption rates remain above the 2/min goal. Results include per-label recall, confusion counts, arrow accuracy/coverage, and manual quiet/talking context slices. No music spans or explosion targets were annotated in the development set, so those corresponding scores are N/A.

**Training stopped below 50%; no release model was selected.** Local classifier heads are experimental checkpoints. Do not use this as permission to run the final tests or proceed to live-helper work. Discuss a revised approach first. The test answer keys changed after round 0; a future final comparison must score cached baseline predictions and the final model against the same current answer keys, preserving the historical baseline report.

To reproduce this environment on another machine, install `requirements-model.txt`, restore media, review state and aligned copies, then obtain the official [EfficientAT source](https://github.com/fschmid56/EfficientAT) at the recorded revision (MIT license in `third_party/EfficientAT-LICENSE`):

```sh
git clone https://github.com/fschmid56/EfficientAT.git .local/vendor/EfficientAT
git -C .local/vendor/EfficientAT checkout a425fdce92572e602a1d5634799bd9f1f2efa806
```

Commands are `.venv-model/bin/python run_training.py 0` (development reference), then rounds `1`, `2`, and `3`. **Already-recorded rounds refuse to overwrite their artifacts. Do not rerun the current experiment or open sealed tests to tune.** Results/protocols are under `training/round-N/`, the augmentation audit is `training/augmentation-audit.json`, and fitted heads/cached features remain local under `.local/`. `training/environment.json` records package versions. CPU feature extraction timings measure offline throughput, not live end-to-end delay.

```sh
.venv-model/bin/python -m unittest discover -s tests -v
```

All 28 tests passed before training. The stop report also records post-run checks that the eleven reviewed annotation/copy hashes and outer-fold exclusion lists stayed correct.

## Repository and checks

Media, shortcut links, model binaries, `.env` credentials, and local backups are ignored by Git. CSV annotations, the class map, and source code are tracked candidates. No remote has been created and nothing has been published. `.env.example` contains empty placeholders for the future Twitch bot; credentials are unnecessary for step 1.

```sh
python3 -m unittest discover -s tests -v
```

The labeling tests use temporary generated stereo audio and exercise CSV validation/round trips, backups, stale-save conflicts, held-out isolation, HTTP saving, byte ranges, review decisions/completion, blind answer-key labeling, and paired training-fold isolation. Step 2 adds event-scoring and caption-policy tests. Step 6 will add chat-command tests. Baseline accuracy is recorded in `rounds.md`.

The Xonotic source/licensing notes remain in `test-audio/README.md` and `test-audio/xonotic-sounds/COPYING`. Preserve attribution and review distribution requirements before shipping third-party assets; the audio itself is not committed.
