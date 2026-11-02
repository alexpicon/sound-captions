> Historical development log. Relative project paths below are relative to the repository root. For current setup and final results, see the [main README](../README.md).

# Sound Captions

Important game sounds, captioned for deaf and hard-of-hearing viewers and people watching muted.

**The live helper and step-5 overlay are ready to try.** The active model remains round 2 with footsteps/gunshots at 0.9, as the user chose after reviewing the round-6 threshold trade-off. The approved purple option-1 overlay shows at most two captions for up to 3 seconds. Start `Try helper.command`, then add `http://127.0.0.1:8771/overlay` as a 1920×1080 OBS Browser Source above the video. See `OVERLAY.md`. No more training is planned; all sealed tests remain closed. The Twitch chat bot is still step 6.

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

## Round 4: fold-local augmentation and strict false-caption limit

The user authorized exactly one additional experiment after the earlier stop. Its frozen settings are in `round-4-plan.json`, with code in `cutpaste.py` and `run_round4.py`. It retains the paired mono YAMNet model, adds pasted sound snippets at three gains and two pans, and creates background-only controls. Both donor and recipient must be in the particular training fold before any cache or waveform is read. Overlapping sound labels move with the donor audio; original CSVs and validation audio are never edited.

This created 666 mixes and 3,996 synthetic/control feature windows for the full-data checkpoint. Nested validation filters the recipe set separately for every inner/outer training subset. Thresholds are chosen strictly among inner results with <=2 literal false captions/min; the search extends past 0.9 through 0.99999 plus an explicit abstain-all candidate. It never falls back to an infeasible threshold.

**Outcome:** held-out nested validation caught 3/40 sounds (7.5%) at 1.13 literal false/min and 3.57 unmatched/min, including wrong labels/repeats and excluded centered targets. Every inner calibration met the 2/min limit, but recall did not reach the new 35% gate. The separate full-data checkpoint calibration scored 10% at 1.69 false/min; this is not the independent validation score. **Step 4 was not started and all sealed tests remain closed.** Results, split provenance, calibration searches, and the gate decision are under `training/round-4/` and `training/decision-round-4.json`.

Reproduction command: `.venv-model/bin/python run_round4.py`. The recorded run refuses to overwrite its protocol/results. The complete log is `rounds.md`. All 32 tests passed before the run. Annotation hashes and all 67 training-subset provenance lists were verified afterward. Earlier rounds, their metrics, and the historical stop decision are preserved.

## Repository and checks

The selected final report is `evaluations/final-valorant-selected.json`: the four original-audio Valorant test clips, scored with round 2 at 0.9 and policy v2. `splits.json` limits active evaluation to Valorant. CS, the synthetic lab test, and the voice-removal experiment are excluded from the selected report; historical artifacts remain unchanged for reproducibility. Retired test clips remain sealed and never enter training.

### Hand-label replay for a demo

To show the current OBS media source's **human annotations**, run:

```sh
.venv-model/bin/python live_helper.py --obs-media clip --replay-labels
```

This reads the `.csv` next to the media file selected in OBS and follows its playhead, including pause, resume, seek, and restart. It skips talking/music, includes hand-labeled reloads and centered sounds, and uses the existing purple overlay with at most two captions lasting up to three seconds. Centered sounds have no arrow. No model or encoder is loaded; events identify themselves as `csv_replay`. This is a demonstration of the labeled captions, not a model accuracy result. Sealed clips remain blocked. Omit `--replay-labels` to return to the unchanged model-inference mode.

If a same-named `.srt` is present beside the clip, replay mode also shows its speech subtitles at their exact cue times. Speech uses a separate lower lane (40 px at 1080p); the two sound captions keep their upper lane and larger 52 px text. Both use bold white outlined text and purple boxes. Subtitles clear on pause, stop, disconnection or seeking outside a cue, and resync when playback resumes. SRT text, including `[ __ ]` censoring, is preserved. Clips without an SRT still replay sound captions normally.

For smooth demo playback, the current OBS scene has no deliberate video/audio delay. Do not reapply the old one-second delay filters: they caused a startup hold. The replay uses the clip's CSV timestamps directly.

For the two-scene demo, use `.venv-model/bin/python live_helper.py --follow-scene --replay-labels`. It follows the one visible Media Source in the **program (live) scene**, clears captions when scenes change, and waits safely on scenes with no single visible clip. `Squeaky` contains clip 13 with separate `Sound captions` and `Speech captions` browser sources, initially hidden. `Marshel` contains clip 08, `Sound captions`, and a separate `Speech captions` source hidden by default. Each source's eye icon controls it independently. The browser URLs are `/overlay?layer=sound` and `/overlay?layer=speech`; `/overlay` continues to show both layers.

Media, shortcut links, model binaries, `.env` credentials, and local backups are ignored by Git. CSV annotations, the class map, and source code are tracked candidates. No remote has been created and nothing has been published. `.env.example` contains empty placeholders for the future Twitch bot; credentials are unnecessary for step 1.

```sh
python3 -m unittest discover -s tests -v
```

The labeling tests use temporary generated stereo audio and exercise CSV validation/round trips, backups, stale-save conflicts, held-out isolation, HTTP saving, byte ranges, review decisions/completion, blind answer-key labeling, and paired training-fold isolation. Step 2 adds event-scoring and caption-policy tests. Step 6 will add chat-command tests. Baseline accuracy is recorded in `rounds.md`.

The Xonotic source/licensing notes remain in `test-audio/README.md` and `test-audio/xonotic-sounds/COPYING`. Preserve attribution and review distribution requirements before shipping third-party assets; the audio itself is not committed.
