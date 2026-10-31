# Sound Captions

Important game sounds, captioned for deaf and hard-of-hearing viewers and people watching muted.

**Step 1 is ready: local labeling workspace.** Detection, training, live capture, overlay, and Twitch bot are future steps. Nothing in the labeling tool connects to Twitch, uploads files, or runs a model.

## Try the labeling tool

Python 3.10+ and FFmpeg's `ffprobe` are required. Both are already available on the development Mac. No Python or JavaScript packages are needed.

```sh
cd /Users/alexnpc/hackathons/sound-captions
python3 labeler.py
```

Open **http://127.0.0.1:8765** in Chrome or Safari. Keep the terminal open; Ctrl+C stops the server. Use `python3 labeler.py --port 8766` if the port is busy. On another machine, install Python and FFmpeg first and put the untracked media back at the paths in `splits.json`.

1. Pick a development clip. Wear headphones to hear left versus right.
2. Play/pause and find the beginning of a sound. Click **Set start here**.
3. Find its end and click **Set end here**. You can also type precise times.
4. Choose the sound and direction, then **Replay selection** to check it.
5. Click **Add annotation**, then **Save CSV**. Saving writes a four-column CSV beside the original media file. Existing labels reopen automatically.

Use row buttons to play, edit, or delete an annotation. **Undo** restores annotation changes during this clip session. Saving an existing CSV first copies its exact previous contents into `.local/backups/<clip-id>/`. Multiple browser windows cannot silently overwrite each other's saves. Switching clips or closing the page warns about unsaved list changes. Editing fields is a draft: click Add/Update before saving the CSV.

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

`splits.json` records whole-clip roles and blocks all six test media files and their labels from the labeling server. There is deliberately no unlock control. No model evaluation has occurred.

| Role | Data | Use |
|---|---|---|
| Valorant development | 11 videos: 03, 04, 06–09, 11–15 | Label now; later leave-one-clip-out validation for model choices |
| Sealed Valorant test | 01, 02, 05, 10 | Pretrained before baseline and final selected after comparison only |
| Xonotic second-game development | 9 synthetic clips, including `clip_voice_heldout` despite its historical filename | Reserve for step 7 adaptation; do not train the initial Valorant model on these |
| Sealed lab test | `clip_voice_final` | Before/final comparison only; never training or tuning |
| Sealed unseen-game test | 16, CS | Never training or tuning; report unseen-game transfer |

The four Valorant holdouts were selected using the supplied descriptions, before evaluation: 01 is a hype/talking candidate, 05 a quiet clutch candidate, 02 a long busy fight, and 10 a defuse scenario. Their audio conditions still need independent annotation when the baseline test is opened; those descriptions are not verified context labels. Freeze this selection before any modeling. Test labeling/scoring will be a separate explicit baseline/final workflow, not an unrestricted development-tool option.

Xonotic provides nine development clips, close to the requested ten. Its synthetic mixes reuse the same 19 source sounds, so its eventual score demonstrates adaptation to that controlled library, not proven generalization to fresh gameplay. Keep source-family overlap in mind even with whole-clip splitting.

## Agreed implementation direction

- Aim to finish implementation by **November 1, 2026**, leaving demo preparation time. The submission deadline is believed to be November 2 and still needs confirmation.
- First live demo: this Mac plays the user's Twitch clips as an OBS Media Source, and the helper captions that source in real time. Capture only that source's audio; avoid the microphone and audio feedback. A Windows game-audio capture adapter follows later. The audio/model pipeline should stay independent of its capture adapter.
- Preserve stereo for direction. Compare mono, side (`L−R`), and combined recognition inputs in development validation. Side audio can suppress identical centered speech; it is not itself a left/right direction detector.
- Keep centered captions hidden by default, with an eventual setting to show them without arrows. Unknown direction is not proof of a centered sound.
- Log development validation scores in `rounds.md` at step 2 onward. Open sealed sets only for the pretrained before baseline and final selected after comparison, with no tuning from their errors.
- YAMNet has no explicit reload class in the supplied map. Report that limitation rather than inventing baseline performance.
- Stop after each build step. Before implementing any stream overlay UI, show a mockup and wait for approval. The labeling workspace is not that overlay.

See `plan.md` for the full original brief. These agreements reflect the follow-up instructions and take precedence where the original plan differs.

## Repository and checks

Media, shortcut links, model binaries, `.env` credentials, and local backups are ignored by Git. CSV annotations, the class map, and source code are tracked candidates. No remote has been created and nothing has been published. `.env.example` contains empty placeholders for the future Twitch bot; credentials are unnecessary for step 1.

```sh
python3 -m unittest discover -s tests -v
```

The tests use temporary generated stereo audio and exercise CSV validation/round trips, backups, stale-save conflicts, held-out isolation, HTTP saving, and byte ranges for media seeking. Step 2 will add scoring tests; step 6 will add chat-command tests. No accuracy numbers exist yet.

The Xonotic source/licensing notes remain in `test-audio/README.md` and `test-audio/xonotic-sounds/COPYING`. Preserve attribution and review distribution requirements before shipping third-party assets; the audio itself is not committed.
