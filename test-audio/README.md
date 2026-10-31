# test audio

Current data roles are frozen in `../splits.json`; see the main README. Xonotic development clips are reserved for second-game adaptation and CS is an unseen-game test. Test model scoring stays sealed outside the before/final comparisons; separate blind manual labeling of the five real test answer keys is allowed. The collection notes below describe the original files.

quick notes on what's in here:

- `xonotic-sounds/`: 19 of Xonotic's own sound files (footsteps, guns, reload, explosions) that I grabbed from https://gitlab.com/xonotic/xonotic-data.pk3dir. they're GPLv3, license is in `xonotic-sounds/COPYING`
- `clips/`: 10 labeled 2-min stereo clips (48 kHz `.wav`) mixed from those sounds. some have fake voice chat (macOS `say`) in the center. every `.wav` has a `.csv` with one row per sound: `start_s, end_s, label, direction` (`left` / `right` / `center`)
  - `clip_voice_final`: scoring only. never train or tune on this one
  - the other nine clips: second-game development (not initial Valorant training)
- `model/`: YAMNet from Google (TFLite, 16 kHz mono, 15600-sample input, 521 class scores) + `yamnet_class_map.csv`
- `real-clips/`: my own Twitch clips, see the README in there

the `clips/` ones are computer mixes, not real gameplay, so don't trust scores on them too much. the real clips are the actual test.
