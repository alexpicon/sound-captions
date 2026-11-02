# Live helper guide

The helper is ready for a local try. It prints captions in Terminal and sends the same events to `ws://127.0.0.1:8770`. The step-5 overlay is implemented; add `http://127.0.0.1:8771/overlay` as an OBS Browser Source (see `OVERLAY.md`). Nothing here starts a Twitch stream or changes OBS scenes/playback.

## Quick check

Double-click **Try helper.command** in the project folder and choose **2**. It runs your development clip 03 at normal speed and prints captions such as `#3 [gunshot → right]`. This mode analyzes the file; it does not play its sound/video. It is a plumbing test on familiar development audio, not a new accuracy result.

Or run in Terminal:

```sh
cd /path/to/sound-captions
.venv-model/bin/python live_helper.py --file 'test-audio/real-clips/03 SOBAAAA ACEEEEE.mp4'
```

## Follow the clip you play in OBS on this Mac

1. Open OBS and add one of your **development clips** as a local Media Source. Keep using development clips while the test clips are sealed.
2. In OBS, open **Tools → WebSocket Server Settings**, enable its WebSocket server, and leave it on the local port 4455. Copy `.env.example` to `.env` and put the OBS WebSocket password after `OBS_WS_PASSWORD=`. The helper reads that local file without executing it; it never prints the password.
3. Double-click **Try helper.command**, choose **1**, and enter the Media Source name printed by the helper. Start playing the clip in OBS. Caption lines appear in Terminal. Stop with Ctrl+C.

This mode follows the selected source's playhead and analyzes its original stereo file in real time. It follows pauses, seeks, loops, source-file changes and OBS connection loss, resetting stale captions. It does not capture the OBS master mix, add your current microphone, or include other desktop apps. Any voice already recorded in the clip remains in that file. OBS mixer volume/pan/filter changes are not applied to this file-follow mode; use the source's original stereo audio for the demo. This mode does not drive OBS or go live for you.

CLI equivalents:

```sh
.venv-model/bin/python live_helper.py --list-obs-inputs
.venv-model/bin/python live_helper.py --obs-media 'Media Source'
```

The OBS media-following and replay paths were used for the recorded demo. Automated tests additionally cover authentication, play/pause/seek, and connection loss against a local mock OBS server; model inference and outbound WebSocket delivery have been smoke-tested on this Mac. If OBS is closed the helper waits/reconnects; configuration/source-name errors are printed for correction.

## Future real game audio on Windows or Mac

Route **only the game** into a stereo input bus (for example a virtual audio cable on Windows or a stereo loopback bus on Mac). Keep microphone, music and other apps outside that route. Choose that bus explicitly:

```sh
.venv-model/bin/python live_helper.py --list-devices
.venv-model/bin/python live_helper.py --device 7
```

Replace `7` with the actual **game bus** ID. It is an example, not a device recommendation. The helper never picks a default input and rejects mono inputs. It preserves the two channels for direction and resamples on the processing thread to 16 kHz for inference. The capture queue is bounded; lost/stale audio resets state instead of accumulating delay. A suitable game-only bus was not present for hardware capture testing here, so that path still needs a routed-device check. It is not automatic Windows per-process capture.

## Model, captions and timing

The active model is the verified **round-2 checkpoint at 0.9**, kept by the user after the round-6 threshold review. Its settings live in `live-config.json`. No new fitting was done. Recognition uses the stereo average; original stereo supplies direction. Centered sounds stay hidden unless you add `--show-centered`, which shows them without an arrow. Reload captions are disabled. Same-label/side candidates merge until there is a full 2-second quiet gap. The frozen legacy two-slot scheduler runs before this shared postprocessor, matching the cached development replay; hidden reload candidates can still occupy an internal slot. The recognition scheduler still uses its frozen 2-second TTL. The approved overlay displays the newest two captions for up to 3 seconds, fading during the last 250 ms and replacing the oldest if a third arrives.

The model looks back over 975 ms of audio and advances every 480 ms. The console reports measured processing time on exit; processing time does not include the observation window or prove every detected sound appears within a second. Allow approximately one second plus source-polling/capture and processing overhead in the OBS video delay. Do a visual/audio sync check after adding the overlay. No live vocal separator is used.

`--json` prints full machine events. `--seconds 10` stops a smoke test after ten seconds. The local receive-only WebSocket emits `snapshot`, `caption`, `reset`, `status`, and `performance` events. Caption events include a session ID, numeric caption ID, fixed label, direction, text, confidence, source timestamp, wall-clock expiry, and timing. Reconnecting consumers receive only still-active captions. No chat messages, training commands, file-read commands, or credentials are accepted over this socket.

## Final evaluation

The final evaluation has been completed. The submitted report is [evaluations/final-valorant-selected.json](evaluations/final-valorant-selected.json), covering four original-audio Valorant holdouts. The helper continues to block held-out media to keep demonstration and training separate from evaluation. Historical experiment details are retained in `rounds.md`.

## Reproduce this development replay

`rescore_policy_v2.py` uses the cached outer-held-out emissions: ten folds retained 0.9 and clip 03 retained 0.8, exactly as originally inner-calibrated. It does not run the full-data model over its training clips for an accuracy claim. The recorded replay refuses to overwrite its results. Detailed metrics and protocol are under `training/policy-v2-round-2/`, with the plain explanation in `rounds.md`.

Install dependencies on another machine with Python 3.11 and the packages in `requirements-live.txt`; FFmpeg/ffprobe must be on PATH. The frozen encoder and classifier are bundled: run `python3 scripts/setup_models.py` after cloning to verify and install the classifier into its expected local path. On this Mac the dependencies and checkpoint are already installed. Windows paths/activation differ; Windows hardware capture has not yet been tested.

Implementation references: [OBS WebSocket v5 protocol](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md), [sounddevice input streams](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html), [websockets server](https://websockets.readthedocs.io/en/15.0.1/reference/asyncio/server.html).
