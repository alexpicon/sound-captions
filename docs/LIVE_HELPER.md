# Live audio helper

The helper processes stereo game audio and sends captions to the local OBS overlay. It can analyze a recording, follow an OBS Media Source, or read a manually selected stereo input device. It does not start a broadcast or change OBS scenes.

Complete the [quick start](../README.md#quick-start) first. Run all commands below from the repository root with the Python environment activated.

## Local recording

```sh
python live_helper.py --file /absolute/path/to/gameplay.mp4
```

The file is analyzed at normal speed; this command does not play its audio or video. Display the result using the [OBS browser overlay](OVERLAY.md).

On macOS, `Try helper.command` offers the same source choices interactively. Choose your own recording; no gameplay recordings are included in the repository.

## Follow an OBS Media Source

1. Add a local gameplay recording as a Media Source in OBS.
2. Enable **Tools → WebSocket Server Settings** in OBS.
3. Copy `.env.example` to `.env` and set `OBS_WS_PASSWORD`. Keep `.env` private.
4. List the inputs, then select a source by its exact name:

```sh
python live_helper.py --list-obs-inputs
python live_helper.py --obs-media 'Your Media Source Name'
```

The helper follows the file's playhead, including pause, seek, restart, file changes, and connection loss. It analyzes the original stereo recording. OBS mixer levels, filters, and other sources do not affect that audio; speech already recorded in the file remains present.

## Live stereo input

Route game audio into a stereo input or loopback bus, then select it explicitly:

```sh
python live_helper.py --list-devices
python live_helper.py --device 7
```

Replace `7` with the actual device ID. Keep microphone, music, and other apps out of the bus if you want game-only input. The helper rejects mono devices and never selects a default microphone automatically. It resamples audio to 16 kHz while preserving stereo information for direction estimation.

Game-only capture requires manual audio routing. The OBS recording workflow was used for the demo; routed hardware capture and Windows setup need validation on the target machine.

## Hand-label replay

```sh
python live_helper.py --obs-media 'Your Media Source Name' --replay-labels
```

This mode reads the same-named CSV next to the selected recording. Required columns are `start_s,end_s,label,direction`; direction is `left`, `right`, or `center`. It replays sound labels without running the recognition model. An optional same-named SRT supplies speech subtitles.

To follow the visible Media Source as OBS scenes change:

```sh
python live_helper.py --follow-scene --replay-labels
```

Scenes with no single visible Media Source clear captions and wait. Hand-label replay is the mode shown in the demo; it is separate from the held-out model evaluation.

## Recognition and timing

The selected model is round 2, using a 0.9 threshold. YAMNet embeddings feed the trained classifier, while stereo levels supply direction. Reload captions are disabled. Centered sounds are hidden by default; `--show-centered` enables captions without an arrow.

Audio windows cover 975 ms and advance every 480 ms. Processing and input polling add latency. Check synchronization on your own setup and adjust OBS video delay if needed. Replay mode follows annotation timestamps directly and does not need an inference delay.

The internal recognition scheduler uses a two-second lifetime. The overlay independently displays up to two captions for up to three seconds, including a 250 ms fade. Same-label/side bursts merge until a two-second gap. Pausing, seeking, or disconnecting clears old captions.

## Ports and diagnostics

- Overlay: `http://127.0.0.1:8771/overlay`
- Receive-only caption events: `ws://127.0.0.1:8770`
- `--json`: machine-readable events.
- `--seconds 10`: stop after a short run.
- `--port` and `--overlay-port`: select alternate local ports.

Events include `snapshot`, `caption`, `speech`, `reset`, `status`, and `performance`. Reconnecting consumers receive only captions that have not expired. The event socket does not accept file edits, training commands, or OBS control commands.

Held-out recordings remain blocked in the live helper to keep demonstrations separate from evaluation. See the [selected results](../evaluations/final-valorant-selected.json) and [experiment history](../rounds.md).
