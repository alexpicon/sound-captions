# OBS overlay

Start the [audio helper](LIVE_HELPER.md), then add a **Browser Source** above your gameplay in OBS:

- URL: `http://127.0.0.1:8771/overlay`
- Width: `1920`
- Height: `1080`

The background is transparent. Use your canvas dimensions for other 16:9 resolutions. The helper starts the local overlay server automatically.

## Appearance

Captions use bold white text with a black outline, a purple (`#9146FF`) background behind the sound label, and a direction arrow when available. At 1080p, sound text is 52 px. Captions sit above the lower speech-caption lane.

The overlay shows at most two sound captions at once. Each stays for up to three seconds, with a 250 ms fade. A third caption replaces the oldest. Pause, seek, stop, and disconnect clear stale text. Reconnecting preserves remaining display time rather than restarting it. Reduced-motion preferences disable fades.

## Speech and sound layers

Hand-label replay can also read a same-named SRT file next to the gameplay recording. Use separate OBS Browser Sources if you want independent visibility controls:

| URL | Visible content |
| --- | --- |
| `/overlay` | Sound and available speech captions |
| `/overlay?layer=sound` | Sound captions only |
| `/overlay?layer=speech` | Replayed SRT speech subtitles only |

All paths use `http://127.0.0.1:8771`. The speech layer displays supplied subtitles; it does not perform speech recognition. For live speech transcription, use your existing captioning workflow alongside the sound overlay.

## Troubleshooting

A blank overlay is normal while disconnected or when no captions are active. If captions do not appear:

1. Check that the helper reports `listening`.
2. Confirm the selected OBS Media Source name or audio device.
3. Check the Browser Source URL and its position above the gameplay source.
4. Reload the Browser Source if it loaded before the helper started.

For inference mode, check audio/video synchronization on your setup. Recognition uses a 975 ms observation window plus processing overhead. Hand-label replay instead uses exact annotation timestamps; adding an inference delay to that mode can misalign the demo.

To keep the HTTP server running independently:

```sh
python overlay_server.py
```

The server binds only to loopback and serves a fixed set of overlay files. The helper reuses a matching local overlay server and rejects a conflicting app on the same port. The overlay does not need Twitch credentials.

## Validation

The test suite covers caption expiration, replacement, replay synchronization, input validation, and WebSocket events. The OBS media-following and replay workflow was used in the recorded demo. Routed live game-audio capture still needs validation on the target machine.

The active model and display settings are in [`live-config.json`](../live-config.json). The final held-out model results are in [`evaluations/final-valorant-selected.json`](../evaluations/final-valorant-selected.json). Historical overlay checks in `overlay/checks.json` describe the earlier implementation state, not the complete final demo.
