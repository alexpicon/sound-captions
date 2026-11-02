# Use the overlay in OBS

The active model is **round 2**, with footsteps and gunshots both at **0.9**, as chosen after reviewing the round-6 threshold sweep. Reload output stays disabled. No further training was performed.

1. Double-click **Try helper.command** in this project folder. Choose **1** to follow your OBS Media Source, or **2** for the familiar clip-03 quick test. The helper also starts the overlay server if needed.
2. In OBS, add a **Browser Source** above your game/video source in the Sources list.
3. Set its URL to **http://127.0.0.1:8771/overlay**, width **1920**, height **1080**, then fit it to the canvas. The page is transparent. Keep the Browser Source above the video source.
4. Play a development clip in the selected OBS Media Source. Captions appear in the approved **option 1 / near-speech-captions** position, centered with the lower edge 17% above the bottom. The space below is reserved for the usual bottom speech-caption area. Twitch viewers can move native speech captions themselves; the overlay cannot see a viewer's personal placement.

For a different 16:9 canvas, use that canvas's width and height; the styling scales with the browser viewport. If nothing appears, check that the helper says `listening`, the exact OBS Media Source name is selected, and the Browser Source URL includes `/overlay`. While the helper is stopped/disconnected, a blank transparent page is expected. Reload the Browser Source if its first load happened before the server started.

## What it shows

- `#14 [footsteps ← left]`, `#15 [gunshot → right]`, or an explosion label.
- Big bold white Arial text with an opaque black outline/halo. Only the sound name has the solid **#9146FF** background. No speech placeholders or connection notices appear on stream.
- **At most two captions.** Each stays for up to **3 seconds**, with a soft **250 ms fade at the end**. A newer third caption replaces the oldest to keep the screen clear. Unknown/center directions have no arrow; centered sounds remain hidden by default in the helper.
- Reload stays off; the existing 2-second same-label/side repeat-merging rule remains unchanged.

The v2 recognizer still uses its frozen 2-second raw scheduler so the saved development scores remain reproducible. The **3 seconds is the display lifetime**, separate from recognition/scoring. Newest-two display behavior is shared by the live hub's reconnect snapshots and the browser. Reconnecting does not restart a caption's remaining lifetime. Pausing/seeking/stopping the source clears old captions; disconnecting also clears the display. Reduced-motion settings disable animated fades.

At full opacity, white on #9146FF has **4.6419:1** contrast (meets the 4.5:1 WCAG AA normal-text threshold); white against its opaque black halo is **21:1**. The halo protects text over bright gameplay. The approved 1080p styling is 52 px bold, scaling with the canvas. The final 250 ms is the requested disappearance transition, not a claim of fixed contrast during fading. [W3C contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html).

## Starting from Terminal

```sh
cd /Users/alexnpc/hackathons/sound-captions
.venv-model/bin/python live_helper.py --obs-media 'Media Source'
```

Replace `Media Source` with the actual source name. OBS WebSocket setup and game-only stereo capture routing are described in `LIVE_HELPER.md`. The OBS file-follow mode uses original stereo from the selected local file; OBS mixer volume/pan/filter changes are not included. Use original stereo source settings for the clip demo.

The helper starts HTTP on port **8771** and caption WebSocket output on **8770**, both on loopback. To keep the transparent source server available independently:

```sh
python3 overlay_server.py
```

The helper recognizes and reuses its own existing server. It rejects a different app occupying the same HTTP port. The WebSocket is receive-only; it cannot change files, labels, training, model settings, OBS scenes or stream state. No Twitch account/credentials are needed for this local overlay, and no broadcast is started automatically.

## Quick check and timing

Choose option **2** in `Try helper.command`, then open the overlay URL in a browser or add it to OBS. This analyzes development clip 03 at normal speed; that mode does not play the video/audio for you. It is a plumbing test, not an accuracy score.

Model observation window remains 975 ms, advancing every 480 ms, plus a few milliseconds of inference and source/browser overhead. Start around a one-second OBS video delay and check actual alignment on your setup. This build does not promise exact synchronization for every detected onset or configure OBS delays automatically.

The actual model → WebSocket → production browser source was smoke-tested locally. Browser tests cover transparency, placement, size/colors, invalid-label/script-injection rejection, duplicate suppression, max-two replacement, 3-second expiry/fade, reset/pause clearing, and reconnect snapshots preserving remaining lifetime. Native OBS scene compositing and game-only capture hardware still need your local check. There is no chat bot yet; that is step 6.

## Frozen settings and final evaluation

`live-config.json` is the active, verified model/threshold/display profile. The trained checkpoint bytes, encoder, recognition core, v2 scoring policy and scorer are checked before the helper starts. Historical model files, policies and training logs were not overwritten.

The final-only runner reads this same active profile. The same v2 scoring/filtering rules will apply to the cached pretrained baseline and active learned model. The final command still requires explicit final authorization; **it has not been run**. All sealed test clips remain closed.
