# Gameplay annotation sources

These recordings come from my [Twitch channel, washedlatino](https://www.twitch.tv/washedlatino/clips?range=all). I used to stream Valorant and still watch Twitch. The original clips include my microphone audio, including loud speech during fights.

Only annotations are included in this repository. Restore the corresponding MP4 filenames from `splits.json` if you want to reproduce the original corpus. The source notes below record why each clip was initially selected; the event annotations, rather than these descriptions, define the evaluation targets.

| # | clip | length | why I picked it |
|---|---|---|---|
| 1 | [JETT KNIVES ACE KEKW](https://www.twitch.tv/washedlatino/clip/PiercingHungryDunlinPJSalt-5PaLGoltwL3JUB5w) | 0:23 | ace, lots of fighting |
| 2 | [jett ace](https://www.twitch.tv/washedlatino/clip/HotConfidentBaboonNinjaGrumpy-6W2ahGuP2ag6i3uU) | 0:59 | longest ace |
| 3 | [SOBAAAA ACEEEEE](https://www.twitch.tv/washedlatino/clip/DirtyKawaiiPanPeteZaroll-r_bUvE-2njhyLI56) | 0:30 | ace |
| 4 | [deagace](https://www.twitch.tv/washedlatino/clip/KindSpeedySpaghettiTinyFace-Tl_WTe5j63PWCbOm) | 0:27 | ace with a pistol, different gun sound |
| 5 | [CLUTCH](https://www.twitch.tv/washedlatino/clip/FurtiveCaringJackalHeyGirl-diiL25UNtCpo7AAL) | 0:23 | clutch, quiet and full of footsteps |
| 6 | [Insane clutch](https://www.twitch.tv/washedlatino/clip/SuccessfulAstutePeppermintTriHard-WCMrjwrtjctjY2Cq) | 0:32 | clutch |
| 7 | [CLUTCH UWU](https://www.twitch.tv/washedlatino/clip/ObliquePopularFriesWutFace-omfGVqnt23i0Cbnh) | 0:29 | clutch |
| 8 | [MASTER MARSHEL GETS A JUICY JUICY 3K](https://www.twitch.tv/washedlatino/clip/WiseWonderfulCobraKappaPride-PowIWDpLf_8-W05L) | 0:27 | 3k |
| 9 | [3k kinda lame](https://www.twitch.tv/washedlatino/clip/HonestVenomousCrabMikeHogu-ZeKCQ34L-ICbwfZf) | 0:27 | 3k |
| 10 | [NINA DEFUSE](https://www.twitch.tv/washedlatino/clip/TrappedScaryCarrotArsonNoSexy-0t720Gy3JvYqTS5Q) | 0:30 | defuse, footsteps around |
| 11 | [SHEEEESH MY AIM TOO CRISP](https://www.twitch.tv/washedlatino/clip/ArbitrarySpineyGoshawkMingLee-AAP83l527advyO8C) | 0:30 | gunfights |
| 12 | [1hp btw](https://www.twitch.tv/washedlatino/clip/TastyDeliciousParrotBleedPurple-8WJw5vsbj2wW5mOY) | 0:16 | close fight |
| 13 | [IM SQUEAKY CLEAN~](https://www.twitch.tv/washedlatino/clip/GoldenConfidentTriangleUWot-f76mViE15p2VOOmB) | 0:33 | pretty sure it's a clutch |
| 14 | [sova poppin off](https://www.twitch.tv/washedlatino/clip/ComfortableBigGoblinAMPTropPunch-73-rjQ-qv5pO3n69) | 0:08 | short fight |
| 15 | [they mad](https://www.twitch.tv/washedlatino/clip/ArbitraryIncredulousShrimpDendiFace-A0cD5RT1c0LxxOVo) | 0:57 | long one, mix of everything |
| 16 | [First CS Stream. ESEA PUG](https://www.twitch.tv/washedlatino/clip/RichClearNeanderthalFailFish) | 0:30, CS | CS, not Valorant, so it's my second-game test |


## Split and evaluation

Clips 01, 02, 05, and 10 form the held-out Valorant evaluation. The other eleven original Valorant clips form the initial development set. Later development experiments added reviewed portions of a separate skirmish recording, grouped by source recording during validation. See [`splits.json`](../../splits.json) for exact membership.

The Counter-Strike clip is retained as a historical unseen-game test and is excluded from the selected submission report. Test annotations never enter training or threshold selection. The [results guide](../../docs/RESULTS.md) identifies the final four-clip comparison and separates it from historical experiments.

`08 MASTER MARSHEL GETS A JUICY JUICY 3K.srt` and `13 IM SQUEAKY CLEAN~.srt` supply the speech subtitles used alongside hand-labeled sound captions in the demo replay.
