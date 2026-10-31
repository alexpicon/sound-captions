# Sound Captions: my hackathon project (please read all of this first)

hey! so I'm doing the Codex x TwitchCon hackathon and I want you to build my project with me. below is my idea, how I think it should work, the rules I have to follow, and where all my files are. please read the whole thing before you write any code, and if something doesn't make sense just ask me.

## the idea

I want to build Sound Captions. while I'm streaming, it puts the important game sounds on screen with an arrow showing which side they came from, like:

```
#14 [footsteps ← left]
#15 [gunshot → right]
```

it's for deaf and hard-of-hearing viewers, and also anyone watching with the sound off (on the bus, in class, at work, whatever). right now they can see the game but they have no idea someone is sneaking up on the left. streamers caption their voice sometimes, but nobody captions the actual game.

one thing I found while researching: most deaf viewers (57% in one study) only want the important sounds captioned, not every single noise. so it has to stay chill and readable. max 2 captions on screen at once, never spam the screen.

## how I think it should work

1. listen: a little helper app on my PC grabs only the game audio (not my mic)
2. recognize: use a pretrained sound model (YAMNet to start) plus a small add-on trained on clips I label, to sort the audio into `footsteps`, `gunshot`, `reload`, `explosion`, or `nothing`
3. direction: use mid/side. anything in the center cancels out of L-R, so whatever's left tells you left vs right. if it's not sure, no arrow
4. skip my own stuff: center sounds (like my own gun) don't get captioned
5. importance: a close gunshot matters more than far away footsteps. max 2 captions, they fade after a few seconds
6. show it: an OBS browser source overlay draws the captions on stream. I'll delay the game feed ~1 sec in OBS so they line up
7. chat stuff (Twitch EventSub + Helix):
   - `!sounds`: bot explains what the captions are
   - mods only: `!sounds off`, `!sounds less`, `!sounds more`
   - mods only: `!wrong 14 reload` / `!ok 14` to fix a caption. fixes can only use the fixed label list. saving fixes for training is local and off by default

## what I need from you (this is actually part of judging)

one of the judging criteria is literally "how well does it use Codex", so I want you doing the real work, not just autocomplete:

1. build the helper app, the overlay, the chat bot, and a tool for labeling clips
2. train the model in rounds: run the accuracy test, look at what it got wrong, change something, run it again. **log every round in `rounds.md`** (what you changed + score before and after). this log is going in my demo video so please keep it real and easy to read
3. add a new game: from ~10 labeled clips of a second game, make its sound profile, test it, and tell me the accuracy

## my files

everything's in `~/hackathons/sound-captions/`:

| path | what it is |
|---|---|
| `plan.md` | this file |
| `test-audio/README.md` | quick notes on the audio files |
| `test-audio/xonotic-sounds/` | 19 of Xonotic's own sound files (footsteps, guns, reload, explosions). GPLv3, license is in `COPYING` |
| `test-audio/clips/` | 10 labeled 2-min stereo clips (`.wav`) mixed from those sounds, each with a `.csv`: `start_s, end_s, label, direction` |
| `test-audio/clips/clip_voice_final.*` | the scoring clip. **never train or tune on it**, only use it for the final number |
| `test-audio/clips/` (the rest) | training clips. some have fake voice chat (macOS `say`) in the center |
| `test-audio/model/` | YAMNet (TFLite) + its class list |
| `test-audio/real-clips/` | 16 of my own Twitch clips as MP4s (stereo, not labeled yet) + links. real stream audio with my mic in it. mostly Valorant (my main game, it's what I stream) plus one CS clip |

there's no code yet, you're building all of it from an empty repo. fyi the `clips/` files are computer mixes of the same 19 sounds, so scores on them will look better than real gameplay. my real Twitch clips are the actual test. I'll also record a fresh Valorant stream later for step 8.

## rules I have to follow

- it has to connect to Twitch (chat bot + captions on the live stream) and be built with Codex. I need the Codex session ID (`/status`) at the end
- no Twitch Extension (review takes way too long). OBS browser source + chat bot instead
- the demo is my own Valorant stream on Twitch. the overlay is my project, not a Valorant mod or anything. no copyrighted music in the video (music off while recording) and no other streamers
- free Codex tier only
- demo video under 3 minutes, on YouTube, showing it working on Twitch and how Codex built it

## stuff I don't want

- no AI voices, nothing that talks or acts as me, no AI art
- no training on chat messages. only the fixed label list from mod fixes, and only if I turn it on
- no speech-to-text for my voice (streamers can use LocalVocal for that)
- please don't add features that aren't in here without asking me first

## accessibility (this one really matters to me)

- captions need high contrast, a big readable font, a solid background box, and no flashing
- arrows and words ("left"/"right") so it's not just colors or symbols
- stick to the importance limit so it stays readable
- before you build any overlay UI, **show me a mockup** (exactly how it looks, how big, where on screen) and wait for me to say yes

## Twitch setup notes

- docs: https://dev.twitch.tv/docs/chat/chatbot-guide/
- EventSub WebSocket: `wss://eventsub.wss.twitch.tv/ws`, subscription `channel.chat.message` v1
- sending messages: `POST https://api.twitch.tv/helix/chat/messages`
- scopes: `user:bot`, `user:read:chat`, `user:write:chat`
- their sample code doesn't handle reconnects or keepalives, so please handle both
- I'll register the Twitch app myself (it needs 2FA). read credentials from a `.env`, never commit it, and add a `.env.example`

## the model, splitting the data, and measuring accuracy

### which pretrained model
from what I can tell there's no ready-made model trained just on game sounds. the good ones are trained on AudioSet (~2 million YouTube sound clips), which already has stuff like "Gunshot, gunfire", "Walk, footsteps" and "Explosion". so my plan is to use one as a feature extractor and train a small add-on on my labeled clips.

| model | why try it | catch |
|---|---|---|
| YAMNet (Google) | start here. tiny, fast on CPU, easy to run live | least accurate of these (AudioSet mAP ~0.31) |
| EfficientAT MobileNet (`mn10_as` etc.) | small and fast but way more accurate than YAMNet. probably the best live pick | PyTorch, check the license |
| PANNs CNN14 | well known, accurate (mAP ~0.43), has a frame-by-frame version | bigger and slower, might be too heavy live |
| BEATs / AST | most accurate | way too heavy for live, only for comparison |

use YAMNet for round 1, then try EfficientAT (and maybe PANNs) in later rounds. pick whatever scores best and still keeps up in real time.

### how to split the data (please do this exactly)
- **split by whole clip, never by frame**. frames next to each other are almost the same, so mixing them between train and test gives a fake high score
- three groups:
  - train: what the add-on learns from
  - validation: for picking thresholds and settings. use leave-one-clip-out on the train clips
  - test: only for each round's final number. don't look at its mistakes to tune stuff
- my test sets:
  - lab: `test-audio/clips/clip_voice_final`
  - real: hold out 4 of my 16 Twitch clips. make sure they include one where I'm talking a lot, a quiet clutch, and a busy fight. the other 12 are train/validation
  - new game: the CS clip (`16 First CS Stream ESEA PUG`) stays out of training completely. it's the "does it work on a game it's never seen" test
- save the split in `splits.json` so it doesn't change between rounds

### how to label
- same CSV as the lab clips: `start_s, end_s, label, direction`
- also mark `talking` and `music` parts in the same file. they never get captioned, but they let us score quiet vs talking vs music separately

### how to measure accuracy (same script every round)
- caught: a real sound counts as caught if a caption with the same label starts within 1 sec of it (the model looks at ~1 sec windows)
- false caption: a caption with no real sound within 1 sec
- report every round:
  - caught % per label (footsteps, gunshot, reload, explosion)
  - false captions per minute
  - arrow correct %
  - all of that split into quiet / talking / music
  - a small confusion table (what it mixed up)
- before vs after: round 1 is the pretrained model by itself (no add-on, just mapping its AudioSet classes to my labels). every round after that gets compared against it on the same test clips, in a table in `rounds.md` like:

| round | what changed | caught % | false/min | arrow % | talking: caught % | music: caught % |
|---|---|---|---|---|---|---|

### talking and music
my old Twitch clips have my mic (I get loud lol) and sometimes music mixed into the same audio, so they're the hard test.
- live, this should mostly go away: the helper should only capture the game's audio (OBS application audio capture or the game's own output), not my mic or Spotify. in-game music off
- for the old mixed clips, try these as rounds:
  1. feed the side channel (L-R) to the model, not just the mono mix. my mic is mono so it's the same in left and right and cancels out, but off-center game sounds don't. my own gun is centered so it drops out too, which is fine. compare mono vs side vs both
  2. mix extra talking and music into the training clips so the add-on learns to ignore them
  3. direction from mid/side already ignores centered stuff like my voice. check it still works when music is wide stereo
  4. if you have time, try a vocal separator (like Demucs) before the model and tell me if it helps enough to be worth the delay
- report talking and music scores separately so we know if it only works when I'm quiet
- some old clips might have music in them, so don't use those parts in the video, and keep all clips out of the repo

## build order (stop and show me after each step)

1. repo + labeling tool. fresh repo. a simple tool where I play a clip and tag sounds (plus talking/music). saves the same CSV format as `test-audio/clips`. keep `.wav` and `.mp4` files out of git (they're huge)
2. accuracy test + round 1. make `splits.json`, then score the pretrained model by itself (the "before") on the test clips only. write round 1 in `rounds.md`
3. training rounds. make it better and log each round. goal: at least 70% of footsteps and gunshots caught, 2 or less false captions per minute, arrows right 90%+ of the time. checkpoint: if after a few rounds it's still catching under 50% on my 4 real test clips, **stop and tell me**. I'd rather rethink than keep pushing
4. live helper. captures game audio in real time, runs the model, sends captions to the overlay (a local WebSocket is fine). keep the delay low enough that a ~1 sec OBS delay covers it
5. overlay (only after I've said yes to the mockup)
6. chat bot with the commands above, mods-only where I said
7. second game from ~10 clips, and tell me the accuracy
8. real game check. once I record a fresh Valorant stream (music off, game audio captured on its own), label a few clips and tell me how the score changes. tell me if it drops, don't hide it

## what "done" means

- every step above works and I've seen it
- `rounds.md` shows accuracy going up over the rounds
- the README explains setup in a few steps
- there are tests for the scoring and the chat commands
- no secrets in the repo

## demo video plan (so you know what we're going for)

1. 0:00-0:20: a muted clip where you can't tell where the enemy is
2. 0:20-1:10: live on Twitch. footsteps on the left get caught while I'm talking. a viewer types `!sounds`. a mod fixes a wrong caption
3. 1:10-2:10: how Codex built it: labeling clips, the training rounds pushing accuracy up, then adding the second game
4. 2:10-2:50: who it helps, and how Twitch could make this a caption option for everyone

thanks!! start with step 1 and show me before moving on.
