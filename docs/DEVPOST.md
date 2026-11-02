## Inspiration

I used to stream Valorant on Twitch, and I still watch it. I checked every streamer I follow, and none had captions, let alone captions for game sounds.

Even when a stream has speech captions, viewers only get part of the experience. They know what's being said, but an offscreen gunshot or footsteps approaching from the left can still be missing.

Deaf streamer Sasiah, Christina Haslage, described this problem in a [2019 interview](https://butwhytho.net/2019/07/interview-the-deaf-and-hard-of-hearing-stream-team-with-christina-haslage/): games without descriptions for sounds like faint footsteps behind her were difficult to play. That stuck with me.

I used to stream so people could hang out with me. If you can't hear, or you're watching muted, you're only getting half the stream. That's why I built Sound Captions: captions for the game, not just the chat.

## What it does

Sound Captions adds the game's sounds to the stream: what you'd hear, and which side it's coming from.

It displays short captions like `[gunshot → right]` and `[footsteps ← left]` through an OBS overlay. The streamer runs a local audio helper and adds an OBS browser source. No extension is needed, and viewers don't have to install anything.

It can sit alongside speech captions so viewers get both what the streamer says and what's happening in the game. The overlay shows at most two sound captions at once, using short lowercase labels, brackets, directional arrows, and purple boxes. Captions disappear after a few seconds.

The captions in our demo are replayed from my own hand labels to demonstrate the experience. The model results below come from a separate held-out evaluation.

## How we built it

I labeled sounds in my own gameplay clips and corrected the AI's guesses. Codex built the labeling tool, training pipeline, evaluation tools, local audio helper, and OBS overlay.

The labeler lets me mark each sound's type, start and end time, and left/right/center direction, then confirm or correct model suggestions.

The sound model uses YAMNet audio embeddings with a small classifier head trained on my labels. Direction is estimated separately from the stereo left/right levels. Codex trained it on my corrections across six training rounds. We kept round 2 based on development results and the balance between missed sounds and false captions.

Under the hood, audio goes into the sound model, the helper estimates the sound and its side, and a WebSocket sends the caption to OBS. A separate replay mode follows the OBS media playhead to display my hand-labeled captions in sync with recorded gameplay.

## Challenges we ran into

Labeling was harder than I expected. Sounds overlap, my voice competes with the game, and deciding whether a sound belongs on the left or in the center isn't always obvious.

Reducing false captions also came at a cost. One experiment caught only 7.5% of eligible development sounds. Later training rounds didn't automatically make the model better, so we returned to the earlier checkpoint.

The selected model still produces about 10.9 false captions per minute, compared with 2.6 for the baseline. Footsteps remain weak: it missed the single eligible non-centered footstep event in the test. Reload captions are disabled in model mode because they were too noisy.

## Accomplishments that we're proud of

We built the full path from correcting sound labels to seeing captions in OBS.

We then tested the selected model on four held-out Valorant clips it had never trained on, with my voice still in the audio. It went from catching 20% of eligible game sounds to 65%: from 4 out of 20 to 13 out of 20. During speech, it caught 62%; without speech, 71%.

These percentages measure event recall for non-centered sounds, rather than overall accuracy. Gunshot recall increased from 21% to 68%. Five of the six evaluated direction arrows were correct, although arrows were only emitted for six of the 13 matched directional events.

It's a small test set, and the false-caption rate needs work. But we now have a working prototype and a measured improvement over the pretrained baseline.

## What we learned

Speech captions leave out information that matters to understanding gameplay. A short sound label and an arrow can communicate something the streamer's words never mention.

We also learned that catching more sounds is only part of the problem. Too many false captions can become distracting, and a higher detection threshold can hide the sounds we want viewers to notice. Evaluating both missed sounds and false captions helped us choose a more useful checkpoint.

Keeping the labeled demo separate from the model evaluation let us show the intended viewing experience while being clear about the prototype's current performance.

## What's next for Sound Captions

First, better footsteps and fewer false captions through more labeled gameplay and feedback from deaf and hard-of-hearing viewers. Then, support for more games.

I also want to build a Twitch extension so each viewer can turn sound captions on or off. The bigger idea is that Twitch could ship this as a caption option: deaf viewers get the game, not just the chat.
