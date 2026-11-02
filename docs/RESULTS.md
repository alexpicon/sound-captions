# Which results to use

The submission's headline comparison is **20% → 65% recall** for eligible non-centered sound events on four held-out Valorant clips. It comes from [`evaluations/final-valorant-selected.json`](../evaluations/final-valorant-selected.json).

Both sides use the final v2 caption policy and original mixed audio. The selected model catches 13 of 20 eligible events, compared with 4 of 20 for plain YAMNet. False captions increase from 2.62 to 10.94 per minute. This is a small test set, not a general accuracy guarantee.

## Evidence map

| Artifact | Meaning |
| --- | --- |
| `evaluations/final-valorant-selected.json` | Selected four-clip Valorant comparison, including per-label and speech-context slices |
| `evaluations/final-policy-v2/` | Frozen predictions, protocol, and broader raw final report from which the selected comparison was derived |
| `evaluations/round-0/` | Original pretrained baseline under the earlier policy; use the selected report for a like-for-like headline comparison |
| `evaluations/final-voice-removed-policy-v2/` | Historical voice-removal experiment; excluded from the submission's headline result |
| `training/` | Development validation, calibration, split audits, and decisions across training rounds |
| `rounds.md` | Chronological experiment log; statements about unopened tests describe their state at that point in time |
| `overlay/checks.json` | Historical overlay verification snapshot; not a model accuracy report |

The hand-label replay in the video demonstrates the viewing experience. It is separate from model predictions. Direction accuracy is 5/6 among evaluated arrows, with arrows supplied for only six of 13 matched directional events. Footstep recall is 0/1 in the primary test; reload output is disabled.

Recorded experiment files retain their original timestamps, local paths, and hashes for traceability. They have not been rewritten to make older experiments resemble the final result.
