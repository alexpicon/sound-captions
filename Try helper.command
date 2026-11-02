#!/bin/zsh
# Local development/demo helper only; never starts OBS or a Twitch stream.
cd "${0:A:h}" || exit 1
print 'Sound Captions — live helper + overlay'
print 'OBS Browser Source: http://127.0.0.1:8771/overlay (1920 x 1080)'
print 'Purple captions: round 2 at 0.9, max 2, up to 3 seconds.'
print '1: Follow a clip playing in OBS (OBS must be open; WebSocket enabled)'
print '2: Quick terminal test using your development clip 03'
read 'mode?Choose 1 or 2: '
if [[ "$mode" == 1 ]]; then
  print '\nLocal OBS Media Sources:'
  .venv-model/bin/python live_helper.py --list-obs-inputs
  if [[ $? == 0 ]]; then
    read 'source_name?Type the source name exactly as shown above: '
    .venv-model/bin/python -u live_helper.py --obs-media "$source_name"
  fi
elif [[ "$mode" == 2 ]]; then
  .venv-model/bin/python -u live_helper.py --file 'test-audio/real-clips/03 SOBAAAA ACEEEEE.mp4'
else
  print 'No source selected.'
fi
print '\nHelper stopped. Press Enter to close.'
read
