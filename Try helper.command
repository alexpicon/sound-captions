#!/bin/zsh
# macOS launcher; OBS and streaming remain under your control.
cd "${0:A:h}" || exit 1
python_bin='.venv-model/bin/python'
if [[ ! -x "$python_bin" ]]; then
  print 'Set up the Python environment first. See README.md > Quick start.'
  read 'reply?Press Enter to close.'
  exit 1
fi
print 'Sound Captions'
print 'OBS Browser Source: http://127.0.0.1:8771/overlay (1920 x 1080)'
print '1: Follow an OBS Media Source with the model'
print '2: Analyze a local gameplay recording with the model'
print '3: Replay hand-labeled captions from an OBS Media Source'
read 'mode?Choose 1, 2, or 3: '
case "$mode" in
  1|3)
    "$python_bin" live_helper.py --list-obs-inputs
    if [[ $? == 0 ]]; then
      read 'source_name?Type the source name exactly as shown above: '
      if [[ -n "$source_name" ]]; then
        if [[ "$mode" == 3 ]]; then
          "$python_bin" -u live_helper.py --obs-media "$source_name" --replay-labels
        else
          "$python_bin" -u live_helper.py --obs-media "$source_name"
        fi
      fi
    fi
    ;;
  2)
    read 'clip_path?Enter the full path to your gameplay recording: '
    if [[ -n "$clip_path" ]]; then
      "$python_bin" -u live_helper.py --file "$clip_path"
    fi
    ;;
  *) print 'No source selected.' ;;
esac
print '\nHelper stopped.'
read 'reply?Press Enter to close.'
