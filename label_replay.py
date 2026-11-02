"""Hand-label playback for demos; never runs a classifier or scores a clip."""
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

SOUNDS = {'footsteps', 'gunshot', 'reload', 'explosion'}
DIRECTIONS = {'left', 'right', 'center', 'unknown'}


def load_speech(media):
    """Read an optional adjacent SRT, preserving its censored text verbatim."""
    path = Path(media).with_suffix('.srt')
    if path.is_symlink():
        raise ValueError('Replay subtitles must be the actual adjacent SRT, not a symlink')
    if not path.exists():
        return []
    timestamp = r'(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})'

    def seconds(parts):
        hours, minutes, seconds, millis = map(int, parts)
        if minutes >= 60 or seconds >= 60:
            raise ValueError('Invalid SRT timestamp')
        return hours * 3600 + minutes * 60 + seconds + millis / 1000

    rows = []
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n').strip()
    for number, block in enumerate(re.split(r'\n\s*\n', text) if text else [], 1):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines.pop(0)
        match = re.fullmatch(timestamp + r'\s*-->\s*' + timestamp, lines[0].strip()) if lines else None
        if not match:
            raise ValueError(f'Invalid SRT timing in cue {number}')
        start, end = seconds(match.groups()[:4]), seconds(match.groups()[4:])
        words = '\n'.join(lines[1:]).strip()
        if end <= start or not words or len(words) > 1000:
            raise ValueError(f'Invalid SRT cue {number}')
        rows.append(dict(cue_id=number, start_s=start, end_s=end, text=words))
    return sorted(rows, key=lambda row: row['start_s'])


class SpeechTimeline:
    def __init__(self, rows):
        self.rows = rows
        self.current = None
        self.speed = 1.0

    def update(self, cursor, playing, *, force=False, speed=1.0):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('Invalid playback speed')
        active = next((row for row in reversed(self.rows)
                       if playing and cursor is not None and row['start_s'] <= cursor < row['end_s']), None)
        key = active['cue_id'] if active else None
        changed = key != self.current or (active is not None and (force or speed != self.speed))
        self.current, self.speed = key, speed
        return changed, ({**active, 'remaining_s': (active['end_s'] - cursor) / speed} if active else None)


def load_labels(media):
    path = Path(media).with_suffix('.csv')
    if path.is_symlink():
        raise ValueError('Replay labels must be the actual adjacent CSV, not a symlink')
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'start_s', 'end_s', 'label', 'direction'} <= set(reader.fieldnames or []):
            raise ValueError('Replay CSV requires start_s, end_s, label and direction')
        for number, row in enumerate(reader, 2):
            label = row['label'].strip().lower()
            if label in {'talking', 'music'}:
                continue
            start, end = float(row['start_s']), float(row['end_s'])
            direction = row['direction'].strip().lower()
            if (label not in SOUNDS or direction not in DIRECTIONS
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start):
                raise ValueError(f'Invalid sound annotation at CSV row {number}')
            rows.append(dict(start_s=start, end_s=end, label=label,
                             direction=direction, annotation_row=number))
    return sorted(rows, key=lambda r: (r['start_s'], r['annotation_row']))


@dataclass
class ReplayUpdate:
    reason: str | None
    captions: list


class LabelTimeline:
    def __init__(self, rows, duration=3.0):
        self.rows = rows
        self.duration = duration
        self.cursor = self.clock = None

    def update(self, cursor, playing, clock):
        if not playing or cursor is None:
            reason = 'Label replay paused/stopped' if self.cursor is not None else None
            self.cursor = self.clock = None
            return ReplayUpdate(reason, [])
        if not math.isfinite(cursor) or cursor < 0:
            raise ValueError('Invalid OBS playhead')
        first = self.cursor is None
        seek = not first and (cursor < self.cursor - .05
                              or cursor - self.cursor > clock - self.clock + .35)
        reason = 'Label replay started/resumed' if first else 'Label replay seek/loop' if seek else None
        if first or seek:
            candidates = [r for r in self.rows if r['start_s'] <= cursor
                          < max(r['end_s'], r['start_s'] + self.duration)]
        else:
            candidates = [r for r in self.rows if self.cursor < r['start_s'] <= cursor]
        captions = []
        for row in candidates:
            remaining = min(self.duration, max(row['end_s'], row['start_s'] + self.duration) - cursor)
            if remaining > 0:
                captions.append({**row, 'remaining_s': remaining})
        self.cursor, self.clock = cursor, clock
        return ReplayUpdate(reason, captions)


class ReplaySettings:
    """Small settings adapter: no model checkpoint or encoder is loaded."""
    def __init__(self, duration=3.0):
        self.deployment = {'display_duration_s': duration, 'model_round': None, 'thresholds': {}}
        self.compute_ms = []

    def reset(self):
        pass
