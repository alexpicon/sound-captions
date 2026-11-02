"""Training-fold-only cut/paste augmentation with donor AND recipient provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np

from labeler import ROOT
from model_tools import decode
from review import atomic_json
from scoring import LABELS
from training_core import window_targets


def require_membership(donor_id, recipient_id, allowed_ids):
    if donor_id == recipient_id or donor_id not in allowed_ids or recipient_id not in allowed_ids:
        raise ValueError("Augmentation donor and recipient must be different clips inside this training fold")


def shifted_labels(rows, start, end, offset=0, direction=None):
    result = []
    for row in rows:
        if row["label"] not in LABELS:
            continue
        a, b = max(row["start_s"], start), min(row["end_s"], end)
        if b > a:
            result.append({"start_s": a - start + offset, "end_s": b - start + offset,
                           "label": row["label"], "direction": direction or row["direction"]})
    return result


def mix_audio(background, source, gain, pan, offset_samples, rate=16000):
    """Return a new stereo array; never mutate cached source/background recordings."""
    result = background.copy()
    foreground = source.mean(axis=1).copy()
    fade = min(round(rate * .005), len(foreground) // 2)
    if fade:
        envelope = np.linspace(0, 1, fade, dtype=np.float32)
        foreground[:fade] *= envelope
        foreground[-fade:] *= envelope[::-1]
    count = min(len(foreground), len(result) - offset_samples)
    if count <= 0:
        raise ValueError("No space for the pasted sound")
    channel = {"left": 0, "right": 1}[pan]
    result[offset_samples:offset_samples + count, channel] += gain * foreground[:count]
    peak = float(np.max(np.abs(result)))
    if peak > .98:
        result *= .98 / peak
    return result


class FoldAugmenter:
    def __init__(self, records, encoder, plan):
        self.records = {r["id"]: r for r in records}
        self.ids = sorted(self.records)
        self.encoder = encoder
        self.plan = plan
        self.audio = {}
        self.cache_dir = ROOT / ".local/training-runs/round-4/augmentation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def waveform(self, key, variant, allowed):
        if key not in allowed:
            raise ValueError("Attempted waveform access outside training fold")
        cache_key = (key, variant)
        if cache_key not in self.audio:
            record = self.records[key]
            self.audio[cache_key] = decode(record["cleaned"] if variant == "cleaned" else record["path"], self.encoder.rate)
        return self.audio[cache_key]

    def recipe(self, donor, recipient, event_index, gain, pan, allowed):
        require_membership(donor, recipient, allowed)  # Check BEFORE reading any cache or audio.
        record, target = self.records[donor], self.records[recipient]
        item = {"donor": donor, "recipient": recipient, "event_index": event_index,
                "gain": gain, "pan": pan, "seed": self.plan["seed"],
                "donor_revision": record["csv_revision"], "recipient_revision": target["csv_revision"],
                "encoder": self.encoder.identity, "version": 1}
        item["audio_stamps"] = {name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                                for name, path in [("donor_cleaned", record["cleaned"]),
                                                   ("recipient_original", target["path"]), ("recipient_cleaned", target["cleaned"])]}
        tag = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:24]
        cache = self.cache_dir / f"{tag}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                return data["features"], data["targets"], item
        rng = np.random.default_rng(int(tag[:16], 16))
        event = record["truth"][event_index]
        rate, size = self.encoder.rate, self.encoder.size
        source_audio = self.waveform(donor, "cleaned", allowed)
        source_start = round(event["start_s"] * rate)
        source_end = min(len(source_audio), round(event["end_s"] * rate))
        source = source_audio[source_start:source_end]
        variant = "original" if rng.random() < .5 else "cleaned"
        recipient_audio = self.waveform(recipient, variant, allowed)
        offset = round(.5 * rate)
        length = min(len(source) + 2 * offset, len(recipient_audio))
        if length < size or not len(source):
            raise ValueError("Invalid labeled snippet length")
        source = source[:max(1, length - offset)]
        destination_start = int(rng.integers(0, max(1, len(recipient_audio) - length + 1)))
        background = recipient_audio[destination_start:destination_start + length]
        mixed = mix_audio(background, source, gain, pan, offset, rate)
        actual_donor_end = event["start_s"] + len(source) / rate
        original_labels = shifted_labels(target["truth"], destination_start / rate, (destination_start + length) / rate)
        pasted_labels = shifted_labels(record["truth"], event["start_s"], actual_donor_end, .5, pan)
        starts = sorted(set([0, max(0, (length - size) // 2), max(0, length - size)]))
        frames = [mixed[start:start + size].mean(axis=1) for start in starts]
        frames += [background[start:start + size].mean(axis=1) for start in starts]
        features, _ = self.encoder.encode(frames)
        times = np.asarray(starts) / rate
        targets = np.concatenate([window_targets(original_labels + pasted_labels, times, times + size / rate),
                                  window_targets(original_labels, times, times + size / rate)])
        np.savez_compressed(cache, features=features, targets=targets)
        atomic_json(self.cache_dir / f"{tag}.json", {**item, "source_start_s": event["start_s"],
                    "source_end_s": actual_donor_end, "recipient_start_s": destination_start / rate,
                    "recipient_variant": variant, "windows": len(frames), "paired_background_controls": True})
        return features, targets, item

    def for_fold(self, allowed_ids):
        allowed = set(allowed_ids)
        if not allowed <= set(self.ids):
            raise ValueError("Unknown or non-development training IDs")
        features, targets, provenance = [], [], []
        for donor in sorted(allowed):
            choices = [key for key in self.ids if key != donor]
            for event_index, event in enumerate(self.records[donor]["truth"]):
                if event["label"] not in LABELS:
                    continue
                recipient = choices[event_index % len(choices)]
                if recipient not in allowed:
                    continue  # No cache/audio read for an excluded donor or recipient.
                for gain in self.plan["augmentation"]["gains"]:
                    for pan in self.plan["augmentation"]["pan"]:
                        x, y, item = self.recipe(donor, recipient, event_index, gain, pan, allowed)
                        features.append(x); targets.append(y); provenance.append(item)
        if not features:
            raise ValueError("No fold-local augmentation examples")
        return np.concatenate(features), np.concatenate(targets), provenance
