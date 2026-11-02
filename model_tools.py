"""Offline development-only model suggestions and aligned voice-removal augmentation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from labeler import Dataset, ROOT
from review import atomic_json, augmentation_status, training_pairs, workspace

os.environ.setdefault("TORCH_HOME", str(ROOT / ".local" / "models"))


def decode(path, sample_rate):
    import numpy as np
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "2",
        "-af", "aresample=async=1:first_pts=0", "-ar", str(sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"
    ])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()


def estimate_direction(stereo):
    import numpy as np
    energy = np.mean(stereo ** 2, axis=0)
    total = float(energy.sum())
    if total < 1e-9:
        return "unknown"
    balance = float((energy[0] - energy[1]) / total)
    if abs(balance) > .18:
        return "left" if balance > 0 else "right"
    side = float(np.mean((stereo[:, 0] - stereo[:, 1]) ** 2)) / total
    return "center" if side < .03 else "unknown"


def generate_suggestions(dataset, keys, labels=None):
    for key in keys:
        dataset.clip(key)  # Reject test IDs before importing or executing a model.
    import numpy as np
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from ai_edge_litert.interpreter import Interpreter
    model = ROOT / "test-audio/model/yamnet.tflite"
    interpreter = Interpreter(model_path=str(model), num_threads=2)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    score_info = next(item for item in outputs if int(item["shape"][-1]) == 521)
    size = int(np.prod(input_info["shape"]))
    if size != 15600 or input_info["dtype"] != np.float32:
        raise ValueError(f"Unexpected YAMNet input: {input_info}")
    # Fixed proposal thresholds, not accuracy claims or a selected live configuration.
    classes = {"footsteps": ([48], .06), "gunshot": ([421, 422, 423, 424], .08),
               "explosion": ([420], .12),
               "talking": (list(range(0, 11)), .4), "music": ([132], .25)}
    if labels is not None:
        if not labels or set(labels) - set(classes):
            raise ValueError("Unsupported suggestion labels")
        classes = {label: classes[label] for label in labels}
    fingerprint = hashlib.sha256(model.read_bytes()).hexdigest()
    for key in keys:
        clip = dataset.clip(key)  # Intentionally cannot open test-mode media.
        clip_classes = {label: value for label, value in classes.items()
                        if clip.get("suggestion_labels") is None or label in clip["suggestion_labels"]}
        target = dataset.root / ".local" / "suggestions" / f"{key}.json"
        if target.exists():
            print(f"Keep existing suggestions: {clip['path'].name}", flush=True)
            continue
        if dataset.annotations(key)["rows"]:
            print(f"Keep existing human labels: {clip['path'].name}", flush=True)
            continue
        audio = decode(clip["path"], 16000)
        duration = dataset.metadata(key)["duration"]
        candidates = {label: [] for label in clip_classes}
        hop = 7680
        for offset in range(0, len(audio), hop):
            frame = audio[offset:offset + size]
            if len(frame) < size:
                frame = np.pad(frame, ((0, size - len(frame)), (0, 0)))
            mono, side = frame.mean(axis=1), (frame[:, 0] - frame[:, 1]) * .5
            predictions = []
            for channel in ([mono] if set(clip_classes) <= {"talking", "music"} else [mono, side]):
                interpreter.set_tensor(input_info["index"], channel.reshape(input_info["shape"]).astype(np.float32))
                interpreter.invoke()
                predictions.append(interpreter.get_tensor(score_info["index"]).reshape(-1, 521).max(axis=0))
            start, end = offset / 16000, min((offset + size) / 16000, duration)
            if start >= end:
                continue
            for label, (indices, threshold) in clip_classes.items():
                # Speech/music context comes from the original mono mix only.
                score = float((predictions[0] if label in {"talking", "music"} else np.maximum(*predictions))[indices].max())
                if score >= threshold:
                    candidates[label].append({"start_s": round(start, 3), "end_s": round(end, 3),
                                              "label": label, "direction": "unknown" if label in {"talking", "music"} else estimate_direction(frame),
                                              "score": round(score, 6)})
        rows = []
        for label, events in candidates.items():
            merged = []
            for event in events:
                if merged and event["start_s"] <= merged[-1]["end_s"] and event["direction"] == merged[-1]["direction"]:
                    merged[-1]["end_s"] = event["end_s"]
                    merged[-1]["score"] = max(merged[-1]["score"], event["score"])
                else:
                    merged.append(event.copy())
            for event in merged:
                event["id"] = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()[:16]
                event["status"] = "unconfirmed"
                rows.append(event)
        rows.sort(key=lambda event: (event["start_s"], event["label"]))
        atomic_json(target, {"model": "YAMNet TFLite", "model_sha256": fingerprint,
                             "purpose": "human-review proposals only; not training labels", "rows": rows,
                             "reload_supported": False, "thresholds": {k: v[1] for k, v in clip_classes.items()}})
        print(f"{clip['path'].name}: {len(rows)} unconfirmed proposals", flush=True)


def remove_voice(audio, sample_rate=44100):
    """Return aligned stereo residual; this estimates vocals, not perfect speech removal."""
    import numpy as np
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = get_model("htdemucs").cpu().eval()
    if model.samplerate != sample_rate or model.audio_channels != 2:
        raise ValueError("Unexpected separator sample rate/channel count.")
    waveform = torch.from_numpy(audio.T.copy())
    reference = waveform.mean(0)
    mean, std = reference.mean(), reference.std().clamp_min(1e-8)
    with torch.inference_mode():
        sources = apply_model(model, ((waveform - mean) / std)[None], device="cpu",
                              shifts=0, split=True, overlap=.25, progress=True)[0]
        vocals = sources[model.sources.index("vocals")] * std + mean
        residual = (waveform - vocals).T.numpy()
    if residual.shape != audio.shape or not np.isfinite(residual).all():
        raise ValueError("Separator changed the sample count or produced invalid audio.")
    return residual


def augment(dataset, key):
    clip = dataset.clip(key)  # Reject test IDs before importing or executing a model.
    if not clip.get("voice_removal_required", True):
        print(f"No voice-removed copy needed for {clip['path'].name}", flush=True)
        return
    import numpy as np
    import soundfile as sf
    reviewed = workspace(dataset, key)
    if not reviewed["review"]["complete"]:
        raise ValueError("Finish reviewing the entire clip before creating training copies.")
    folder = dataset.root / ".local" / "augmentation" / key
    folder.mkdir(parents=True, exist_ok=True)
    stamp = {"size": clip["path"].stat().st_size, "mtime_ns": clip["path"].stat().st_mtime_ns}
    old = augmentation_status(dataset.root, key)
    output = folder / "voice_removed.wav"
    if output.exists() and old.get("source") == stamp and old.get("separator") == "htdemucs-vocals-residual-v1":
        metadata = old
    else:
        audio = decode(clip["path"], 44100)
        # Align to the player timeline, preserving time zero and adding only tail padding.
        samples = round(dataset.metadata(key)["duration"] * 44100)
        audio = np.pad(audio[:samples], ((0, max(0, samples - len(audio))), (0, 0)))
        cleaned = remove_voice(audio)
        temp = folder / "voice_removed.tmp.wav"
        sf.write(temp, cleaned, 44100, subtype="FLOAT")
        info = sf.info(temp)
        if info.frames != samples or info.channels != 2:
            raise ValueError("Generated audio lost timing or stereo channels.")
        temp.replace(output)
        metadata = {"audio": str(output.relative_to(dataset.root)), "source": stamp,
                    "samples": samples, "sample_rate": 44100, "channels": 2,
                    "separator": "htdemucs-vocals-residual-v1", "offset_samples": 0,
                    "note": "Estimated voice reduction; artifacts may alter game sounds. Originals remain in training."}
    latest = workspace(dataset, key)
    label_revision = latest["revision"] if latest["review"]["complete"] else reviewed["revision"]
    metadata.update({"state": "ready", "csv_revision": label_revision})
    atomic_json(folder / "status.json", metadata)
    print(f"Prepared aligned stereo pair for {clip['path'].name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["suggest", "augment", "training-manifest"])
    parser.add_argument("--id")
    parser.add_argument("--group", default="valorant", choices=["valorant", "xonotic_second_game"])
    parser.add_argument("--validation-id", action="append", default=[])
    parser.add_argument("--labels", nargs="+", choices=["talking", "music", "footsteps", "gunshot", "explosion"])
    args = parser.parse_args()
    dataset = Dataset(ROOT)
    if args.command == "suggest":
        keys = [args.id] if args.id else [key for key, clip in dataset.clips.items() if clip["group"] == args.group]
        generate_suggestions(dataset, keys, args.labels)
    elif args.command == "augment":
        if not args.id:
            parser.error("augment requires --id")
        augment(dataset, args.id)
    else:
        pairs = training_pairs(dataset, args.group, args.validation_id)
        target = ROOT / ".local" / "training-manifest.json"
        atomic_json(target, {"group": args.group, "validation_ids": args.validation_id, "examples": pairs})
        print(f"{len(pairs)} ready training examples written to {target}")


if __name__ == "__main__":
    main()
