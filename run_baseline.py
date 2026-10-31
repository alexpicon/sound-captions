"""Run the frozen, untrained before baseline on the five manually labeled real tests."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from labeler import Dataset, ROOT, read_annotations, validate_rows
from model_tools import decode
from review import atomic_json
from scoring import LABELS, aggregate, score_clip


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captions_from_windows(windows, config):
    """Causal fixed policy shared by the before/final comparison."""
    import numpy as np
    captions, active, previous = [], [], set()
    for frame in windows:
        now = frame["time"]
        active = [row for row in active if row["start_s"] + config["caption_ttl_s"] > now]
        direction = frame["direction"]
        candidates, positive = [], set()
        for label, indices in config["class_mapping"].items():
            if not indices:
                continue
            score = float(np.max(frame["scores"][indices]))
            if score < config["thresholds"][label] or (config["hide_centered"] and direction == "center"):
                continue
            key = (label, direction)
            positive.add(key)
            if key not in previous:
                candidates.append({"start_s": round(now, 6), "end_s": round(now + config["caption_ttl_s"], 6),
                                   "label": label, "direction": direction, "score": score})
        for candidate in sorted(candidates, key=lambda row: (-row["score"], row["label"])):
            if len(active) < config["max_captions"]:
                active.append(candidate)
                captions.append(candidate)
        previous = positive
    return captions


def infer(path, duration, interpreter, config):
    import numpy as np
    audio = decode(path, config["sample_rate"])
    source_audio_sha = hashlib.sha256(audio.tobytes()).hexdigest()
    input_info = interpreter.get_input_details()[0]
    output_info = next(item for item in interpreter.get_output_details() if int(item["shape"][-1]) == 521)
    size, rate = config["window_samples"], config["sample_rate"]
    if int(np.prod(input_info["shape"])) != size or input_info["dtype"] != np.float32:
        raise ValueError("Unexpected model input.")
    windows, scores = [], []
    for offset in range(0, len(audio), config["hop_samples"]):
        now = min((offset + size) / rate, duration)
        if offset / rate >= duration:
            break
        if windows and now <= windows[-1]["time"]:
            break  # Emit at most one zero-padded tail window at the clip endpoint.
        stereo = audio[offset:offset + size]
        mono = np.pad(stereo.mean(axis=1), (0, max(0, size - len(stereo))))
        interpreter.set_tensor(input_info["index"], mono.reshape(input_info["shape"]).astype(np.float32))
        interpreter.invoke()
        vector = interpreter.get_tensor(output_info["index"]).reshape(-1, 521).max(axis=0)
        energy = np.mean(stereo ** 2, axis=0)
        total = float(energy.sum())
        balance = float((energy[0] - energy[1]) / total) if total > 1e-9 else 0
        side = float(np.mean((stereo[:, 0] - stereo[:, 1]) ** 2)) / total if total > 1e-9 else 1
        direction = "unknown"
        if abs(balance) > config["direction_balance_threshold"]:
            direction = "left" if balance > 0 else "right"
        elif side < config["center_side_energy_threshold"]:
            direction = "center"
        windows.append({"time": now, "direction": direction, "scores": vector})
        scores.append(vector)
    return captions_from_windows(windows, config), windows, np.asarray(scores), source_audio_sha


def main():
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from importlib.metadata import version
    config_path = ROOT / "baseline-config.json"
    config = json.loads(config_path.read_text())
    output = ROOT / "evaluations" / "round-0"
    raw = ROOT / ".local" / "evaluations" / "round-0"
    if (output / "summary.json").exists() or (output / "protocol.json").exists():
        raise SystemExit("Round 0 already exists. Preserve it; do not rerun or overwrite the baseline.")
    dataset = Dataset(ROOT)
    # Freeze all settings and code fingerprints before reading any test answer labels.
    protocol = {"created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                "config": config, "config_sha256": sha(config_path),
                "model_sha256": sha(ROOT / config["model"]), "splits_sha256": sha(ROOT / "splits.json"),
                "runner_sha256": sha(Path(__file__)), "scorer_sha256": sha(ROOT / "scoring.py"),
                "git_commit_before_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "runtime": {name: version(name) for name in ["ai-edge-litert", "numpy"]},
                "no_training": True, "no_separation": True, "no_test_tuning": True}
    atomic_json(output / "protocol.json", protocol)
    interpreter = Interpreter(model_path=str(ROOT / config["model"]), num_threads=2)
    interpreter.allocate_tensors()
    records = []
    for key, clip in dataset.test_clips.items():
        started = time.perf_counter()
        duration = dataset.metadata(key, "test")["duration"]
        predictions, windows, scores, decoded_sha = infer(clip["path"], duration, interpreter, config)
        atomic_json(raw / f"{key}-predictions.json", {"clip_id": key, "predictions": predictions})
        np.savez_compressed(raw / f"{key}-scores.npz", scores=scores,
                            times=np.asarray([w["time"] for w in windows]),
                            directions=np.asarray([w["direction"] for w in windows]))
        # Ground truth is opened only after predictions have been fixed and saved.
        csv_path = clip["path"].with_suffix(".csv")
        if not csv_path.exists():
            raise ValueError(f"Missing test annotations: {csv_path.name}")
        truth = validate_rows(read_annotations(csv_path), duration)
        record = {"clip_id": key, "clip": clip["relative"], "group": clip["group"],
                  "annotation_sha256": sha(csv_path), "media_sha256": sha(clip["path"]),
                  "decoded_audio_sha256": decoded_sha, "inference_and_scoring_s": time.perf_counter() - started,
                  "primary": score_clip(truth, predictions, duration, config["matching_tolerance_s"], True),
                  "all_directions": score_clip(truth, predictions, duration, config["matching_tolerance_s"], False),
                  "context_annotation_spans": sum(r["label"] in {"talking", "music"} for r in truth),
                  "contexts": {name: None for name in ["quiet", "talking", "music"]}}
        atomic_json(raw / f"{key}-metrics.json", record)
        records.append(record)
        print(f"Scored {clip['path'].name}; settings unchanged.", flush=True)
    groups = {group: {"primary": aggregate([r["primary"] for r in records if r["group"] == group]),
                      "all_directions": aggregate([r["all_directions"] for r in records if r["group"] == group]),
                      "clips": sum(r["group"] == group for r in records)}
              for group in sorted({r["group"] for r in records})}
    summary = {"round": 0, "protocol_sha256": sha(output / "protocol.json"), "groups": groups,
               "overall": aggregate([r["primary"] for r in records]),
               "contexts": {name: None for name in ["quiet", "talking", "music"]},
               "context_note": "Manual talking/music annotation is incomplete; all context slices, including quiet, are N/A.",
               "clips": records, "final_lab_opened": False, "training_performed": False}
    atomic_json(output / "summary.json", summary)
    print(f"Saved immutable baseline to {output}", flush=True)


if __name__ == "__main__":
    main()
