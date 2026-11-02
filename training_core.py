"""Development-only data, fixed feature encoders and clip-isolated classifier heads."""
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from labeler import Dataset, ROOT
from model_tools import decode, estimate_direction
from review import augmentation_status, workspace
from scoring import LABELS


def development_clips(dataset):
    """The only corpus accessor used in training. Never request test-mode entries."""
    records = []
    for key, clip in dataset.clips.items():
        if clip["group"] != "valorant":
            continue
        value = workspace(dataset, key)
        status = augmentation_status(dataset.root, key)
        if not value["review"]["complete"] or status.get("csv_revision") != value["revision"] or status.get("state") != "ready":
            raise ValueError(f"Incomplete labels or augmentation: {clip['path'].name}")
        cleaned = dataset.root / status["audio"]
        if not cleaned.is_file():
            raise ValueError("Missing aligned copy")
        records.append({"id": key, "path": clip["path"], "cleaned": cleaned,
                        "truth": value["rows"], "csv_revision": value["revision"],
                        "duration": dataset.metadata(key)["duration"]})
    return records


def training_ids(all_ids, excluded):
    ids = sorted(set(all_ids) - set(excluded))
    if not ids or set(ids) & set(excluded):
        raise ValueError("Invalid whole-clip split")
    return ids


def window_targets(truth, starts, ends):
    result = np.zeros((len(starts), len(LABELS)), dtype=np.float64)
    for label_index, label in enumerate(LABELS):
        for row in truth:
            if row["label"] == label:
                overlap = np.minimum(ends, row["end_s"]) - np.maximum(starts, row["start_s"])
                result[:, label_index] = np.maximum(result[:, label_index], overlap >= .020)
    return result


class YamnetEncoder:
    name, rate, size, hop = "yamnet", 16000, 15600, 7680

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        path = ROOT / "test-audio/model/yamnet.tflite"
        self.identity = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model = Interpreter(model_path=str(path), num_threads=2, experimental_preserve_all_tensors=True,
                                 experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        self.model.allocate_tensors()
        self.input = self.model.get_input_details()[0]
        self.output = self.model.get_output_details()[0]
        self.embedding = next(r["index"] for r in self.model.get_tensor_details()
                              if r["name"].endswith("layer28/reduce_mean") and np.prod(r["shape"]) == 1024)

    def encode(self, frames):
        features, scores = [], []
        for frame in frames:
            self.model.set_tensor(self.input["index"], frame.astype(np.float32).reshape(self.input["shape"]))
            self.model.invoke()
            features.append(self.model.get_tensor(self.embedding).reshape(-1).copy())
            scores.append(self.model.get_tensor(self.output["index"]).reshape(-1).copy())
        return np.asarray(features), np.asarray(scores)


class EfficientATEncoder:
    name, rate, size, hop = "efficientat", 32000, 31200, 15360

    def __init__(self):
        import torch
        vendor = ROOT / ".local/vendor/EfficientAT"
        sys.path.insert(0, str(vendor))
        previous = Path.cwd()
        try:
            os.chdir(vendor)  # Upstream reads its AudioSet map and weights relative to cwd.
            from models.mn.model import get_model
            from models.preprocess import AugmentMelSTFT
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = get_model(width_mult=1.0, pretrained_name="mn10_as").cpu().eval()
            self.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320,
                                      freqm=0, timem=0).cpu().eval()
        finally:
            os.chdir(previous)
        torch.set_num_threads(4)
        self.torch = torch
        path = vendor / "resources/mn10_as_mAP_471.pt"
        self.identity = hashlib.sha256(path.read_bytes()).hexdigest()

    def encode(self, frames):
        features, scores = [], []
        with self.torch.inference_mode():
            for offset in range(0, len(frames), 16):
                batch = self.torch.from_numpy(np.asarray(frames[offset:offset + 16], dtype=np.float32))
                spec = self.mel(batch).unsqueeze(1)
                logits, embedding = self.model(spec)
                features.extend(embedding.numpy())
                scores.extend(self.torch.sigmoid(logits).numpy())
        return np.asarray(features), np.asarray(scores)


def feature_record(record, variant, encoder, use_side=False):
    path = record["cleaned"] if variant == "voice_removed" else record["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tag = hashlib.sha256(f"v1:{digest}:{encoder.identity}:{use_side}".encode()).hexdigest()[:20]
    cache = ROOT / ".local/training-features" / encoder.name / f"{tag}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
    else:
        audio = decode(path, encoder.rate)
        frames, side_frames, starts, ends, directions = [], [], [], [], []
        for offset in range(0, len(audio), encoder.hop):
            end = min((offset + encoder.size) / encoder.rate, record["duration"])
            start = offset / encoder.rate
            if start >= record["duration"] or (ends and end <= ends[-1]):
                break
            stereo = audio[offset:offset + encoder.size]
            directions.append(estimate_direction(stereo))
            pad = (0, max(0, encoder.size - len(stereo)))
            frames.append(np.pad(stereo.mean(axis=1), pad))
            side_frames.append(np.pad((stereo[:, 0] - stereo[:, 1]) * .5, pad))
            starts.append(start); ends.append(end)
        started = time.perf_counter()
        features, scores = encoder.encode(frames)
        if use_side:
            side_features, _ = encoder.encode(side_frames)
            features = np.concatenate([features, side_features], axis=1)
        elapsed = time.perf_counter() - started
        data = {"features": features, "pretrained_scores": scores, "starts": np.asarray(starts),
                "times": np.asarray(ends), "directions": np.asarray(directions),
                "encoder_seconds": np.asarray(elapsed), "audio_sha256": np.asarray(digest)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
    data["targets"] = window_targets(record["truth"], data["starts"], data["times"])
    return data


class ClassifierHead:
    def fit(self, x, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
        self.scaler = StandardScaler().fit(x)
        normalized = self.scaler.transform(x)
        self.models = []
        with threadpool_limits(limits=1):
            for column in range(y.shape[1]):
                values = y[:, column]
                if len(np.unique(values)) < 2:
                    self.models.append(float(values[0]))
                else:
                    model = LogisticRegression(C=.1, class_weight="balanced", max_iter=400,
                                               random_state=20261101, solver="liblinear")
                    model.fit(normalized, values)
                    self.models.append(model)
        return self

    def predict(self, x):
        normalized = self.scaler.transform(x)
        return np.column_stack([np.full(len(x), model) if isinstance(model, float)
                                else model.predict_proba(normalized)[:, 1] for model in self.models])
