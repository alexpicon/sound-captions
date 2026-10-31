"""Keep model proposals separate from human CSV labels; gate training on review."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from labeler import FIELDS, WRITE_LOCK, revision, validate_rows

JOBS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-removal")
PENDING = set()
PENDING_LOCK = threading.Lock()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def suggestions(dataset, key, mode):
    dataset.clip(key, mode)
    if mode == "test":
        return []  # Do not even read cached model output for test clips.
    path = dataset.root / ".local" / "suggestions" / f"{key}.json"
    return load_json(path, {"rows": []})["rows"]


def workspace(dataset, key, mode="development"):
    with WRITE_LOCK:
        value = dataset.annotations(key, mode)
        state_path = dataset.root / ".local" / "reviews" / f"{key}.json"
        state = load_json(state_path, {})
        current = state.get("csv_revision") == value["revision"]
        value.update({"suggestions": suggestions(dataset, key, mode),
                      "review": {"decisions": state.get("decisions", {}) if current else {},
                                 "complete": bool(current and state.get("complete"))},
                      "review_revision": revision(state_path), "mode": mode})
        return value


def save_workspace(dataset, body, mode="development"):
    key = body.get("id")
    dataset.clip(key, mode)
    state_path = dataset.root / ".local" / "reviews" / f"{key}.json"
    review = body["review"]
    if not isinstance(review, dict) or not isinstance(review.get("complete"), bool):
        raise ValueError("Invalid review state.")
    decisions = review.get("decisions", {})
    proposals = suggestions(dataset, key, mode)
    ids = {row["id"] for row in proposals}
    if not isinstance(decisions, dict) or any(k not in ids or v not in {"accepted", "rejected"} for k, v in decisions.items()):
        raise ValueError("Invalid suggestion decision.")
    if review["complete"] and set(decisions) != ids:
        raise ValueError("Confirm or delete every suggestion before finishing this clip.")
    # Confirmed CSV contains only human-entered or explicitly accepted rows.
    clean = validate_rows(body.get("rows"), dataset.metadata(key, mode)["duration"])
    with WRITE_LOCK:
        if body.get("review_revision") != revision(state_path):
            raise FileExistsError("Review changed in another window. Reload before saving.")
        result = dataset.save(key, clean, body.get("revision"), mode)
        atomic_json(state_path, {"csv_revision": result["revision"], "decisions": decisions,
                                 "complete": review["complete"], "mode": mode})
        result = workspace(dataset, key, mode)
    if mode == "development" and review["complete"]:
        queue_augmentation(dataset.root, key)
    return result


def augmentation_status(root, key):
    return load_json(Path(root) / ".local" / "augmentation" / key / "status.json", {"state": "not_prepared"})


def queue_augmentation(root, key):
    root = Path(root)
    identity = (str(root), key)
    with PENDING_LOCK:
        if identity in PENDING:
            return
        PENDING.add(identity)
    def run():
        folder = root / ".local" / "augmentation" / key
        try:
            atomic_json(folder / "status.json", {**augmentation_status(root, key), "state": "preparing"})
            python = root / ".venv-model" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.exists():
                raise RuntimeError("Install requirements-model.txt into .venv-model to prepare voice-removed audio.")
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "worker.log").open("w") as log:
                result = subprocess.run([str(python), str(root / "model_tools.py"), "augment", "--id", key],
                                        cwd=root, stdout=log, stderr=log, timeout=1800)
            if result.returncode:
                raise RuntimeError("Voice removal failed; see the local worker.log. Re-finish the clip to retry.")
        except Exception as error:
            atomic_json(folder / "status.json", {"state": "error", "error": str(error)})
        finally:
            with PENDING_LOCK:
                PENDING.discard(identity)
    JOBS.submit(run)


def training_pairs(dataset, group="valorant", validation_ids=()):
    """Augment training folds only; validation keeps the original mixed audio."""
    pairs = []
    for key, clip in dataset.clips.items():
        if clip["group"] != group or key in validation_ids:
            continue
        value = workspace(dataset, key)
        if not value["review"]["complete"]:
            continue
        status = augmentation_status(dataset.root, key)
        if status.get("state") != "ready" or status.get("csv_revision") != value["revision"]:
            continue
        stamp = {"size": clip["path"].stat().st_size, "mtime_ns": clip["path"].stat().st_mtime_ns}
        if status.get("source") != stamp:
            continue
        cleaned = dataset.root / status["audio"]
        if not cleaned.is_file():
            continue
        labels = [{field: row[field] for field in FIELDS} for row in value["rows"] if row["label"] not in {"talking", "music"}]
        for variant, audio in [("original", clip["path"]), ("voice_removed", cleaned)]:
            pairs.append({"clip_id": key, "fold_group": key, "variant": variant,
                          "audio": str(audio.relative_to(dataset.root)), "labels": labels,
                          "csv_revision": value["revision"], "game_group": group})
    return pairs
