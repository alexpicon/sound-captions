#!/usr/bin/env python3
"""Local stereo clip labeler. Python standard library + ffprobe for MP4 metadata."""

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import wave

ROOT = Path(__file__).resolve().parent
FIELDS = ["start_s", "end_s", "label", "direction"]
SOUNDS = {"footsteps", "gunshot", "reload", "explosion"}
CONTEXT = {"talking", "music"}
DIRECTIONS = {"left", "right", "center", "unknown"}
WRITE_LOCK = threading.RLock()


def validate_rows(rows, duration):
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ValueError("Expected a list with at most 10,000 annotations.")
    clean = []
    for index, row in enumerate(rows, 1):
        try:
            start, end = float(row["start_s"]), float(row["end_s"])
            label, direction = row["label"], row["direction"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Row {index}: missing or invalid fields.") from None
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration + 0.001:
            raise ValueError(f"Row {index}: require 0 ≤ start < end ≤ {duration:.3f}s.")
        if label not in SOUNDS | CONTEXT:
            raise ValueError(f"Row {index}: unknown label.")
        if direction not in DIRECTIONS:
            raise ValueError(f"Row {index}: unknown direction.")
        if label in CONTEXT:
            direction = "unknown"  # Context is an interval, not a caption target.
        start, end = round(start, 3), round(min(end, duration), 3)
        if start >= end:
            raise ValueError(f"Row {index}: interval must be at least 0.001s.")
        clean.append(dict(zip(FIELDS, (start, end, label, direction))))
    return sorted(clean, key=lambda r: (r["start_s"], r["end_s"], r["label"]))


def revision(path):
    return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()


def read_annotations(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV header must be start_s,end_s,label,direction.")
        return list(reader)


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / "splits.json").read_text())
        sealed = {p for group in manifest["sealed_tests"].values() for p in group}
        self.clips = {}
        self.test_clips = {}
        self.metadata_cache = {}
        for group, paths in manifest["development"].items():
            for relative in paths:
                if relative in sealed:
                    raise ValueError(f"Clip appears in both development and test: {relative}")
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root) or path.suffix not in {".mp4", ".wav"}:
                    raise ValueError("Invalid development media path.")
                clip_id = hashlib.sha256(relative.encode()).hexdigest()[:16]
                options = manifest.get("clip_options", {}).get(relative, {})
                self.clips[clip_id] = {"path": path, "relative": relative, "group": group,
                                      "voice_removal_required": options.get("voice_removal_required", True),
                                      "suggestion_labels": options.get("suggestion_labels"),
                                      "fold_group": options.get("fold_group", clip_id)}
        for group, paths in manifest["sealed_tests"].items():
            if group == "lab_final":
                continue  # This answer key already exists; keep the final lab clip closed.
            if "active_test_groups" in manifest and group not in manifest["active_test_groups"]:
                continue  # Retired tests remain sealed but are excluded from active evaluation.
            for relative in paths:
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root) or path.suffix not in {".mp4", ".wav"}:
                    raise ValueError("Invalid test media path.")
                key = hashlib.sha256(relative.encode()).hexdigest()[:16]
                self.test_clips[key] = {"path": path, "relative": relative, "group": group}

    def collection(self, mode="development"):
        if mode not in {"development", "test"}:
            raise ValueError("Unknown labeling mode.")
        return self.test_clips if mode == "test" else self.clips

    def clip(self, clip_id, mode="development"):
        clips = self.collection(mode)
        if clip_id not in clips:
            raise KeyError("Clip is unavailable or sealed.")
        clip = clips[clip_id]
        if not clip["path"].is_file():
            raise FileNotFoundError("Media file is missing from this checkout.")
        return clip

    def metadata(self, clip_id, mode="development"):
        clip = self.clip(clip_id, mode)
        path = clip["path"]
        stamp = path.stat().st_mtime_ns
        cached = self.metadata_cache.get(clip_id)
        if cached and cached[0] == stamp:
            return cached[1]
        if path.suffix == ".wav":
            with wave.open(str(path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
                channels, sample_rate = audio.getnchannels(), audio.getframerate()
        else:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                 "format=duration:stream=channels,sample_rate", "-of", "json", str(path)],
                capture_output=True, text=True, check=True, timeout=15,
            )
            info = json.loads(probe.stdout)
            duration = float(info["format"]["duration"])
            channels = int(info["streams"][0]["channels"])
            sample_rate = int(info["streams"][0]["sample_rate"])
        data = {"duration": duration, "channels": channels, "sample_rate": sample_rate}
        self.metadata_cache[clip_id] = (stamp, data)
        return data

    def annotations(self, clip_id, mode="development"):
        path = self.clip(clip_id, mode)["path"].with_suffix(".csv")
        with WRITE_LOCK:
            return {"rows": validate_rows(read_annotations(path), self.metadata(clip_id, mode)["duration"]),
                    "revision": revision(path), "csv_path": str(path.relative_to(self.root))}

    def save(self, clip_id, rows, expected_revision, mode="development"):
        path = self.clip(clip_id, mode)["path"].with_suffix(".csv")
        clean = validate_rows(rows, self.metadata(clip_id, mode)["duration"])
        with WRITE_LOCK:
            if expected_revision != revision(path):
                raise FileExistsError("CSV changed in another window. Reload this clip before saving.")
            if path.exists():
                backup = self.root / ".local" / "backups" / clip_id
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup / f"{time.time_ns()}.csv")
            # Replace only after a complete write; never leave a half-written CSV.
            fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(clean)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return {"rows": clean, "revision": revision(path), "csv_path": str(path.relative_to(self.root))}


def byte_range(header, size):
    if header is None:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length <= 0:
            raise ValueError("Invalid suffix range")
        return max(0, size - length), size - 1
    start, end = int(first), min(int(last) if last else size - 1, size - 1)
    if start > end or start >= size:
        raise ValueError("Unsatisfiable range")
    return start, end


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        super().end_headers()

    def json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("Host", "") not in {
            f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"
        }:
            self.json(403, {"error": "Use the local app address."})
            return
        url = urlsplit(self.path)
        params = parse_qs(url.query)
        clip_id = params.get("id", [""])[0]
        mode = params.get("mode", ["development"])[0]
        try:
            if url.path == "/api/clips":
                clips = []
                for key, clip in self.server.dataset.collection(mode).items():
                    clips.append({"id": key, "name": clip["path"].stem, "group": clip["group"],
                                  "available": clip["path"].is_file()})
                self.json(200, {"clips": clips, "sealed_count": 6})
            elif url.path == "/api/annotations":
                from review import workspace
                self.json(200, workspace(self.server.dataset, clip_id, mode))
            elif url.path == "/api/metadata":
                self.json(200, self.server.dataset.metadata(clip_id, mode))
            elif url.path == "/media":
                self.send_file(self.server.dataset.clip(clip_id, mode)["path"], ranges=True)
            elif url.path == "/api/augmentation":
                from review import augmentation_status
                clip = self.server.dataset.clip(clip_id)  # Development only.
                self.json(200, augmentation_status(self.server.dataset.root, clip_id)
                          if clip["voice_removal_required"] else {"state": "not_required"})
            elif url.path in {"/", "/app.js", "/styles.css"}:
                name = "index.html" if url.path == "/" else url.path[1:]
                self.send_file(self.server.dataset.root / "labeler-ui" / name)
            else:
                self.json(404, {"error": "Not found."})
        except (KeyError, FileNotFoundError):
            self.json(404, {"error": "Clip is unavailable or sealed."})
        except (ValueError, subprocess.SubprocessError, OSError) as error:
            self.json(400, {"error": str(error)})

    def do_POST(self):
        # A local web page must not be able to write annotations cross-origin.
        host = self.headers.get("Host", "")
        expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in expected or (origin is not None and origin != f"http://{host}"):
            self.json(403, {"error": "Only this local app may save annotations."})
            return
        if urlsplit(self.path).path != "/api/annotations":
            self.json(404, {"error": "Not found."})
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json(415, {"error": "Expected application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            mode = body.get("mode", "development")
            if "review" in body:
                from review import save_workspace
                value = save_workspace(self.server.dataset, body, mode)
            else:
                value = self.server.dataset.save(body.get("id"), body.get("rows"), body.get("revision"), mode)
            self.json(200, value)
        except FileExistsError as error:
            self.json(409, {"error": str(error)})
        except (KeyError, FileNotFoundError):
            self.json(404, {"error": "Clip is unavailable or sealed."})
        except (ValueError, TypeError, OSError) as error:
            self.json(400, {"error": str(error)})

    def send_file(self, path, ranges=False):
        size = path.stat().st_size
        header = self.headers.get("Range") if ranges else None
        try:
            start, end = byte_range(header, size)
        except ValueError:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        if ranges:
            self.send_header("Accept-Ranges", "bytes")
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    data = handle.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Normal when the player seeks or switches clips.


def make_server(root=ROOT, port=8765):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.dataset = Dataset(root)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not shutil.which("ffprobe"):
        parser.error("ffprobe is needed for MP4 metadata. Install FFmpeg first (Mac: brew install ffmpeg).")
    server = make_server(port=args.port)
    print(f"Sound Captions labeler: http://127.0.0.1:{server.server_port}", flush=True)
    print("Local only. Test-labeling mode never runs model guesses. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    # Keep one shared lock/module when review helpers import this running script.
    import sys
    sys.modules["labeler"] = sys.modules[__name__]
    main()
