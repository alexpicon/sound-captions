"""Exercise CSV safety, held-out isolation, and actual HTTP media seeking/saving."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
import wave

from labeler import Dataset, byte_range, make_server, validate_rows


class LabelerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root / "train.wav"
        with wave.open(str(self.audio), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0" * 64000)
        (self.root / "sealed.wav").write_bytes(self.audio.read_bytes())
        self.manifest = {"development": {"valorant": ["train.wav"]},
                         "sealed_tests": {"test": ["sealed.wav"]}}
        (self.root / "splits.json").write_text(json.dumps(self.manifest))
        self.dataset = Dataset(self.root)
        self.clip_id = next(iter(self.dataset.clips))
        self.row = {"start_s": .1, "end_s": .6, "label": "footsteps", "direction": "left"}

    def tearDown(self):
        self.temp.cleanup()

    def test_csv_roundtrip_context_overlap_and_backup(self):
        initial = self.dataset.annotations(self.clip_id)
        rows = [self.row, {"start_s": 0, "end_s": 1, "label": "talking", "direction": "center"}]
        saved = self.dataset.save(self.clip_id, rows, initial["revision"])
        self.assertEqual(saved["rows"][0]["direction"], "unknown")
        self.assertEqual(self.dataset.annotations(self.clip_id), saved)
        before = self.audio.with_suffix(".csv").read_bytes()
        self.dataset.save(self.clip_id, [], saved["revision"])
        backups = list((self.root / ".local" / "backups").rglob("*.csv"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)

    def test_rejects_stale_save_without_overwriting(self):
        initial = self.dataset.annotations(self.clip_id)
        saved = self.dataset.save(self.clip_id, [self.row], initial["revision"])
        with self.assertRaises(FileExistsError):
            self.dataset.save(self.clip_id, [], initial["revision"])
        self.assertEqual(self.dataset.annotations(self.clip_id), saved)

    def test_invalid_annotations_never_write(self):
        for change in [{"end_s": 1.1}, {"start_s": -.1}, {"end_s": .1},
                       {"start_s": float("nan")}, {"end_s": float("inf")},
                       {"label": "arbitrary chat text"}, {"direction": "north"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.dataset.save(self.clip_id, [{**self.row, **change}], self.dataset.annotations(self.clip_id)["revision"])
        self.assertFalse(self.audio.with_suffix(".csv").exists())

    def test_existing_four_column_csv_is_readable(self):
        self.audio.with_suffix(".csv").write_bytes(b"start_s,end_s,label,direction\r\n0.1,0.6,gunshot,center\r\n")
        self.assertEqual(self.dataset.annotations(self.clip_id)["rows"][0]["label"], "gunshot")

    def test_sealed_clip_and_path_traversal_are_inaccessible(self):
        for key in ["sealed.wav", "../sealed.wav", "../../plan.md"]:
            with self.subTest(key=key), self.assertRaises(KeyError):
                self.dataset.clip(key)
        self.manifest["development"]["valorant"].append("sealed.wav")
        (self.root / "splits.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            Dataset(self.root)

    def test_ranges_for_browser_seeking(self):
        self.assertEqual(byte_range("bytes=10-20", 100), (10, 20))
        self.assertEqual(byte_range("bytes=90-", 100), (90, 99))
        self.assertEqual(byte_range("bytes=-10", 100), (90, 99))
        self.assertEqual(byte_range("bytes=0-500", 100), (0, 99))
        for header in ["bytes=101-", "bytes=20-10", "bytes=-0", "bytes=0-1,3-4"]:
            with self.subTest(header=header), self.assertRaises(ValueError):
                byte_range(header, 100)

    def test_http_save_reload_seek_and_isolation(self):
        server = make_server(self.root, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)

        def request(method, path, body=None, headers=None):
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()

        try:
            status, _, data = request("GET", "/api/clips")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(data)["clips"]), 1)
            status, headers, data = request("GET", f"/media?id={self.clip_id}", headers={"Range": "bytes=10-29"})
            self.assertEqual(status, 206)
            self.assertEqual(data, self.audio.read_bytes()[10:30])
            self.assertEqual(headers["Content-Range"], "bytes 10-29/64044")
            self.assertEqual(request("GET", "/media?id=sealed.wav")[0], 404)
            self.assertEqual(request("GET", "/../splits.json")[0], 404)
            self.assertEqual(request("GET", "/api/clips", headers={"Host": "attacker.invalid"})[0], 403)
            initial = self.dataset.annotations(self.clip_id)
            payload = json.dumps({"id": self.clip_id, "rows": [self.row], "revision": initial["revision"]})
            status, _, data = request("POST", "/api/annotations", payload,
                                      {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{server.server_port}"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["rows"], [self.row])
            self.assertEqual(request("POST", "/api/annotations", payload,
                                     {"Content-Type": "application/json"})[0], 409)
            self.assertEqual(request("POST", "/api/annotations", payload,
                                     {"Content-Type": "application/json", "Origin": "http://attacker.invalid"})[0], 403)
            status, _, data = request("GET", f"/api/annotations?id={self.clip_id}")
            self.assertEqual(json.loads(data)["rows"], [self.row])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
