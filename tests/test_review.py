import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from labeler import Dataset, revision
from model_tools import augment, generate_suggestions
from review import atomic_json, save_workspace, training_pairs, workspace


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ["train.wav", "test.wav", "final.wav"]:
            with wave.open(str(self.root / name), "wb") as audio:
                audio.setnchannels(2); audio.setsampwidth(2); audio.setframerate(16000)
                audio.writeframes(b"\0" * 64000)
        atomic_json(self.root / "splits.json", {"development": {"valorant": ["train.wav"]},
                    "sealed_tests": {"valorant": ["test.wav"], "lab_final": ["final.wav"]}})
        self.dataset = Dataset(self.root)
        self.key = next(iter(self.dataset.clips))
        self.test_key = next(iter(self.dataset.test_clips))
        self.row = {"start_s": .1, "end_s": .6, "label": "gunshot", "direction": "left"}
        atomic_json(self.root / ".local/suggestions" / f"{self.key}.json",
                    {"rows": [{**self.row, "id": "guess-1", "score": .7, "status": "unconfirmed"}]})

    def tearDown(self):
        self.temp.cleanup()

    def body(self, mode="development", complete=False, decisions=None, rows=None):
        key = self.test_key if mode == "test" else self.key
        value = workspace(self.dataset, key, mode)
        return {"id": key, "mode": mode, "rows": rows or [], "revision": value["revision"],
                "review_revision": value["review_revision"],
                "review": {"complete": complete, "decisions": decisions or {}}}

    def test_guesses_are_not_csv_labels_and_finish_requires_review(self):
        value = workspace(self.dataset, self.key)
        self.assertEqual(value["rows"], [])
        self.assertEqual(len(value["suggestions"]), 1)
        self.assertEqual(training_pairs(self.dataset), [])
        with self.assertRaises(ValueError):
            save_workspace(self.dataset, self.body(complete=True))
        self.assertFalse((self.root / "train.csv").exists())

    @patch("review.queue_augmentation")
    def test_accept_and_reject_persist_and_queue_only_on_completion(self, queue):
        result = save_workspace(self.dataset, self.body(decisions={"guess-1": "accepted"}, rows=[self.row]))
        self.assertEqual(result["rows"], [self.row])
        self.assertFalse(result["review"]["complete"])
        queue.assert_not_called()
        result = save_workspace(self.dataset, self.body(complete=True, decisions={"guess-1": "rejected"}))
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["review"]["complete"])
        queue.assert_called_once_with(self.root.resolve(), self.key)

    @patch("review.queue_augmentation")
    def test_test_labeling_has_no_guesses_or_augmentation(self, queue):
        # Even an accidentally cached test suggestion must never be read or displayed.
        atomic_json(self.root / ".local/suggestions" / f"{self.test_key}.json", {"rows": [{"id": "leak"}]})
        value = workspace(self.dataset, self.test_key, "test")
        self.assertEqual(value["suggestions"], [])
        saved = save_workspace(self.dataset, self.body("test", complete=True, rows=[self.row]), "test")
        self.assertEqual(saved["rows"], [self.row])
        queue.assert_not_called()
        self.assertEqual(training_pairs(self.dataset), [])
        final_key = hashlib.sha256(b"final.wav").hexdigest()[:16]
        with self.assertRaises(KeyError):
            self.dataset.clip(final_key, "test")
        for operation in [lambda: generate_suggestions(self.dataset, [self.test_key]), lambda: augment(self.dataset, self.test_key)]:
            with self.assertRaises(KeyError): operation()

    def test_review_conflict_even_when_csv_is_unchanged(self):
        body = self.body(decisions={"guess-1": "rejected"})
        save_workspace(self.dataset, body)
        with self.assertRaises(FileExistsError):
            save_workspace(self.dataset, body)

    @patch("review.queue_augmentation")
    def test_paired_examples_share_fold_and_keep_validation_original(self, queue):
        save_workspace(self.dataset, self.body(complete=True, decisions={"guess-1": "accepted"}, rows=[self.row]))
        folder = self.root / ".local/augmentation" / self.key
        folder.mkdir(parents=True)
        (folder / "voice_removed.wav").write_bytes((self.root / "train.wav").read_bytes())
        source = self.root / "train.wav"
        atomic_json(folder / "status.json", {"state": "ready", "csv_revision": revision(self.root / "train.csv"),
            "source": {"size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns},
            "audio": str((folder / "voice_removed.wav").relative_to(self.root))})
        pairs = training_pairs(self.dataset)
        self.assertEqual([p["variant"] for p in pairs], ["original", "voice_removed"])
        self.assertEqual(pairs[0]["fold_group"], pairs[1]["fold_group"])
        self.assertEqual(pairs[0]["labels"], pairs[1]["labels"])
        self.assertEqual(training_pairs(self.dataset, validation_ids=[self.key]), [])
        # External label edits invalidate completion and the training examples.
        (self.root / "train.csv").write_text("start_s,end_s,label,direction\n0.2,0.6,reload,center\n")
        self.assertEqual(training_pairs(self.dataset), [])


if __name__ == "__main__":
    unittest.main()
