import unittest
import importlib.util


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Use the model environment")
class CutPasteTests(unittest.TestCase):
    def test_donor_and_recipient_must_both_be_in_fold(self):
        from cutpaste import require_membership, FoldAugmenter
        require_membership("a", "b", {"a", "b"})
        for donor, recipient in [("a", "heldout"), ("heldout", "a"), ("a", "a")]:
            with self.assertRaises(ValueError):
                require_membership(donor, recipient, {"a", "b"})
        # Checks must happen before touching data or even looking up a cache path.
        fake = object.__new__(FoldAugmenter)
        with self.assertRaises(ValueError):
            fake.recipe("a", "heldout", 0, .5, "left", {"a", "b"})

    def test_mixing_is_panned_and_does_not_mutate_source_arrays(self):
        import numpy as np
        from cutpaste import mix_audio
        background = np.zeros((2000, 2), dtype=np.float32)
        source = np.ones((500, 2), dtype=np.float32) * .2
        result = mix_audio(background, source, .5, "right", 400)
        self.assertTrue(np.allclose(result[:, 0], 0))
        self.assertAlmostEqual(float(result[600, 1]), .1, places=6)
        self.assertTrue(np.all(background == 0))
        self.assertTrue(np.allclose(source, .2))

    def test_copy_carries_overlapping_labels_and_clips_timestamps(self):
        from cutpaste import shifted_labels
        rows = [{"start_s": 1, "end_s": 3, "label": "gunshot", "direction": "center"},
                {"start_s": 2, "end_s": 4, "label": "reload", "direction": "left"},
                {"start_s": 1, "end_s": 4, "label": "talking", "direction": "unknown"}]
        result = shifted_labels(rows, 1.5, 3.5, .5, "right")
        self.assertEqual(result, [
            {"start_s": .5, "end_s": 2, "label": "gunshot", "direction": "right"},
            {"start_s": 1, "end_s": 2.5, "label": "reload", "direction": "right"}])

    def test_infeasible_threshold_never_wins_on_recall(self):
        from run_round4 import strict_choice
        selected = strict_choice([
            {"threshold": .9, "caught_pct": 80, "false_per_min": 2.01, "unmatched_per_min": 3},
            {"threshold": .99, "caught_pct": 35, "false_per_min": 2, "unmatched_per_min": 2.5},
            {"threshold": 1.000001, "caught_pct": 0, "false_per_min": 0, "unmatched_per_min": 0}])
        self.assertEqual(selected["threshold"], .99)
        with self.assertRaises(ValueError):
            strict_choice([{"threshold": .9, "caught_pct": 80, "false_per_min": 3, "unmatched_per_min": 3}])


if __name__ == "__main__":
    unittest.main()
