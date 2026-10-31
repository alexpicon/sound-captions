import unittest
import importlib.util

from scoring import aggregate, match_events, score_clip


def event(start, label="footsteps", direction="left"):
    return {"start_s": start, "end_s": start + .2, "label": label, "direction": direction}


class ScoringTests(unittest.TestCase):
    def test_one_caption_cannot_catch_two_sounds(self):
        result = score_clip([event(1), event(1.1)], [event(1.2)], 60)
        self.assertEqual(result["caught"], 1)
        self.assertEqual(result["caught_pct"], 50)

    def test_matching_does_not_lose_valid_matches_to_greedy_order(self):
        # First truth can use either caption; the second can use only caption 0.
        truth = [event(1), event(2)]
        predictions = [event(1.2), event(.1)]
        self.assertEqual(len(match_events(truth, predictions)), 2)

    def test_tolerance_boundary_is_inclusive(self):
        self.assertEqual(len(match_events([event(0)], [event(1)])), 1)
        self.assertEqual(len(match_events([event(0)], [event(1.001)])), 0)

    def test_wrong_labels_and_duplicates_are_not_credited(self):
        truth = [event(2)]
        predictions = [event(2, "gunshot"), event(2.1), event(2.2), event(10)]
        result = score_clip(truth, predictions, 30)
        self.assertEqual(result["caught"], 1)
        self.assertEqual(result["false_captions"], 1)  # Only time 10 has no nearby real onset.
        self.assertEqual(result["unmatched_captions"], 3)
        self.assertEqual(result["false_per_min"], 2)
        self.assertEqual(result["unmatched_per_min"], 6)

    def test_confusion_table_counts_wrong_label_and_misses(self):
        result = score_clip([event(1), event(10, "reload")], [event(1.1, "gunshot")], 60)
        self.assertEqual(result["caught"], 0)
        self.assertEqual(result["confusion"]["footsteps"], {"gunshot": 1})
        self.assertEqual(result["confusion"]["reload"], {"missed": 1})

    def test_center_suppression_has_explicit_all_direction_comparison(self):
        truth = [event(1, direction="center"), event(5)]
        result = score_clip(truth, [event(5.5)], 60)
        self.assertEqual(result["targets"], 1)
        self.assertEqual(result["centered_targets_excluded"], 1)
        all_sounds = score_clip(truth, [event(5.5)], 60, hide_centered=False)
        self.assertEqual(all_sounds["targets"], 2)
        self.assertEqual(all_sounds["caught_pct"], 50)

    def test_arrow_accuracy_has_coverage_and_ignores_unknown_truth(self):
        truth = [event(1), event(4), event(7), event(10, direction="unknown")]
        predictions = [event(1, direction="left"), event(4, direction="right"),
                       event(7, direction="unknown"), event(10)]
        result = score_clip(truth, predictions, 60)
        self.assertEqual(result["arrow_correct_pct"], 50)
        self.assertAlmostEqual(result["arrow_coverage_pct"], 200 / 3)
        self.assertEqual(result["matched_directional_targets"], 3)

    def test_empty_denominators_are_unavailable_not_perfect(self):
        result = score_clip([], [], 60)
        self.assertIsNone(result["caught_pct"])
        self.assertIsNone(result["arrow_correct_pct"])
        self.assertIsNone(result["per_label"]["reload"]["caught_pct"])
        self.assertEqual(result["false_per_min"], 0)

    def test_aggregation_uses_counts_and_duration_not_mean_of_percentages(self):
        a = score_clip([event(1)], [event(1)], 60)
        b = score_clip([event(1), event(3), event(5)], [], 120)
        combined = aggregate([a, b])
        self.assertEqual(combined["caught_pct"], 25)
        self.assertEqual(combined["duration_s"], 180)
        self.assertEqual(combined["per_label"]["footsteps"]["targets"], 4)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Use .venv-model for caption-policy tests")
class CaptionPolicyTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from run_baseline import captions_from_windows
        self.np, self.caption = np, captions_from_windows
        self.config = {"class_mapping": {"footsteps": [0], "gunshot": [1], "explosion": [2], "reload": []},
                       "thresholds": {"footsteps": .1, "gunshot": .1, "explosion": .1},
                       "caption_ttl_s": 2, "max_captions": 2, "hide_centered": True}

    def window(self, time, scores, direction="left"):
        return {"time": time, "scores": self.np.asarray(scores), "direction": direction}

    def test_two_caption_limit_and_no_reload_invention(self):
        rows = self.caption([self.window(.975, [.7, .9, .8])], self.config)
        self.assertEqual([row["label"] for row in rows], ["gunshot", "explosion"])
        self.assertEqual(rows[0]["start_s"], .975)

    def test_center_is_hidden_and_unknown_can_have_no_arrow(self):
        self.assertEqual(self.caption([self.window(.975, [.9, 0, 0], "center")], self.config), [])
        rows = self.caption([self.window(.975, [.9, 0, 0], "unknown")], self.config)
        self.assertEqual(rows[0]["direction"], "unknown")

    def test_continuous_positive_run_is_not_repeated(self):
        rows = self.caption([self.window(.975, [.9, 0, 0]), self.window(1.455, [.9, 0, 0]),
                             self.window(3, [0, 0, 0]), self.window(3.48, [.9, 0, 0])], self.config)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
