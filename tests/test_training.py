import unittest
import importlib.util


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Use .venv-model with training requirements")
class TrainingTests(unittest.TestCase):
    def test_excluded_clip_ids_never_enter_training(self):
        from training_core import training_ids
        ids = ["a", "b", "c", "d"]
        self.assertEqual(training_ids(ids, ["b"]), ["a", "c", "d"])
        self.assertEqual(training_ids(ids, ["b", "c"]), ["a", "d"])
        with self.assertRaises(ValueError):
            training_ids(ids, ids)

    def test_shared_timing_gives_identical_targets_and_context_is_not_a_target(self):
        import numpy as np
        from training_core import window_targets
        truth = [{"start_s": .1, "end_s": .4, "label": "footsteps", "direction": "center"},
                 {"start_s": 0, "end_s": 3, "label": "talking", "direction": "unknown"}]
        starts, ends = np.asarray([0, 1]), np.asarray([.975, 1.975])
        a = window_targets(truth, starts, ends)
        b = window_targets(truth, starts.copy(), ends.copy())
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.tolist(), [[1, 0, 0, 0], [0, 0, 0, 0]])

    def test_absent_class_cannot_be_hallucinated_by_head(self):
        import numpy as np
        from training_core import ClassifierHead
        x = np.asarray([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
        y = np.asarray([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
        head = ClassifierHead().fit(x, y)
        result = head.predict(np.asarray([[100, -100]], dtype=float))
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result[0, 2], 0)
        self.assertEqual(result[0, 3], 0)
        self.assertTrue(np.array_equal(head.scaler.mean_, x.mean(axis=0)))

    def test_context_uses_union_duration_and_retains_matched_arrow_counts(self):
        from run_training import context_scores
        row = {"start_s": 1, "end_s": 1.2, "label": "gunshot", "direction": "left"}
        record = {"duration": 10, "truth": [row,
                  {"start_s": 0, "end_s": 3, "label": "talking", "direction": "unknown"},
                  {"start_s": 2, "end_s": 4, "label": "talking", "direction": "unknown"}]}
        score = context_scores(record, [dict(row)])
        self.assertEqual(score["talking"]["duration_s"], 4)
        self.assertEqual(score["quiet"]["duration_s"], 6)
        self.assertEqual(score["talking"]["caught"], 1)
        self.assertEqual(score["talking"]["arrows_correct"], 1)
        self.assertIsNone(score["music"])


if __name__ == "__main__":
    unittest.main()
