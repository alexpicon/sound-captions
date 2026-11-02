import unittest
from caption_policy import apply_policy, load_policy, supported_by_span
from policy_scoring import score_clip_v2


def event(t,label='gunshot',side='left',end=None):
    return {'start_s':t,'end_s':end if end is not None else t+.2,'label':label,'direction':side,'score':.99}


class PolicyTests(unittest.TestCase):
    def test_false_support_is_same_label_full_span_and_two_seconds_after_start(self):
        truth=[event(5,end=15),event(20,'footsteps')]
        for t in [4,5,10,15]:self.assertTrue(supported_by_span(event(t),truth))
        self.assertTrue(supported_by_span(event(22,'footsteps'),truth))
        self.assertFalse(supported_by_span(event(22.001,'footsteps'),truth))
        self.assertFalse(supported_by_span(event(10,'footsteps'),truth))
        self.assertFalse(supported_by_span(event(3.999),truth))

    def test_repeat_chain_merges_until_two_seconds_of_quiet(self):
        rows=[event(0),event(1.8),event(3.6),event(5.6)]
        self.assertEqual([r['start_s'] for r in apply_policy(rows)],[0,5.6])
        self.assertEqual(len(apply_policy([event(0),event(1,'gunshot','right'),event(1,'footsteps')])),3)

    def test_reload_removed_without_mutating_predictions_or_labels(self):
        rows=[event(0,'reload'),event(1)];original=[dict(r) for r in rows]
        self.assertEqual(len(apply_policy(rows)),1);self.assertEqual(rows,original)
        result=score_clip_v2([event(0,'reload'),event(1)],apply_policy(rows),60)
        self.assertEqual(result['targets'],1);self.assertEqual(result['per_label']['reload']['targets'],0)

    def test_support_does_not_inflate_onset_recall(self):
        result=score_clip_v2([event(0,end=10)],[event(5)],60)
        self.assertEqual(result['false_captions'],0)
        self.assertEqual(result['caught'],0)
        self.assertEqual(result['unmatched_captions'],1)

    def test_wrong_label_remains_false_despite_nearby_other_sound(self):
        result=score_clip_v2([event(0)],[event(0,'footsteps')],60)
        self.assertEqual(result['false_captions'],1)


class FinalGateTests(unittest.TestCase):
    def test_final_entrypoint_requires_explicit_flag_before_any_test_access(self):
        from unittest.mock import patch
        import contextlib,io
        import run_final
        with patch('run_final.Dataset') as dataset,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):run_final.main([])
            dataset.assert_not_called()
