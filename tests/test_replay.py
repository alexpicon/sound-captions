import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from label_replay import LabelTimeline, SpeechTimeline, load_labels, load_speech


class ReplayTests(unittest.TestCase):
    def test_srt_censor_text_multiline_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'clip.mp4'
            self.assertEqual(load_speech(media), [])
            media.with_suffix('.srt').write_text('\ufeff1\r\n00:00:01,980 --> 00:00:03,200\r\nGood [ __ ].\r\nSecond line.\r\n')
            cue = load_speech(media)[0]
            self.assertEqual(cue['text'], 'Good [ __ ].\nSecond line.')
            self.assertEqual(cue['start_s'], 1.98)
            media.with_suffix('.srt').write_text('1\n00:00:05,000 --> 00:00:01,000\nReversed\n')
            with self.assertRaises(ValueError):
                load_speech(media)

    def test_speech_is_active_only_inside_cue_and_resyncs(self):
        t = SpeechTimeline([dict(cue_id=1, start_s=1., end_s=3., text='Hello')])
        self.assertEqual(t.update(.5, True), (False, None))
        changed, cue = t.update(1.2, True)
        self.assertTrue(changed)
        self.assertAlmostEqual(cue['remaining_s'], 1.8)
        self.assertEqual(t.update(1.3, True), (False, cue | {'remaining_s': 1.7}))
        self.assertEqual(t.update(1.3, False), (True, None))
        self.assertTrue(t.update(2., True)[0])
        self.assertTrue(t.update(1.5, True, force=True)[0])
        changed, cue = t.update(1.5, True, speed=.5)
        self.assertTrue(changed)
        self.assertEqual(cue['remaining_s'], 3.)
        self.assertEqual(t.update(3., True), (True, None))

    def test_csv_keeps_all_sound_labels_and_drops_context(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / 'clip.mp4'
            with media.with_suffix('.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['start_s', 'end_s', 'label', 'direction'])
                writer.writerows([(0, 1, 'talking', 'unknown'), (1, 2, 'music', 'unknown'),
                                 (3, 4, 'reload', 'center'), (2, 3, 'footsteps', 'left')])
            rows = load_labels(media)
            self.assertEqual([r['label'] for r in rows], ['footsteps', 'reload'])
            self.assertEqual(rows[-1]['direction'], 'center')
            media.with_suffix('.csv').write_text('start_s,end_s,label,direction\nnan,3,gunshot,right\n')
            with self.assertRaises(ValueError):
                load_labels(media)

    def test_start_pause_resume_seek_and_loop(self):
        rows = [dict(start_s=0., end_s=.5, label='gunshot', direction='center'),
                dict(start_s=4., end_s=5., label='reload', direction='left')]
        t = LabelTimeline(rows)
        self.assertEqual(t.update(0, True, 10).captions[0]['label'], 'gunshot')
        self.assertFalse(t.update(.04, True, 10.04).captions)
        self.assertIsNotNone(t.update(.1, False, 10.1).reason)
        resumed = t.update(1, True, 11)
        self.assertEqual(resumed.captions[0]['remaining_s'], 2)
        sought = t.update(4.1, True, 11.1)
        self.assertIn('seek', sought.reason)
        self.assertEqual([r['label'] for r in sought.captions], ['reload'])
        loop = t.update(0, True, 11.2)
        self.assertIn('loop', loop.reason)
        self.assertEqual(loop.captions[0]['label'], 'gunshot')
        self.assertFalse(t.update(12, True, 12).captions)

    def test_normal_crossing_emits_once_and_preserves_short_cues(self):
        t = LabelTimeline([dict(start_s=.15, end_s=.17, label='footsteps', direction='right')])
        self.assertFalse(t.update(0, True, 0).captions)
        event = t.update(.2, True, .2).captions[0]
        self.assertAlmostEqual(event['remaining_s'], 2.95)
        self.assertFalse(t.update(.25, True, .25).captions)


class ReplayModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_scene_selection_follows_program_and_ignores_hidden_sources(self):
        from live_helper import current_scene_media
        class FakeObs:
            scene = 'Squeaky'
            async def request(self, kind, data=None):
                if kind == 'GetCurrentProgramScene':
                    return {'currentProgramSceneName': self.scene}
                return {'sceneItems': [
                    {'sourceName': self.scene + ' clip', 'inputKind': 'ffmpeg_source', 'sceneItemEnabled': True},
                    {'sourceName': 'hidden clip', 'inputKind': 'ffmpeg_source', 'sceneItemEnabled': False},
                    {'sourceName': 'Sound captions', 'inputKind': 'browser_source', 'sceneItemEnabled': True}]}
        obs = FakeObs()
        self.assertEqual(await current_scene_media(obs), ('Squeaky', 'Squeaky clip'))
        obs.scene = 'Marshel'
        self.assertEqual(await current_scene_media(obs), ('Marshel', 'Marshel clip'))

    async def test_ambiguous_scene_does_not_pick_the_wrong_clip(self):
        from live_helper import current_scene_media
        class FakeObs:
            async def request(self, kind, data=None):
                if kind == 'GetCurrentProgramScene':
                    return {'currentProgramSceneName': 'Other'}
                return {'sceneItems': [dict(sourceName=name, inputKind='ffmpeg_source', sceneItemEnabled=True)
                                       for name in ['one', 'two']]}
        self.assertEqual(await current_scene_media(FakeObs()), ('Other', None))

    async def test_replay_does_not_load_model(self):
        from live_helper import run
        args = SimpleNamespace(list_obs_inputs=False, list_devices=False,
                               replay_labels=True, json=True, port=8770, overlay_port=8771)
        async def check(args, engine, hub, runner, origins):
            self.assertEqual(engine.deployment['thresholds'], {})
        with patch('live_helper.load_env'), patch('live_helper.LiveEngine', side_effect=AssertionError('model loaded')), \
             patch('live_helper.overlay_service'), patch('live_helper.serve_source', side_effect=check):
            await run(args)

    async def test_sealed_media_blocked_before_csv_open(self):
        from live_helper import replay_source, SealedMediaError
        class FakeObs:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def request(self, *args):
                return {'inputKind': 'ffmpeg_source', 'inputSettings': {'local_file': 'sealed.mp4'}}
        runner = SimpleNamespace(reset=lambda reason: None)
        with patch('live_helper.ObsClient', FakeObs), \
             patch('live_helper.guard_media', side_effect=SealedMediaError('sealed')), \
             patch('live_helper.load_labels', side_effect=AssertionError('CSV opened')):
            with self.assertRaises(SealedMediaError):
                await replay_source(runner, 'clip')
