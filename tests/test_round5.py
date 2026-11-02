"""Data leakage checks for the one-clip addition; no real media or model fitting."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_round5 import collect_records,fold_ids,NEW_PATH


class Round5SplitTests(unittest.TestCase):
    def test_old_new_comparison_excludes_same_heldout_recording(self):
        records=[{'id':'a','fold_group':'a'},{'id':'b','fold_group':'b'},
                 {'id':'new','fold_group':'recording17'}]
        old={'a','b'}
        self.assertEqual(fold_ids(records,old,'a',False),['b'])
        self.assertEqual(fold_ids(records,old,'a',True),['b','new'])
        self.assertEqual(fold_ids(records,old,'new',False),['a','b'])
        self.assertEqual(fold_ids(records,old,'new',True),['a','b'])
        records.append({'id':'sibling','fold_group':'recording17'})
        self.assertEqual(fold_ids(records,old,'new',True),['a','b'])

    def test_collector_never_opens_unlabeled_siblings_and_preserves_original_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['old.wav','clean.wav','new.mp4']:(root/name).write_bytes(b'fixture')
            calls=[]
            class FakeDataset:
                def __init__(self):
                    self.root=root
                    self.clips={'old':{'path':root/'old.wav','relative':'old.wav','voice_removal_required':True},
                        'new':{'path':root/'new.mp4','relative':NEW_PATH,'voice_removal_required':False,'fold_group':'recording17'},
                        'unlabeled':{'relative':'test-audio/real-clips/17 skirmish game audio - part 02.mp4'}}
                def clip(self,key):
                    calls.append(key);assert key!='unlabeled';return self.clips[key]
                def metadata(self,key):
                    assert key!='unlabeled';return {'duration':60}
            def reviewed(dataset,key):
                assert key!='unlabeled'
                return {'rows':[{'label':'gunshot'}]*(34 if key=='new' else 1),'revision':'new-rev' if key=='new' else 'old-rev','review':{'complete':True}}
            stat=(root/'old.wav').stat()
            status={'state':'ready','csv_revision':'old-rev','audio':'clean.wav','source':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}}
            with patch('run_round5.workspace',side_effect=reviewed),patch('run_round5.augmentation_status',return_value=status):
                records,new_id=collect_records(FakeDataset(),{'old':'old-rev'})
            self.assertEqual(new_id,'new');self.assertEqual(calls,['old','new'])
            self.assertEqual(records[0]['cleaned'],root/'clean.wav');self.assertIsNone(records[1]['cleaned'])
