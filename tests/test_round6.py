"""Guard the shared-recording split and explicit finished gate; no real media."""
import unittest
from pathlib import Path
from unittest.mock import patch
from run_round6 import allowed_ids,collect_round6,PARTS


class Round6Tests(unittest.TestCase):
    def test_skirmish_group_is_held_out_in_full_for_both_versions(self):
        records=[{'id':k,'fold_group':k} for k in ['a','b']]+[{'id':str(i),'fold_group':'skirmish'} for i in range(1,5)]
        old={'a','b','1'}
        for held in ['1','2','3','4']:
            self.assertEqual(allowed_ids(records,old,held,False),['a','b'])
            self.assertEqual(allowed_ids(records,old,held,True),['a','b'])
        self.assertEqual(allowed_ids(records,old,'a',False),['1','b'])
        self.assertEqual(allowed_ids(records,old,'a',True),['1','2','3','4','b'])

    def test_unfinished_new_part_stops_before_other_parts_or_training(self):
        base=[{'id':'old','csv_revision':'old-rev'},{'id':'1','csv_revision':'one-rev'}]
        calls=[]
        class Dataset:
            clips={str(i):{'relative':relative} for i,relative in enumerate(PARTS,2)}
            clips['5']={'relative':'test-audio/real-clips/17 skirmish game audio - part 05.mp4'}
            def clip(self,key):
                calls.append(key);return {'path':Path('part02.mp4')}
        with patch('run_round6.collect_records',return_value=(base,'1')),patch('run_round6.workspace',return_value={'review':{'complete':False}}):
            with self.assertRaisesRegex(ValueError,'Not marked finished'):
                collect_round6(Dataset(),{}, {'old':'old-rev','1':'one-rev'})
        self.assertEqual(calls,['2'])
