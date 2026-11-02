"""Read-only sweep of stored round-6 OOF scores. No inference or fitting."""
from datetime import datetime
import json
from types import SimpleNamespace
import numpy as np
from caption_policy import apply_policy,load_policy
from labeler import Dataset,ROOT
from policy_scoring import score_policy_round
from review import atomic_json
from run_baseline import captions_from_windows
from run_round5 import label_details,sha
from run_round6 import collect_round6
from training_core import feature_record
from scoring import LABELS

GRID=[.9,.8,.7,.6,.5,.4,.3,.2,.1,.05]

def captions_at(data,probabilities,foot_threshold,base):
    config={**base,'class_mapping':{label:[i] for i,label in enumerate(LABELS)},
            'thresholds':{label:foot_threshold if label=='footsteps' else .9 for label in LABELS}}
    return apply_policy(captions_from_windows([{'time':float(t),'direction':str(d),'scores':p}
                         for t,d,p in zip(data['times'],data['directions'],probabilities)],config))


def main():
    out=ROOT/'training/round-6-threshold-review'
    if (out/'metrics.json').exists():raise SystemExit('Threshold review already saved; preserve it')
    historical=json.loads((ROOT/'training/round-2/protocol.json').read_text())
    previous=json.loads((ROOT/'training/round-5/protocol.json').read_text())
    records,part01,added=collect_round6(Dataset(ROOT),historical['development_labels'],previous['annotations'])
    protocol=json.loads((ROOT/'training/round-6/protocol.json').read_text())
    assert {r['id']:r['csv_revision'] for r in records}==protocol['annotations']
    base=json.loads((ROOT/'baseline-config.json').read_text());policy=load_policy()
    encoder=SimpleNamespace(name='yamnet',identity=json.loads((ROOT/'training/round-2/metrics.json').read_text())['encoder_sha256'])
    features={r['id']:feature_record(r,'original',encoder) for r in records}
    with np.load(ROOT/'.local/training-runs/round-6/round6-outer-probabilities.npz',allow_pickle=False) as saved:
        probabilities={k:saved[k] for k in saved.files}
    scopes={'same_15_clips':records,'same_12_clips':[r for r in records if r['id'] in previous['annotations']]}
    rows=[]
    for threshold in GRID:
        captions={r['id']:captions_at(features[r['id']],probabilities[r['id']],threshold,base) for r in records}
        row={'footsteps_threshold':threshold,'gunshot_threshold':.9,'scores':{}}
        for name,chosen in scopes.items():
            scores=score_policy_round(chosen,captions,policy);scores['label_details']=label_details(chosen,captions,policy)
            row['scores'][name]=scores
        rows.append(row)
        score=row['scores']['same_15_clips'];m=score['primary']['per_label']['footsteps'];f=score['label_details']['footsteps']
        print(f"{threshold:.2f}: footsteps {m['caught']}/{m['targets']} = {m['caught_pct']:.1f}%; false footsteps/min {f['false_per_min']:.2f}",flush=True)
    assert rows[0]['scores']['same_15_clips']['primary']==json.loads((ROOT/'training/round-6/metrics.json').read_text())['scores']['same_15_clips']['round6']['primary']
    atomic_json(out/'protocol.json',{'created_at':datetime.now().astimezone().isoformat(),'grid':GRID,
        'no_training_or_inference':True,'test_data_opened':False,'predictions_sha256':sha(ROOT/'.local/training-runs/round-6/round6-outer-probabilities.npz'),
        'annotations':protocol['annotations'],'policy_sha256':sha(ROOT/'caption-policy-v2.json'),
        'raw_scheduler_ttl_s':base['caption_ttl_s'],'overlay_display_lifetime_s':3,
        'display_note':'Keep v2 recognition scheduler; overlay holds newest two for up to3s, evicting oldest if needed. Display lifetime does not change inference scores.',
        'selection_note':'Exploratory development calibration, not an unbiased new validation score. No chosen setting or saved checkpoint modified by this sweep.'})
    atomic_json(out/'metrics.json',{'rows':rows,'test_data_opened':False,'training_performed':False})

if __name__=='__main__':main()
