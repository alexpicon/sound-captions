"""Round 6: add only reviewed skirmish 02-04 to round 5, fixed .9 / policy v2."""
from datetime import datetime
import json
import joblib
import numpy as np
from caption_policy import apply_policy,load_policy
from labeler import Dataset,ROOT
from policy_scoring import score_policy_round
from review import atomic_json,workspace
from run_round5 import collect_records,label_details,sha
from run_training import learned_captions
from training_core import ClassifierHead,YamnetEncoder,feature_record

PARTS=[f'test-audio/real-clips/17 skirmish game audio - part {i:02d}.mp4' for i in range(2,5)]


def collect_round6(dataset,historical,previous):
    records,part01=collect_records(dataset,historical)
    if {r['id']:r['csv_revision'] for r in records}!=previous:
        raise ValueError('Round-5 labels changed; stop rather than confound the comparison')
    added=[]
    for relative in PARTS:
        key=next(k for k,c in dataset.clips.items() if c['relative']==relative)
        clip=dataset.clip(key);value=workspace(dataset,key)
        if not value['review']['complete']:
            raise ValueError(f"Not marked finished: {clip['path'].name}")
        if not value['rows']:raise ValueError('Empty clips must not enter training as silence')
        if clip.get('voice_removal_required',True):raise ValueError('Expected original-only game audio')
        records.append({'id':key,'path':clip['path'],'cleaned':None,'truth':value['rows'],
                        'csv_revision':value['revision'],'duration':dataset.metadata(key)['duration'],
                        'fold_group':clip['fold_group']})
        added.append(key)
    return records,part01,added


def allowed_ids(records,old_ids,held_id,new):
    group=next(r['fold_group'] for r in records if r['id']==held_id)
    return sorted(r['id'] for r in records if r['fold_group']!=group and (new or r['id'] in old_ids))


def main():
    out=ROOT/'training/round-6';local=ROOT/'.local/training-runs/round-6'
    if (out/'protocol.json').exists():raise SystemExit('Round 6 already started/recorded; do not overwrite')
    previous=json.loads((ROOT/'training/round-5/protocol.json').read_text())
    for name,digest in previous['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'Frozen round-5 implementation changed: {name}')
    previous_checkpoint=ROOT/'.local/training-runs/round-5/model.joblib'
    records,part01,added=collect_round6(Dataset(ROOT),json.loads((ROOT/'training/round-2/protocol.json').read_text())['development_labels'],previous['annotations'])
    assert len(records)==15
    old_ids=set(previous['annotations']);original_ids=set(previous['original_ids']);by_id={r['id']:r for r in records}
    skirmish={part01,*added};assert len({by_id[k]['fold_group'] for k in skirmish})==1
    policy=load_policy();config=json.loads((ROOT/'baseline-config.json').read_text());threshold=.9
    live=ROOT/'.local/training-runs/round-2/model.joblib'
    protocol={'round':6,'created_at':datetime.now().astimezone().isoformat(),
        'change':'Add only finished skirmish 02-04 to round 5; parts 05-06 never loaded',
        'threshold':threshold,'threshold_selection':'Fixed 0.9; no calibration or hyperparameter search',
        'encoder_head_augmentation':'Exact round-5 setup: fixed YAMNet mono embeddings and ClassifierHead; original+existing voice-reduced for original11, original-only for skirmish01-04; no paste/separation',
        'validation':'Matched leave-one-recording-group-out for both versions. All four skirmish minutes held out together on every skirmish validation fold.',
        'comparison':'Round-5 control versus round 6 on SAME original 12 clips (primary), original11 subset and same15 clips (supplementary)',
        'old_ids':sorted(old_ids),'added_ids':added,'original_ids':sorted(original_ids),'skirmish_ids':sorted(skirmish),
        'excluded_unlabeled_parts':[f'test-audio/real-clips/17 skirmish game audio - part {i:02d}.mp4' for i in [5,6]],
        'annotations':{r['id']:r['csv_revision'] for r in records},'policy':policy,
        'source_hashes':{**previous['source_hashes'],'run_round6.py':sha(ROOT/'run_round6.py')},
        'round5_checkpoint_sha256':sha(previous_checkpoint),'live_checkpoint_sha256':sha(live),
        'test_data_opened':False,'live_model_changed':False}
    atomic_json(out/'protocol.json',protocol)
    saved=joblib.load(previous_checkpoint);encoder=YamnetEncoder()
    assert saved['threshold']==threshold and encoder.identity==saved['encoder_sha256']
    features={}
    for r in records:
        variants=['original','voice_removed'] if r['cleaned'] is not None else ['original']
        features[r['id']]={variant:feature_record(r,variant,encoder) for variant in variants}
        print(f"Features ready: {r['path'].name}",flush=True)
    cache={};audit={}
    def fit(ids):
        key=tuple(sorted(ids))
        if key not in cache:
            x=np.concatenate([v['features'] for k in key for v in features[k].values()])
            y=np.concatenate([v['targets'] for k in key for v in features[k].values()])
            cache[key]=ClassifierHead().fit(x,y)
            audit[','.join(key)]={'train_ids':list(key),'variants':{k:list(features[k]) for k in key},'windows':len(x)}
        return cache[key]
    captions={'round5':{},'round6':{}};probabilities={'round5':{},'round6':{}};folds=[]
    for r in records:
        held=r['id'];entry={'held_out_id':held,'held_out_group':r['fold_group']}
        for name,include_new in [('round5',False),('round6',True)]:
            ids=allowed_ids(records,old_ids,held,include_new)
            assert held not in ids and all(by_id[k]['fold_group']!=r['fold_group'] for k in ids)
            if held in skirmish:assert not set(ids)&skirmish
            p=fit(ids).predict(features[held]['original']['features']);probabilities[name][held]=p
            captions[name][held]=apply_policy(learned_captions(features[held]['original'],p,threshold,config),policy)
            entry[name+'_train_ids']=ids
        folds.append(entry);atomic_json(local/'fold-progress.json',folds)
        print(f"Matched fold {len(folds)}/15: {r['path'].name}",flush=True)
    for key in skirmish:assert np.array_equal(probabilities['round5'][key],probabilities['round6'][key])
    with np.load(ROOT/'.local/training-runs/round-5/new-outer-probabilities.npz',allow_pickle=False) as previous_probs:
        for key in old_ids:assert np.array_equal(probabilities['round5'][key],previous_probs[key])
    rebuilt=fit(sorted(old_ids))
    assert np.array_equal(rebuilt.scaler.mean_,saved['head'].scaler.mean_) and np.array_equal(rebuilt.scaler.scale_,saved['head'].scaler.scale_)
    for a,b in zip(rebuilt.models,saved['head'].models):
        if isinstance(a,float):assert a==b
        else:assert np.array_equal(a.coef_,b.coef_) and np.array_equal(a.intercept_,b.intercept_)
    scopes={'same_12_clips':[r for r in records if r['id'] in old_ids],
            'original_11_clips':[r for r in records if r['id'] in original_ids],
            'same_15_clips':records,'skirmish_recording_held_out':[r for r in records if r['id'] in skirmish]}
    scores={scope:{} for scope in scopes}
    for scope,chosen in scopes.items():
        for name in captions:
            value=score_policy_round(chosen,captions[name],policy);value['label_details']=label_details(chosen,captions[name],policy)
            scores[scope][name]=value
    result={'round':6,'threshold':threshold,'policy_version':2,'primary_comparison_scope':'same_12_clips',
        'scores':scores,'folds':folds,'old_predictions_and_full_head_reproduced_bit_exactly':True,
        'unlabeled_parts_used':False,'test_data_opened':False,'live_model_changed':False}
    atomic_json(out/'metrics.json',result)
    for name in captions:
        atomic_json(local/f'{name}-outer-captions.json',captions[name])
        np.savez_compressed(local/f'{name}-outer-probabilities.npz',**probabilities[name])
    head=fit(sorted(by_id));joblib.dump({**saved,'head':head,'training_round':6,'threshold':threshold},local/'model.joblib')
    atomic_json(out/'fold-audit.json',audit)
    atomic_json(out/'checkpoint.json',{'model':str((local/'model.joblib').relative_to(ROOT)),'training_ids':sorted(by_id),
        'original_only_ids':sorted(skirmish),'threshold':threshold,'policy_version':2,'active_live_checkpoint_changed':False})
    assert sha(previous_checkpoint)==protocol['round5_checkpoint_sha256'] and sha(live)==protocol['live_checkpoint_sha256']
    for r in records:assert sha(r['path'].with_suffix('.csv'))==r['csv_revision']
    for name,digest in protocol['source_hashes'].items():assert sha(ROOT/name)==digest
    print(json.dumps({scope:{v:{'overall':s['primary']['caught_pct'],'false_per_min':s['primary']['false_per_min'],
        'footsteps':s['primary']['per_label']['footsteps'],'gunshots':s['primary']['per_label']['gunshot'],
        'arrows':f"{s['primary']['arrows_correct']}/{s['primary']['arrows_scored']}"} for v,s in pair.items()} for scope,pair in scores.items()},indent=2),flush=True)

if __name__=='__main__':main()
