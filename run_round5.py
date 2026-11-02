"""One frozen round-2 replication: add only reviewed skirmish part 01; fixed .9.

Development-only matched LOCO comparisons. Never uses held-out test accessors.
The saved live checkpoint and all policies remain unchanged.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from caption_policy import apply_policy,load_policy,supported_by_span
from labeler import Dataset,ROOT
from policy_scoring import score_policy_round
from review import atomic_json,augmentation_status,workspace
from run_training import learned_captions
from scoring import LABELS,match_events
from training_core import ClassifierHead,YamnetEncoder,feature_record

NEW_PATH='test-audio/real-clips/17 skirmish game audio - part 01.mp4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_records(dataset,original_revisions):
    new_id=next(k for k,c in dataset.clips.items() if c['relative']==NEW_PATH)
    # Whitelist: the original eleven and part 01 ONLY. Never read other parts.
    wanted=list(original_revisions)+[new_id]
    records=[]
    for key in wanted:
        clip=dataset.clip(key);value=workspace(dataset,key)
        if not value['review']['complete']:raise ValueError(f"Finish review first: {clip['path'].name}")
        if key in original_revisions and value['revision']!=original_revisions[key]:
            raise ValueError('An original label file changed; stop rather than confound the comparison')
        if key==new_id and len(value['rows'])!=34:raise ValueError('Expected the 34 reviewed part-01 labels')
        cleaned=None
        if clip.get('voice_removal_required',True):
            status=augmentation_status(dataset.root,key)
            if status.get('state')!='ready' or status.get('csv_revision')!=value['revision']:
                raise ValueError('Original paired augmentation is missing or stale')
            cleaned=dataset.root/status['audio']
            if not cleaned.is_file():raise ValueError('Missing paired audio')
            stamp={'size':clip['path'].stat().st_size,'mtime_ns':clip['path'].stat().st_mtime_ns}
            if status.get('source')!=stamp:raise ValueError('Paired source changed')
        elif key!=new_id:raise ValueError('Original round-2 pairing changed')
        records.append({'id':key,'path':clip['path'],'cleaned':cleaned,'truth':value['rows'],
                        'csv_revision':value['revision'],'duration':dataset.metadata(key)['duration'],
                        'fold_group':clip.get('fold_group',key)})
    return records,new_id


def fold_ids(records,original_ids,held_id,include_new):
    group=next(r['fold_group'] for r in records if r['id']==held_id)
    return sorted(r['id'] for r in records if r['fold_group']!=group and (include_new or r['id'] in original_ids))


def label_details(records,captions,policy):
    minutes=sum(r['duration'] for r in records)/60
    result={label:{'false_captions':0,'arrows_correct':0,'arrows_scored':0} for label in ['footsteps','gunshot']}
    for record in records:
        targets=[r for r in record['truth'] if r['label'] in policy['enabled_labels'] and r['direction']!='center']
        predictions=captions[record['id']]
        for p in predictions:
            if p['label'] in result and not supported_by_span(p,record['truth'],policy):result[p['label']]['false_captions']+=1
        for i,j in match_events(targets,predictions,policy['caught_onset_tolerance_s']):
            t,p=targets[i],predictions[j]
            if t['label'] in result and t['direction'] in {'left','right'} and p['direction'] in {'left','right'}:
                result[t['label']]['arrows_scored']+=1
                result[t['label']]['arrows_correct']+=t['direction']==p['direction']
    for value in result.values():
        value['false_per_min']=value['false_captions']/minutes
        value['arrow_correct_pct']=100*value['arrows_correct']/value['arrows_scored'] if value['arrows_scored'] else None
    return result


def main():
    out=ROOT/'training/round-5';local=ROOT/'.local/training-runs/round-5'
    if (out/'protocol.json').exists():raise SystemExit('Round 5 already recorded/started; do not overwrite it')
    historical=json.loads((ROOT/'training/round-2/protocol.json').read_text())
    if sha(ROOT/'training_core.py')!=historical['core_sha256']:raise ValueError('Round-2 encoder/head code changed')
    frozen=json.loads((ROOT/'training/policy-v2-round-2/protocol.json').read_text())
    for name,key in [('caption-policy-v2.json','policy_sha256'),('caption_policy.py','filter_sha256'),('policy_scoring.py','scorer_sha256'),('scoring.py','legacy_scorer_sha256')]:
        if sha(ROOT/name)!=frozen[key]:raise ValueError('Policy-v2 implementation changed')
    old_checkpoint=ROOT/'.local/training-runs/round-2/model.joblib'
    if sha(old_checkpoint)!=frozen['model_sha256']:raise ValueError('Original checkpoint changed')
    dataset=Dataset(ROOT);records,new_id=collect_records(dataset,historical['development_labels'])
    assert len(records)==12
    original_ids=set(historical['development_labels']);by_id={r['id']:r for r in records}
    ignored=[c['relative'] for k,c in dataset.clips.items() if c['relative'].startswith('test-audio/real-clips/17 skirmish game audio - part ') and k!=new_id]
    assert len(ignored)==5
    policy=load_policy();config=json.loads((ROOT/'baseline-config.json').read_text());threshold=.9
    protocol={'created_at':datetime.now().astimezone().isoformat(),'round':5,'change':'Only add the 34 human labels from skirmish part 01',
        'validation':'Matched whole-recording-group leave-one-clip-out on the same 12 clips for both models; original-11 matched subset also reported',
        'threshold':threshold,'threshold_selection':'Fixed by user before fitting. No inner threshold search or other tuning.',
        'encoder':'YAMNet mono 1024-D; .975s window/.48s hop','head':'Unchanged training_core.ClassifierHead (C=.1, balanced, fold-only scaler, liblinear, random_state=20261101)',
        'augmentation':'Original + existing aligned voice-removed sibling for original 11; original only for game-only part 01. No cut/paste or new separation.',
        'original_ids':sorted(original_ids),'added_id':new_id,'excluded_unlabeled_parts':ignored,
        'annotations':{r['id']:r['csv_revision'] for r in records},'policy':policy,
        'source_hashes':{name:sha(ROOT/name) for name in ['training_core.py','caption-policy-v2.json','caption_policy.py','policy_scoring.py','scoring.py','run_training.py','run_round5.py','baseline-config.json']},
        'old_checkpoint_sha256':sha(old_checkpoint),'test_data_opened':False,'live_model_changed':False}
    atomic_json(out/'protocol.json',protocol)
    saved=joblib.load(old_checkpoint);encoder=YamnetEncoder()
    if saved['threshold']!=threshold or encoder.identity!=saved['encoder_sha256']:raise ValueError('Round-2 encoder or threshold mismatch')
    features={}
    for record in records:
        variants=['original','voice_removed'] if record['cleaned'] is not None else ['original']
        features[record['id']]={variant:feature_record(record,variant,encoder) for variant in variants}
        print(f"Features ready: {record['path'].name} ({', '.join(variants)})",flush=True)
    cache={};audit={}
    def fit(ids):
        key=tuple(sorted(ids))
        if not key:raise ValueError('Empty fold')
        if key not in cache:
            x=np.concatenate([v['features'] for k in key for v in features[k].values()])
            y=np.concatenate([v['targets'] for k in key for v in features[k].values()])
            cache[key]=ClassifierHead().fit(x,y)
            audit[','.join(key)]={'train_ids':list(key),'variants':{k:list(features[k]) for k in key},'windows':len(x)}
        return cache[key]
    captions={'old':{},'new':{}};probabilities={'old':{},'new':{}};folds=[]
    for record in records:
        held=record['id'];entry={'held_out_id':held,'held_out_group':record['fold_group']}
        for version,include_new in [('old',False),('new',True)]:
            allowed=fold_ids(records,original_ids,held,include_new)
            assert held not in allowed and all(by_id[k]['fold_group']!=record['fold_group'] for k in allowed)
            p=fit(allowed).predict(features[held]['original']['features'])
            probabilities[version][held]=p
            raw=learned_captions(features[held]['original'],p,threshold,config)
            captions[version][held]=apply_policy(raw,policy);entry[version+'_train_ids']=allowed
        folds.append(entry);atomic_json(local/'fold-progress.json',folds)
        print(f"Matched fold {len(folds)}/12: {record['path'].name}",flush=True)
    # The new minute is held out in BOTH systems on its own scoring fold.
    assert np.array_equal(probabilities['old'][new_id],probabilities['new'][new_id])
    rebuilt=fit(sorted(original_ids))
    assert np.array_equal(rebuilt.scaler.mean_,saved['head'].scaler.mean_)
    assert np.array_equal(rebuilt.scaler.scale_,saved['head'].scaler.scale_)
    for a,b in zip(rebuilt.models,saved['head'].models):
        if isinstance(a,float):assert a==b
        else:
            assert np.array_equal(a.coef_,b.coef_) and np.array_equal(a.intercept_,b.intercept_)
    scores={}
    for scope,chosen in [('same_12_clips',records),('original_11_clips',[r for r in records if r['id'] in original_ids]),('new_part_01_held_out',[by_id[new_id]])]:
        scores[scope]={}
        for version in ['old','new']:
            value=score_policy_round(chosen,captions[version],policy)
            value['label_details']=label_details(chosen,captions[version],policy)
            scores[scope][version]=value
    result={'round':5,'threshold':threshold,'policy_version':2,'scores':scores,'folds':folds,
            'original_full_head_reproduced_bit_exactly':True,'unlabeled_parts_used':False,'test_data_opened':False,'live_model_changed':False}
    atomic_json(out/'metrics.json',result);atomic_json(out/'fold-audit.json',audit)
    for version in ['old','new']:
        atomic_json(local/f'{version}-outer-captions.json',captions[version])
        np.savez_compressed(local/f'{version}-outer-probabilities.npz',**probabilities[version])
    head=fit(sorted(by_id))
    joblib.dump({**saved,'head':head,'threshold':threshold,'training_round':5,'policy_version':2},local/'model.joblib')
    atomic_json(out/'checkpoint.json',{'model':str((local/'model.joblib').relative_to(ROOT)),'threshold':threshold,
        'active_live_checkpoint_changed':False,'training_ids':sorted(by_id),'original_only_ids':[new_id]})
    assert sha(old_checkpoint)==protocol['old_checkpoint_sha256']
    for record in records:
        assert sha(record['path'].with_suffix('.csv'))==record['csv_revision']
    print(json.dumps({scope:{v:{k:s['primary'][k] for k in ['caught','targets','caught_pct','false_per_min','arrows_correct','arrows_scored']} for v,s in pair.items()} for scope,pair in scores.items()},indent=2),flush=True)

if __name__=='__main__':main()
