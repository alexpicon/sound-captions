"""FINAL ONLY: compare cached pretrained baseline and round 2 with policy v2.

Do not run until the user explicitly requests the final after evaluation.
Importing or --help never opens test media/labels. Lab-final remains closed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from caption_policy import apply_policy,load_policy
from labeler import Dataset,ROOT,read_annotations,validate_rows
from policy_scoring import score_policy_round
from review import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-final',action='store_true',help='Use only after explicit user authorization to unseal final real/CS tests')
    args=parser.parse_args(argv)
    if not args.confirm_final:
        parser.error('Tests remain closed. This command requires --confirm-final after the user authorizes final evaluation.')
    out=ROOT/'evaluations/final-policy-v2'
    if (out/'protocol.json').exists():raise SystemExit('Final evaluation is already recorded/started. Do not overwrite it.')
    # All sealed access is below the explicit final-only gate.
    from live_engine import LiveEngine
    from model_tools import decode
    engine=LiveEngine();policy=load_policy()
    atomic_json(out/'protocol.json',{'created_at':datetime.now().astimezone().isoformat(),
        'policy':policy,'policy_sha256':hashlib.sha256((ROOT/'caption-policy-v2.json').read_bytes()).hexdigest(),
        'learned_model':'round-2','threshold':.9,'head_sha256':hashlib.sha256((ROOT/'.local/training-runs/round-2/model.joblib').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256((ROOT/'run_final.py').read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256((ROOT/'policy_scoring.py').read_bytes()).hexdigest(),
        'filter_sha256':hashlib.sha256((ROOT/'caption_policy.py').read_bytes()).hexdigest(),
        'legacy_scorer_sha256':hashlib.sha256((ROOT/'scoring.py').read_bytes()).hexdigest(),
        'engine_sha256':hashlib.sha256((ROOT/'live_engine.py').read_bytes()).hexdigest(),
        'frozen_baseline_config_sha256':hashlib.sha256((ROOT/'baseline-config.json').read_bytes()).hexdigest(),
        'both_baseline_and_after_use_policy_v2':True,'lab_final_opened':False,'no_training_or_threshold_tuning':True})
    dataset=Dataset(ROOT);records=[];before={};after={}
    for key,clip in dataset.test_clips.items():
        # Freeze learned predictions before reading this clip's answer key.
        audio=decode(clip['path'],engine.rate);duration=dataset.metadata(key,'test')['duration'];engine.reset();captions=[];last_end=-1
        for start in range(0,len(audio),engine.hop):
            end=min(start+engine.size,len(audio));end_s=min(end/engine.rate,duration)
            if start/engine.rate>=duration or end_s<=last_end:break
            values,_=engine.window(audio[start:end],end_s);captions.extend(values);last_end=end_s
        after[key]=captions
        cached=ROOT/'.local/evaluations/round-0'/f'{key}-predictions.json'
        baseline=json.loads(cached.read_text())
        if baseline['clip_id']!=key:raise ValueError('Baseline ID mismatch')
        before[key]=apply_policy(baseline['predictions'],policy)
        atomic_json(out/f'{key}-predictions.json',{'before':before[key],'after':after[key]})
        rows=validate_rows(read_annotations(clip['path'].with_suffix('.csv')),duration)
        records.append({'id':key,'path':clip['path'],'truth':rows,'duration':duration,'group':clip['group']})
    result={'policy_version':2,'lab_final_opened':False,
            'groups':{group:{'before':score_policy_round([r for r in records if r['group']==group],before,policy),
                             'after':score_policy_round([r for r in records if r['group']==group],after,policy)}
                       for group in sorted({r['group'] for r in records})},
            'context_note':'Quiet means outside supplied talking/music spans; missing context labels are not proof of silence.'}
    atomic_json(out/'metrics.json',result)
    lines=['\n## Final comparison — policy v2 on both before and after\n',
           '| Test group | Model | Caught / targets | False/min | Unmatched/min | Arrow correct |',
           '|---|---|---|---|---|---|']
    for group,pair in result['groups'].items():
        for name,scores in pair.items():
            m=scores['primary'];recall='N/A' if m['caught_pct'] is None else f"{m['caught_pct']:.1f}%"
            lines.append(f"| {group} | {name} | {m['caught']}/{m['targets']} ({recall}) | {m['false_per_min']:.2f} | {m['unmatched_per_min']:.2f} | {m['arrows_correct']}/{m['arrows_scored']} |")
    lines+=['\nSame versioned false-support, no-reload and burst-merging rules apply to both models. These supersede the old-policy baseline for this comparison. This opens the five real/CS tests only; lab-final remains sealed. Full per-label/context/all-direction metrics are in `evaluations/final-policy-v2/metrics.json`. No test-based tuning was performed.\n']
    with (ROOT/'rounds.md').open('a') as handle:handle.write('\n'.join(lines))
    print('Final comparison recorded. Do not tune on these test results.')


if __name__=='__main__':main()
