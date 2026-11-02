"""FINAL ONLY: compare cached pretrained baseline and active frozen model with policy v2.

Do not run until the user explicitly requests the final after evaluation.
Importing or --help never opens test media/labels. Lab-final remains closed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo
from caption_policy import apply_policy,load_policy
from labeler import Dataset,ROOT,read_annotations,validate_rows
from policy_scoring import score_policy_round
from review import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-final',action='store_true',help='Use only after explicit user authorization to unseal final real/CS tests')
    parser.add_argument('--include-lab-final',action='store_true',help='Also evaluate the held-out lab clip after explicit final-test authorization')
    parser.add_argument('--no-rounds-log',action='store_true',help='Save evaluation artifacts without writing rounds.md')
    args=parser.parse_args(argv)
    if not args.confirm_final:
        parser.error('Tests remain closed. This command requires --confirm-final after the user authorizes final evaluation.')
    out=ROOT/'evaluations/final-policy-v2'
    if (out/'protocol.json').exists():raise SystemExit('Final evaluation is already recorded/started. Do not overwrite it.')
    # All sealed access is below the explicit final-only gate.
    from live_engine import LiveEngine
    from model_tools import decode
    engine=LiveEngine();policy=load_policy()
    if engine.deployment['model_round'] != 2 or any(t != .9 for t in engine.deployment['thresholds'].values()):
        raise ValueError('This final comparison requires the selected round-2 model at 0.9')
    baseline_config=json.loads((ROOT/'baseline-config.json').read_text())
    baseline_protocol=json.loads((ROOT/'evaluations/round-0/protocol.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(ROOT/'baseline-config.json')!=baseline_protocol['config_sha256'] or sha(ROOT/baseline_config['model'])!=baseline_protocol['model_sha256']:
        raise ValueError('Frozen baseline settings or weights changed')
    rounds_before=sha(ROOT/'rounds.md')
    atomic_json(out/'protocol.json',{'created_at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'policy':policy,'policy_sha256':hashlib.sha256((ROOT/'caption-policy-v2.json').read_bytes()).hexdigest(),
        'learned_model':f"round-{engine.deployment['model_round']}",'thresholds':engine.deployment['thresholds'],
        'head_sha256':engine.deployment['model_sha256'],
        'live_config_sha256':hashlib.sha256((ROOT/'live-config.json').read_bytes()).hexdigest(),
        'raw_scheduler_ttl_s':engine.config['caption_ttl_s'],'overlay_display_duration_s':engine.deployment['display_duration_s'],
        'script_sha256':hashlib.sha256((ROOT/'run_final.py').read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256((ROOT/'policy_scoring.py').read_bytes()).hexdigest(),
        'filter_sha256':hashlib.sha256((ROOT/'caption_policy.py').read_bytes()).hexdigest(),
        'legacy_scorer_sha256':hashlib.sha256((ROOT/'scoring.py').read_bytes()).hexdigest(),
        'engine_sha256':hashlib.sha256((ROOT/'live_engine.py').read_bytes()).hexdigest(),
        'frozen_baseline_config_sha256':hashlib.sha256((ROOT/'baseline-config.json').read_bytes()).hexdigest(),
        'both_baseline_and_after_use_policy_v2':True,'lab_final_opened':args.include_lab_final,
        'rounds_log_requested':not args.no_rounds_log,'rounds_sha256_before':rounds_before,
        'no_training_or_threshold_tuning':True})
    dataset=Dataset(ROOT);records=[];before={};after={}
    if args.include_lab_final:
        for relative in json.loads((ROOT/'splits.json').read_text())['sealed_tests']['lab_final']:
            key=hashlib.sha256(relative.encode()).hexdigest()[:16]
            dataset.test_clips[key]={'path':ROOT/relative,'relative':relative,'group':'lab_final'}
    baseline_interpreter=None
    for key,clip in dataset.test_clips.items():
        # Freeze learned predictions before reading this clip's answer key.
        audio=decode(clip['path'],engine.rate);duration=dataset.metadata(key,'test')['duration'];engine.reset();captions=[];last_end=-1
        for start in range(0,len(audio),engine.hop):
            end=min(start+engine.size,len(audio));end_s=min(end/engine.rate,duration)
            if start/engine.rate>=duration or end_s<=last_end:break
            values,_=engine.window(audio[start:end],end_s);captions.extend(values);last_end=end_s
        after[key]=captions
        cached=ROOT/'.local/evaluations/round-0'/f'{key}-predictions.json'
        if cached.exists():
            baseline=json.loads(cached.read_text())
            old=json.loads((cached.parent/f'{key}-metrics.json').read_text())
            if old['media_sha256']!=sha(clip['path']):raise ValueError('Baseline media changed')
        elif clip['group']=='lab_final' and args.include_lab_final:
            from ai_edge_litert.interpreter import Interpreter
            from run_baseline import infer
            if baseline_interpreter is None:
                baseline_interpreter=Interpreter(model_path=str(ROOT/baseline_config['model']),num_threads=2)
                baseline_interpreter.allocate_tensors()
            predictions,_,_,_=infer(clip['path'],duration,baseline_interpreter,baseline_config)
            baseline={'clip_id':key,'predictions':predictions}
        else:raise ValueError('Missing frozen baseline predictions')
        if baseline['clip_id']!=key:raise ValueError('Baseline ID mismatch')
        before[key]=apply_policy(baseline['predictions'],policy)
        atomic_json(out/f'{key}-predictions.json',{'before':before[key],'after':after[key]})
        rows=validate_rows(read_annotations(clip['path'].with_suffix('.csv')),duration)
        records.append({'id':key,'path':clip['path'],'truth':rows,'duration':duration,'group':clip['group']})
        print(f"Final predictions fixed and scored: {clip['path'].name}",flush=True)
    def evaluate(selected,predictions):
        scores=score_policy_round(selected,predictions,policy)
        scores['by_label']={}
        for label in ['gunshot','footsteps']:
            subset=[{**r,'truth':[t for t in r['truth'] if t['label'] in {label,'talking','music'}]} for r in selected]
            values={key:[p for p in rows if p['label']==label] for key,rows in predictions.items()}
            per_label=score_policy_round(subset,values,policy)
            scores['by_label'][label]={'primary':per_label['primary'],'contexts':per_label['contexts']}
        return scores
    result={'policy_version':2,'lab_final_opened':args.include_lab_final,
            'groups':{group:{'before':evaluate([r for r in records if r['group']==group],before),
                             'after':evaluate([r for r in records if r['group']==group],after)}
                       for group in sorted({r['group'] for r in records})},
            'annotation_manifest':[{'clip':str(r['path'].relative_to(ROOT)),
                'annotation_sha256':sha(r['path'].with_suffix('.csv')),
                'context_spans':{label:sum(t['label']==label for t in r['truth']) for label in ['talking','music']}}
                for r in records],
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
    if not args.no_rounds_log:
        with (ROOT/'rounds.md').open('a') as handle:handle.write('\n'.join(lines))
    elif sha(ROOT/'rounds.md')!=rounds_before:
        raise ValueError('rounds.md changed during evaluation')
    print('Final comparison saved; rounds.md '+('unchanged.' if args.no_rounds_log else 'updated.'))


if __name__=='__main__':main()
