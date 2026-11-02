"""Replay saved DEVELOPMENT captions under policy v2. No model fit or sealed access."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from caption_policy import apply_policy,load_policy
from labeler import Dataset,ROOT
from policy_scoring import score_policy_round
from review import atomic_json
from training_core import development_clips


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'training/policy-v2-round-2'
    if (out/'protocol.json').exists():raise SystemExit('Replay already recorded; do not overwrite it.')
    records=development_clips(Dataset(ROOT));policy=load_policy()
    original=json.loads((ROOT/'training/round-2/metrics.json').read_text())
    protocol=json.loads((ROOT/'training/round-2/protocol.json').read_text())
    assert {r['id']:r['csv_revision'] for r in records}==protocol['development_labels']
    rawpath=ROOT/'.local/training-runs/round-2/outer-captions.json'
    raw=json.loads(rawpath.read_text());assert set(raw)=={r['id'] for r in records}
    predictions={k:apply_policy(v,policy) for k,v in raw.items()}
    reference_raw=json.loads((ROOT/'.local/training-runs/round-0/outer-captions.json').read_text())
    reference=score_policy_round(records,{k:apply_policy(v,policy) for k,v in reference_raw.items()},policy)
    unmerged=score_policy_round(records,{k:[p for p in v if p['label'] in policy['enabled_labels']] for k,v in raw.items()},policy)
    atomic_json(out/'protocol.json',{'created_at':datetime.now().astimezone().isoformat(),'policy':policy,
        'policy_sha256':sha(ROOT/'caption-policy-v2.json'),'filter_sha256':sha(ROOT/'caption_policy.py'),
        'scorer_sha256':sha(ROOT/'policy_scoring.py'),'legacy_scorer_sha256':sha(ROOT/'scoring.py'),
        'model_sha256':sha(ROOT/'.local/training-runs/round-2/model.joblib'),
        'raw_captions_sha256':sha(rawpath),'annotations':protocol['development_labels'],
        'fold_thresholds':{f['outer_validation_id']:f['threshold'] for f in original['folds']},
        'deployment_threshold':.9,'training_performed':False,'test_data_opened':False,
        'replay_limitation':'Original scheduled OOF emissions only. No per-window round-2 scores were saved. Live/final use that same legacy scheduler before the v2 filter; hidden reload candidates can still occupy internal scheduler slots.'})
    result=score_policy_round(records,predictions,policy)
    result.update({'training_performed':False,'test_data_opened':False,
        'disabled_noncenter_reload_targets':sum(r['label']=='reload' and r['direction']!='center' for record in records for r in record['truth']),
        'raw_captions':sum(map(len,raw.values())),'reload_captions_removed':sum(p['label']=='reload' for rows in raw.values() for p in rows),
        'repeats_merged':sum(map(len,raw.values()))-sum(map(len,predictions.values()))-sum(p['label']=='reload' for rows in raw.values() for p in rows)})
    atomic_json(out/'metrics.json',result);atomic_json(out/'reference-v2.json',reference)
    atomic_json(ROOT/'.local/evaluations/policy-v2-dev/round-2-captions.json',predictions)
    def row(name,s):
        m=s['primary'];return f"| {name} | {m['caught']}/{m['targets']} ({m['caught_pct']:.1f}%) | {m['false_per_min']:.2f} | {m['unmatched_per_min']:.2f} | {m['arrows_correct']}/{m['arrows_scored']} |"
    lines=['\n## Policy v2 — round-2 development replay; no new training\n',
        'The user reviewed false-caption candidates and explicitly authorized new scoring/repeat rules, removal of reload captions, and proceeding to step 4. The former 35% / 2-per-minute gate is superseded by that instruction; no new model was fitted and no sealed tests were opened.\n',
        'A caption is supported only by the **same label** over the labeled span, through at least 2 seconds after that sound starts. The existing 1-second early allowance is retained so an otherwise-correct early caption is not counted as false. Same-label/side candidates less than 2 seconds apart merge into one chain; suppressed candidates extend the quiet-gap timer. Different sides remain independent. Reload captions are suppressed and reload targets are excluded from active-class recall, with the six formerly eligible reload targets reported separately. Original annotations were not changed.\n',
        'Caught recall remains maximum-cardinality one-to-one same-label onset matching within 1 second. A later supported caption can therefore be neither false nor a newly caught onset. Unmatched/min is retained as an extra diagnostic and includes those later supported captions; it is not equivalent to a wrong-caption rate. Arrow correctness remains independent of sound-label support.\n',
        '| Comparison | Caught / targets | False/min | Unmatched/min | Arrow correct |','|---|---|---|---|---|',
        row('Historical round 2, old rules incl. reload (not directly comparable)',original),
        row('Pretrained development reference, policy v2',reference),
        row('Round 2, reload hidden + span scoring, before repeat merge',unmerged),row('Round 2, complete policy v2',result),
        '\nBoth the reference and replay use the same v2 postprocessor/scorer. The old 47.5% result included 40 targets; the active v2 denominator is 34, so do not present any change in percentage as a training improvement.\n',
        '| Active class | Caught / targets | Recall |','|---|---|---|']
    for label,m in result['primary']['per_label'].items():
        recall=f"{m['caught_pct']:.1f}%" if m['caught_pct'] is not None else 'N/A (disabled or no targets)'
        lines.append(f"| {label} | {m['caught']}/{m['targets']} | {recall} |")
    lines+=['\n| Context | Caught / targets | Recall | False/min | Arrow correct |','|---|---|---|---|---|']
    for context,m in result['contexts'].items():
        lines.append(f"| {context} | {m['caught']}/{m['targets']} | {m['caught_pct']:.1f}% | {m['false_per_min']:.2f} | {m['arrows_correct']}/{m['arrows_scored']} |" if m else f'| {context} | N/A | N/A | N/A | N/A |')
    lines += [f"\nReplay removed {result['reload_captions_removed']} reload captions and merged {result['repeats_merged']} repeat candidates from {result['raw_captions']} original emissions. Only cached outer held-out round-2 emissions were replayed: ten folds used 0.9 and clip 03 used 0.8, exactly as originally inner-calibrated. The live full-data round-2 checkpoint keeps its saved deployment threshold **0.9**. Do not describe the replay as fixed-0.9 cross-validation or as inference from the full-data model on its own training clips.\n",
        'The exact legacy scheduler is retained before v2 postprocessing for replay/live/final consistency. Consequently a hidden reload candidate can still consume a raw internal slot; this conservative limitation cannot be removed from cached OOF captions without unavailable per-window predictions. The final scorer must apply the same versioned rules to **both** the cached pretrained baseline captions and the learned-model captions and report active-class denominators. Old-policy final scores must not be compared against new-policy scores.\n',
        'Frozen policy: `caption-policy-v2.json`. Replay protocol/results: `training/policy-v2-round-2/`. This is a user-authorized scoring/presentation revision, **not training round 5**.\n']
    with (ROOT/'rounds.md').open('a') as f:f.write('\n'.join(lines))
    print(json.dumps({k:result[k] for k in ['primary','reload_captions_removed','repeats_merged']},indent=2))

if __name__=='__main__':main()
