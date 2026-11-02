"""Policy-v2 scorer. Historical scoring.py remains unchanged for reproducibility."""
from caption_policy import load_policy, supported_by_span
from scoring import aggregate, score_clip
from run_training import context_scores


def score_clip_v2(truth, captions, duration, policy=None, hide_centered=True):
    policy = policy or load_policy()
    enabled = [r for r in truth if r['label'] in policy['enabled_labels']]
    if any(p['label'] not in policy['enabled_labels'] for p in captions):
        raise ValueError('Apply the shared caption postprocessor before scoring')
    result = score_clip(enabled, captions, duration, policy['caught_onset_tolerance_s'], hide_centered)
    false = sum(not supported_by_span(p, enabled, policy) for p in captions)
    result['false_captions'] = false
    result['false_per_min'] = false / (duration / 60) if duration else None
    return result


def score_policy_round(records, predictions, policy=None):
    policy = policy or load_policy()
    clips = []
    for record in records:
        captions = predictions[record['id']]
        active_record = {**record, 'truth': [r for r in record['truth']
                          if r['label'] in policy['enabled_labels'] or r['label'] in ['talking', 'music']]}
        contexts = context_scores(active_record, captions)
        intervals = {name: [(r['start_s'], r['end_s']) for r in record['truth'] if r['label']==name]
                     for name in ['talking','music']}
        def inside(t, name):
            if name=='quiet':return not any(a<=t<b for rows in intervals.values() for a,b in rows)
            return any(a<=t<b for a,b in intervals[name])
        for name, scores in contexts.items():
            if scores is not None:
                count=sum(inside(p['start_s'],name) and not supported_by_span(p,record['truth'],policy) for p in captions)
                scores['false_captions']=count
                scores['false_per_min']=count/(scores['duration_s']/60)
        clips.append({'clip_id':record['id'],'clip':str(record['path']),
                      'primary':score_clip_v2(record['truth'],captions,record['duration'],policy),
                      'all_directions':score_clip_v2(record['truth'],captions,record['duration'],policy,False),
                      'contexts':contexts})
    return {'policy_version':policy['version'], 'primary':aggregate([r['primary'] for r in clips]),
            'all_directions':aggregate([r['all_directions'] for r in clips]),
            'contexts':{name:aggregate([r['contexts'][name] for r in clips if r['contexts'][name] is not None])
                        if any(r['contexts'][name] is not None for r in clips) else None
                        for name in ['quiet','talking','music']},'clips':clips}
