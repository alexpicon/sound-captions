"""Incremental round-2 inference; preserves the recorded window scheduler exactly."""
import copy
import hashlib
import json
import time
import numpy as np
from caption_policy import CaptionFilter,load_policy
from labeler import ROOT
from model_tools import estimate_direction
from scoring import LABELS
from training_core import YamnetEncoder


class WindowScheduler:
    def __init__(self,config):
        self.config=copy.deepcopy(config);self.active=[];self.previous=set()

    def step(self,now,direction,scores):
        c=self.config
        self.active=[r for r in self.active if r['start_s']+c['caption_ttl_s']>now]
        candidates=[];positive=set()
        for label,indices in c['class_mapping'].items():
            if not indices:continue
            score=float(np.max(scores[indices]))
            if score<c['thresholds'][label] or (c['hide_centered'] and direction=='center'):continue
            key=(label,direction);positive.add(key)
            if key not in self.previous:
                candidates.append({'start_s':round(now,6),'end_s':round(now+c['caption_ttl_s'],6),
                                   'label':label,'direction':direction,'score':score})
        result=[]
        for candidate in sorted(candidates,key=lambda r:(-r['score'],r['label'])):
            if len(self.active)<c['max_captions']:
                self.active.append(candidate);result.append(candidate)
        self.previous=positive
        return result


class LiveEngine:
    rate=16000;size=15600;hop=7680

    def __init__(self,show_centered=False):
        import joblib
        self.policy=load_policy()
        checkpoint=ROOT/'.local/training-runs/round-2/model.joblib'
        frozen=json.loads((ROOT/'training/policy-v2-round-2/protocol.json').read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=frozen['model_sha256']:
            raise ValueError('Round-2 checkpoint changed since policy replay')
        if hashlib.sha256((ROOT/'caption-policy-v2.json').read_bytes()).hexdigest()!=frozen['policy_sha256']:
            raise ValueError('Policy changed since replay; record a new policy version first')
        for name,key in [('caption_policy.py','filter_sha256'),('policy_scoring.py','scorer_sha256'),('scoring.py','legacy_scorer_sha256')]:
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=frozen[key]:
                raise ValueError(f'{name} changed since replay; record a new policy version first')
        saved=joblib.load(checkpoint)
        self.encoder=YamnetEncoder()
        if self.encoder.identity!=saved['encoder_sha256']:raise ValueError('YAMNet weights changed')
        if saved['threshold']!=self.policy['deployment_threshold']:raise ValueError('Unexpected checkpoint threshold')
        self.head=saved['head'];self.config=copy.deepcopy(saved['config'])
        self.config['class_mapping']={label:[i] for i,label in enumerate(LABELS)}
        self.config['thresholds']={label:saved['threshold'] for label in LABELS}
        self.config['hide_centered']=not show_centered
        self.compute_ms=[];self.reset()

    def reset(self):
        self.scheduler=WindowScheduler(self.config);self.filter=CaptionFilter(self.policy)

    def window(self,stereo,end_s):
        started=time.perf_counter()
        stereo=np.asarray(stereo,dtype=np.float32)
        if stereo.ndim!=2 or stereo.shape[1]!=2 or not 0<len(stereo)<=self.size:
            raise ValueError('Expected 1..15600 stereo samples at 16kHz')
        if not np.isfinite(stereo).all():raise ValueError('Non-finite audio')
        mono=np.pad(stereo.mean(axis=1),(0,self.size-len(stereo)))
        features,_=self.encoder.encode([mono]);probabilities=self.head.predict(features)[0]
        raw=self.scheduler.step(end_s,estimate_direction(stereo),probabilities)
        captions=[value for row in raw if (value:=self.filter.process(row)) is not None]
        elapsed=(time.perf_counter()-started)*1000;self.compute_ms.append(elapsed)
        return captions,elapsed
