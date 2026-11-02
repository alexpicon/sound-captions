"""Verified local deployment profile, separate from immutable training/scoring logs."""
import hashlib
import json
import math
from pathlib import Path
from labeler import ROOT
from scoring import LABELS


def load_deployment(path=None,root=ROOT,verify_model=True):
    root=Path(root).resolve();config=json.loads((Path(path) if path else root/'live-config.json').read_text())
    if config.get('schema_version')!=1:raise ValueError('Unsupported live configuration')
    thresholds=config['thresholds']
    if set(thresholds)!=set(LABELS) or any(not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<t<=1 for t in thresholds.values()):
        raise ValueError('Invalid per-class thresholds')
    if thresholds['gunshot']!=.9:raise ValueError('Gunshot threshold must remain 0.9')
    if config['display_duration_s']!=3 or config['max_display_captions']!=2 or config['raw_scheduler_ttl_s']!=2:
        raise ValueError('Unexpected caption timing/capacity')
    model=(root/config['model']).resolve()
    if not model.is_relative_to(root):raise ValueError('Checkpoint must be local to this project')
    if verify_model and hashlib.sha256(model.read_bytes()).hexdigest()!=config['model_sha256']:
        raise ValueError('Active checkpoint changed; freeze a new deployment first')
    for name,digest in config['frozen_sources'].items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen recognition/scoring source changed: {name}')
    return config
