#!/usr/bin/env python3
"""Install the bundled, hash-verified selected head into its frozen local path."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    config = json.loads((ROOT / 'live-config.json').read_text())
    source = ROOT / 'models/round-2.joblib'
    encoder = ROOT / 'test-audio/model/yamnet.tflite'
    for path, expected in [(source, config['model_sha256']),
                           (encoder, config['encoder_sha256'])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Checksum mismatch: {path.relative_to(ROOT)}')
    target = (ROOT / config['model']).resolve()
    if not target.is_relative_to(ROOT):
        raise SystemExit('Model destination must stay within the repository')
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != config['model_sha256']:
        raise SystemExit('Existing local checkpoint differs; refusing to overwrite it')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    print('Selected round-2 classifier and YAMNet encoder verified and ready.')

if __name__ == '__main__':
    main()
