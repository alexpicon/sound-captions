"""Versioned caption postprocessing shared by dev replay, live helper and final scoring."""
import json
from pathlib import Path

POLICY_PATH = Path(__file__).with_name('caption-policy-v2.json')


def load_policy():
    return json.loads(POLICY_PATH.read_text())


class CaptionFilter:
    """Merge chains of same-label/side candidates with less than a 2s quiet gap."""
    def __init__(self, policy=None):
        self.policy = policy or load_policy()
        self.last_seen = {}

    def process(self, caption):
        if caption['label'] not in self.policy['enabled_labels']:
            return None
        key = (caption['label'], caption['direction'])
        now = caption['start_s']
        previous = self.last_seen.get(key)
        self.last_seen[key] = now
        if previous is not None and now - previous < self.policy['repeat_gap_s'] - 1e-9:
            return None
        return dict(caption)


def apply_policy(captions, policy=None):
    merger = CaptionFilter(policy)
    result = []
    for row in sorted(captions, key=lambda r: r['start_s']):
        value = merger.process(row)
        if value is not None:
            result.append(value)
    return result


def supported_by_span(caption, truth, policy=None):
    policy = policy or load_policy()
    t = caption['start_s']
    return any(row['label'] == caption['label']
               and row['start_s'] - policy['false_early_tolerance_s'] - 1e-9 <= t
               <= max(row['end_s'], row['start_s'] + policy['false_after_onset_s']) + 1e-9
               for row in truth if row['label'] in policy['enabled_labels'])
