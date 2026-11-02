"""Predeclared training rounds on development clips with nested whole-clip validation."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from labeler import Dataset, ROOT
from review import atomic_json
from run_baseline import captions_from_windows
from scoring import LABELS, aggregate, score_clip
from training_core import (ClassifierHead, EfficientATEncoder, YamnetEncoder,
                           development_clips, feature_record, training_ids)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learned_captions(data, probabilities, threshold, base_config):
    config = copy.deepcopy(base_config)
    config["class_mapping"] = {label: [i] for i, label in enumerate(LABELS)}
    config["thresholds"] = {label: threshold for label in LABELS}
    return captions_from_windows([{"time": float(t), "direction": str(d), "scores": p}
                                  for t, d, p in zip(data["times"], data["directions"], probabilities)], config)


def choose_threshold(records, features, predictions, config, grid):
    candidates = []
    for threshold in grid:
        metrics = aggregate([score_clip(r["truth"], learned_captions(features[r["id"]]["original"],
                                    predictions[r["id"]], threshold, config), r["duration"])
                             for r in records])
        candidates.append((threshold, metrics))
    feasible = [(t, m) for t, m in candidates if m["false_per_min"] <= 2]
    if feasible:
        selected = max(feasible, key=lambda pair: ((pair[1]["caught_pct"] or 0), -pair[1]["unmatched_per_min"], pair[0]))
    else:
        selected = max(candidates, key=lambda pair: (-pair[1]["false_per_min"], (pair[1]["caught_pct"] or 0), pair[0]))
    return selected[0], [{"threshold": t, "caught_pct": m["caught_pct"], "false_per_min": m["false_per_min"],
                         "unmatched_per_min": m["unmatched_per_min"]} for t, m in candidates]


def union_length(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def context_scores(record, captions):
    """Manual context only. Boundary convention: target and caption onset must be in slice."""
    from scoring import match_events, percent, summarize
    truth = record["truth"]
    sounds = [r for r in truth if r["label"] in LABELS]
    targets = [r for r in sounds if r["direction"] != "center"]
    pairs = match_events(targets, captions)
    matched_p = {j for _, j in pairs}
    intervals = {name: [(r["start_s"], r["end_s"]) for r in truth if r["label"] == name]
                 for name in ["talking", "music"]}
    def inside(t, name):
        if name == "quiet":
            return not any(a <= t < b for values in intervals.values() for a, b in values)
        return any(a <= t < b for a, b in intervals[name])
    results = {}
    for context in ["quiet", "talking", "music"]:
        duration = union_length(intervals[context]) if context != "quiet" else record["duration"] - union_length(intervals["talking"] + intervals["music"])
        if duration <= 0:
            results[context] = None
            continue
        selected = {i for i, r in enumerate(targets) if inside(r["start_s"], context)}
        matches = [(i, j) for i, j in pairs if i in selected]
        arrows = [(i, j) for i, j in matches if targets[i]["direction"] in {"left", "right"}
                  and captions[j]["direction"] in {"left", "right"}]
        pred_ids = [j for j, p in enumerate(captions) if inside(p["start_s"], context)]
        counts = {"duration_s": duration, "targets": len(selected), "caught": len(matches),
                  "predictions": len(pred_ids), "centered_targets_excluded": sum(r["direction"] == "center" and inside(r["start_s"], context) for r in sounds),
                  "false_captions": sum(not any(abs(s["start_s"] - captions[j]["start_s"]) <= 1 + 1e-9 for s in sounds) for j in pred_ids),
                  "unmatched_captions": sum(j not in matched_p for j in pred_ids),
                  "matched_directional_targets": sum(targets[i]["direction"] in {"left", "right"} for i, _ in matches),
                  "arrows_scored": len(arrows), "arrows_correct": sum(targets[i]["direction"] == captions[j]["direction"] for i, j in arrows),
                  "per_label": {label: {"targets": sum(targets[i]["label"] == label for i in selected),
                                         "caught": sum(targets[i]["label"] == label for i, _ in matches)} for label in LABELS},
                  "confusion": {label: {} for label in LABELS}}
        results[context] = summarize(counts)
    return results


def score_round(records, feature_data, predictions):
    individual = []
    for record in records:
        captions = predictions[record["id"]]
        individual.append({"clip_id": record["id"], "clip": str(record["path"].relative_to(ROOT)),
                           "primary": score_clip(record["truth"], captions, record["duration"]),
                           "all_directions": score_clip(record["truth"], captions, record["duration"], hide_centered=False),
                           "contexts": context_scores(record, captions)})
    return {"primary": aggregate([r["primary"] for r in individual]),
            "all_directions": aggregate([r["all_directions"] for r in individual]),
            "contexts": {name: aggregate([r["contexts"][name] for r in individual if r["contexts"][name] is not None])
                          if any(r["contexts"][name] is not None for r in individual) else None
                          for name in ["quiet", "talking", "music"]}, "clips": individual}


def fmt(value, percent=False):
    return "N/A" if value is None else f"{value:.1f}%" if percent else f"{value:.2f}"


def append_log(round_id, description, result, before, thresholds=None):
    m = result["primary"]
    lines = [f"\n## {'Development reference' if round_id == 0 else 'Training round ' + str(round_id)} — {description}\n",
             "**Validation only:** pooled outer leave-one-clip-out predictions on the eleven real Valorant development clips. No sealed test data opened. These are not directly comparable to the historical four-test-clip round 0.\n",
             "| Reference / round | Caught | False/min | Unmatched/min | Arrow correct | Arrow coverage | Quiet caught | Talking caught | Music caught |",
             "|---|---|---|---|---|---|---|---|---|"]
    if before:
        b = before["primary"]
        lines.append(f"| Pretrained development reference | {fmt(b['caught_pct'], True)} ({b['caught']}/{b['targets']}) | {fmt(b['false_per_min'])} | {fmt(b['unmatched_per_min'])} | {fmt(b['arrow_correct_pct'], True)} | {fmt(b['arrow_coverage_pct'], True)} | — | — | — |")
    contexts = [fmt(result["contexts"][name]["caught_pct"], True) if result["contexts"][name] else "N/A" for name in ["quiet", "talking", "music"]]
    lines.append(f"| {'Reference' if round_id == 0 else str(round_id)} | {fmt(m['caught_pct'], True)} ({m['caught']}/{m['targets']}) | {fmt(m['false_per_min'])} | {fmt(m['unmatched_per_min'])} | {fmt(m['arrow_correct_pct'], True)} ({m['arrows_correct']}/{m['arrows_scored']}) | {fmt(m['arrow_coverage_pct'], True)} | {' | '.join(contexts)} |")
    lines += ["\n| Label | Caught / target | Recall |", "|---|---|---|"]
    for label in LABELS:
        value = m["per_label"][label]
        lines.append(f"| {label} | {value['caught']}/{value['targets']} | {fmt(value['caught_pct'], True)} |")
    lines += ["\nConfusion counts (rows true, columns predicted; missed is no matched caption):\n",
              "| True | Footsteps | Gunshot | Reload | Explosion | Missed |", "|---|---|---|---|---|---|"]
    for label in LABELS:
        lines.append("| " + label + " | " + " | ".join(str(m["confusion"][label].get(k, 0)) for k in [*LABELS, "missed"]) + " |")
    a = result["all_directions"]
    lines.append(f"\nAll-direction recall, including intentionally hidden centered events: {a['caught']}/{a['targets']} ({fmt(a['caught_pct'], True)}). Primary scoring excluded {m['centered_targets_excluded']} centered targets.")
    if thresholds:
        lines.append("\nEach outer fold selected its threshold using only inner validation on the other ten clips. Thresholds: " + ", ".join(f"{v:.1f}" for v in thresholds) + ".")
    lines.append(f"\nMachine-readable metrics: `training/round-{round_id}/metrics.json`. No test-based tuning.\n")
    with (ROOT / "rounds.md").open("a") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round", type=int, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    plan = json.loads((ROOT / "training-plan.json").read_text())
    config = json.loads((ROOT / "baseline-config.json").read_text())
    out = ROOT / "training" / f"round-{args.round}"
    if (out / "metrics.json").exists() or (out / "protocol.json").exists():
        raise SystemExit("This round is already recorded. Do not overwrite its protocol/results.")
    records = development_clips(Dataset(ROOT))
    if len(records) != 11:
        raise ValueError("Expected all eleven reviewed development clips")
    spec = plan["rounds"][args.round - 1] if args.round else {"encoder": "yamnet", "inputs": "mono", "paired": False}
    protocol = {"created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                "round": args.round, "spec": spec, "plan_sha256": sha(ROOT / "training-plan.json"),
                "runner_sha256": sha(Path(__file__)), "core_sha256": sha(ROOT / "training_core.py"),
                "scorer_sha256": sha(ROOT / "scoring.py"),
                "development_labels": {r["id"]: r["csv_revision"] for r in records}, "test_data_opened": False,
                "efficientat_source_commit": subprocess.check_output(["git", "-C", str(ROOT / ".local/vendor/EfficientAT"), "rev-parse", "HEAD"], text=True).strip()}
    atomic_json(out / "protocol.json", protocol)
    encoder = YamnetEncoder() if spec["encoder"] == "yamnet" else EfficientATEncoder()
    feature_data = {}
    for record in records:
        variants = ["original", "voice_removed"] if spec["paired"] else ["original"]
        feature_data[record["id"]] = {variant: feature_record(record, variant, encoder, spec["inputs"] == "mono+side") for variant in variants}
        print(f"Features ready: {record['path'].name}", flush=True)
    model_cache = {}
    all_ids = [r["id"] for r in records]
    by_id = {r["id"]: r for r in records}
    def fit(ids):
        key = tuple(sorted(ids))
        if key not in model_cache:
            x = np.concatenate([v["features"] for k in key for v in feature_data[k].values()])
            y = np.concatenate([v["targets"] for k in key for v in feature_data[k].values()])
            model_cache[key] = ClassifierHead().fit(x, y)
        return model_cache[key]
    captions, fold_logs, outer_probabilities = {}, [], {}
    if args.round == 0:
        for record in records:
            data = feature_data[record["id"]]["original"]
            captions[record["id"]] = captions_from_windows([{"time": float(t), "direction": str(d), "scores": p}
                for t, d, p in zip(data["times"], data["directions"], data["pretrained_scores"])], config)
    else:
        for outer_id in all_ids:
            train = training_ids(all_ids, [outer_id])
            inner_predictions = {}
            for inner_id in train:
                inner_train = training_ids(all_ids, [outer_id, inner_id])
                assert outer_id not in inner_train and inner_id not in inner_train
                inner_predictions[inner_id] = fit(inner_train).predict(feature_data[inner_id]["original"]["features"])
            threshold, search = choose_threshold([by_id[k] for k in train], feature_data, inner_predictions, config, plan["threshold_grid"])
            probabilities = fit(train).predict(feature_data[outer_id]["original"]["features"])
            outer_probabilities[outer_id] = probabilities
            captions[outer_id] = learned_captions(feature_data[outer_id]["original"], probabilities, threshold, config)
            fold_logs.append({"outer_validation_id": outer_id, "train_ids": train, "threshold": threshold,
                              "inner_excluded_ids": [outer_id], "inner_search": search})
            print(f"Outer fold {len(fold_logs)}/11 complete; threshold calibrated without this clip.", flush=True)
        # Deployment calibration uses out-of-fold probabilities, never sealed data.
        deployment_threshold, deployment_search = choose_threshold(records, feature_data, outer_probabilities, config, plan["threshold_grid"])
        local = ROOT / ".local/training-runs" / f"round-{args.round}"
        local.mkdir(parents=True, exist_ok=True)
        joblib.dump({"head": fit(all_ids), "threshold": deployment_threshold, "spec": spec,
                     "labels": LABELS, "encoder_sha256": encoder.identity, "config": config}, local / "model.joblib")
        atomic_json(out / "deployment.json", {"threshold": deployment_threshold, "selection": "Development out-of-fold calibration only", "search": deployment_search, "model": str((local / "model.joblib").relative_to(ROOT))})
    result = score_round(records, feature_data, captions)
    result.update({"round": args.round, "spec": spec, "folds": fold_logs, "test_data_opened": False,
                   "encoder_sha256": encoder.identity,
                   "encoding_realtime_factor": sum(float(v["original"]["encoder_seconds"]) for v in feature_data.values()) / sum(r["duration"] for r in records)})
    atomic_json(out / "metrics.json", result)
    atomic_json(ROOT / ".local/training-runs" / f"round-{args.round}" / "outer-captions.json", captions)
    before = json.loads((ROOT / "training/round-0/metrics.json").read_text()) if args.round else None
    description = "Plain pretrained YAMNet on development clips" if not args.round else spec["purpose"]
    append_log(args.round, description, result, before, [f["threshold"] for f in fold_logs])
    m = result["primary"]
    print(json.dumps({"round": args.round, "caught_pct": m["caught_pct"], "caught": m["caught"],
                      "targets": m["targets"], "false_per_min": m["false_per_min"],
                      "unmatched_per_min": m["unmatched_per_min"], "arrow_correct_pct": m["arrow_correct_pct"]}), flush=True)


if __name__ == "__main__":
    main()
