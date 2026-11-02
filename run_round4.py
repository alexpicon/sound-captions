"""One authorized fold-local augmentation round; strict inner threshold feasibility."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np

from cutpaste import FoldAugmenter
from labeler import Dataset, ROOT
from review import atomic_json
from run_training import learned_captions, score_round, append_log
from scoring import aggregate, score_clip
from training_core import ClassifierHead, YamnetEncoder, development_clips, feature_record, training_ids


def strict_choice(candidates):
    feasible = [item for item in candidates if item["false_per_min"] <= 2 + 1e-12]
    if not feasible:
        raise ValueError("No feasible threshold. Do not relax the false-caption limit.")
    return max(feasible, key=lambda m: ((m["caught_pct"] or 0), -m["unmatched_per_min"], m["threshold"]))


def select_threshold(records, features, probabilities, config, grid):
    candidates = []
    for threshold in grid:
        score = aggregate([score_clip(record["truth"], learned_captions(features[record["id"]]["original"],
                            probabilities[record["id"]], threshold, config), record["duration"])
                           for record in records])
        candidates.append({"threshold": threshold, "caught_pct": score["caught_pct"],
                           "false_per_min": score["false_per_min"], "false_captions": score["false_captions"],
                           "duration_s": score["duration_s"], "unmatched_per_min": score["unmatched_per_min"]})
    return strict_choice(candidates)["threshold"], candidates


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "training/round-4"
    if (out / "protocol.json").exists():
        raise SystemExit("Round 4 is already started/recorded; do not overwrite it.")
    plan = json.loads((ROOT / "round-4-plan.json").read_text())
    config = json.loads((ROOT / "baseline-config.json").read_text())
    records = development_clips(Dataset(ROOT))
    if len(records) != 11:
        raise ValueError("Expected eleven completed real development clips")
    atomic_json(out / "protocol.json", {"created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "plan": plan, "plan_sha256": digest(ROOT / "round-4-plan.json"),
        "runner_sha256": digest(Path(__file__)), "augmentation_sha256": digest(ROOT / "cutpaste.py"),
        "scorer_sha256": digest(ROOT / "scoring.py"), "head_code_sha256": digest(ROOT / "training_core.py"),
        "annotations": {r["id"]: r["csv_revision"] for r in records}, "test_data_opened": False})
    encoder = YamnetEncoder()
    features = {r["id"]: {variant: feature_record(r, variant, encoder) for variant in ["original", "voice_removed"]} for r in records}
    augmenter = FoldAugmenter(records, encoder, plan)
    ids = sorted(features)
    by_id = {r["id"]: r for r in records}
    cache, audits = {}, {}
    local = ROOT / ".local/training-runs/round-4"
    def fit(allowed):
        key = tuple(sorted(allowed))
        if key not in cache:
            synthetic_x, synthetic_y, provenance = augmenter.for_fold(key)
            assert all(p["donor"] in key and p["recipient"] in key for p in provenance)
            x = np.concatenate([*[v["features"] for k in key for v in features[k].values()], synthetic_x])
            y = np.concatenate([*[v["targets"] for k in key for v in features[k].values()], synthetic_y])
            cache[key] = ClassifierHead().fit(x, y)
            tag = hashlib.sha256(",".join(key).encode()).hexdigest()[:16]
            audits[tag] = {"train_ids": list(key), "synthetic_mixes": len(provenance),
                          "synthetic_windows_including_controls": len(synthetic_x), "total_training_windows": len(x),
                          "donor_recipient_pairs": sorted({(p["donor"], p["recipient"]) for p in provenance})}
            atomic_json(local / "fold-audit.json", audits)
        return cache[key]
    captions, outer_probabilities, folds = {}, {}, []
    for outer_id in ids:
        allowed = training_ids(ids, [outer_id])
        inner = {}
        for inner_id in allowed:
            inner_allowed = training_ids(ids, [outer_id, inner_id])
            assert outer_id not in inner_allowed and inner_id not in inner_allowed
            inner[inner_id] = fit(inner_allowed).predict(features[inner_id]["original"]["features"])
        threshold, search = select_threshold([by_id[k] for k in allowed], features, inner, config, plan["threshold_grid"])
        outer_probabilities[outer_id] = fit(allowed).predict(features[outer_id]["original"]["features"])
        captions[outer_id] = learned_captions(features[outer_id]["original"], outer_probabilities[outer_id], threshold, config)
        selection = next(item for item in search if item["threshold"] == threshold)
        assert selection["false_per_min"] <= 2 + 1e-12
        folds.append({"outer_validation_id": outer_id, "train_ids": allowed, "threshold": threshold,
                      "selected_inner_metrics": selection, "inner_search": search})
        atomic_json(local / "fold-progress.json", folds)
        print(f"Outer fold {len(folds)}/11: inner false/min={selection['false_per_min']:.2f}, threshold={threshold:.6f}", flush=True)
    result = score_round(records, features, captions)
    result.update({"round": 4, "description": "Fold-local cut/paste augmentation, gains and L/R panning; strict inner false-caption limit",
                   "folds": folds, "test_data_opened": False})
    m = result["primary"]
    result["step4_gate"] = {"recall_required_pct": 35, "false_per_min_required_max": 2,
        "passed": m["caught_pct"] is not None and m["caught_pct"] >= 35 and m["false_per_min"] <= 2}
    atomic_json(out / "metrics.json", result)
    atomic_json(local / "outer-captions.json", captions)
    np.savez_compressed(local / "outer-probabilities.npz", **outer_probabilities)
    threshold, search = select_threshold(records, features, outer_probabilities, config, plan["threshold_grid"])
    # This checkpoint is not a new validation score; the gate uses nested outer results above.
    head = fit(ids)
    joblib.dump({"head": head, "threshold": threshold,
                 "spec": {"encoder": "yamnet", "inputs": "mono", "paired": True, "cutpaste": True},
                 "labels": ("footsteps", "gunshot", "reload", "explosion"),
                 "encoder_sha256": encoder.identity, "config": config}, local / "model.joblib")
    atomic_json(out / "deployment.json", {"threshold": threshold, "selection": "Strict constrained development out-of-fold calibration only",
        "search": search, "model": str((local / "model.joblib").relative_to(ROOT)),
        "ready_for_step4": result["step4_gate"]["passed"]})
    atomic_json(out / "augmentation-audit.json", {"folds": audits, "test_data_opened": False,
        "checks": "Every cache access and waveform access checks donor AND recipient training membership first."})
    reference = json.loads((ROOT / "training/round-0/metrics.json").read_text())
    append_log(4, result["description"], result, reference)
    with (ROOT / "rounds.md").open("a") as handle:
        handle.write("\nRound 4 strictly enforces the 2/min limit on **inner calibration**; it never falls back to an infeasible threshold. The threshold grid extends above 0.9 up to an explicit abstain-all setting. Held-out outer performance, rather than the calibration score, determines the user's 35% / 2-per-minute step-4 gate.\n")
        handle.write("\nExact outer-fold thresholds (selected on inner folds): " + ", ".join(f"{f['threshold']:.6f}" for f in folds) + ".\n")
        handle.write(f"\nStep-4 gate: **{'PASSED' if result['step4_gate']['passed'] else 'NOT PASSED'}**. Nested real-clip validation: {m['caught_pct']:.1f}% recall, {m['false_per_min']:.2f} literal false/min, {m['unmatched_per_min']:.2f} unmatched/min. Sealed tests remain closed.\n")
    print(json.dumps({"recall_pct": m["caught_pct"], "false_per_min": m["false_per_min"],
                      "unmatched_per_min": m["unmatched_per_min"], "step4_gate": result["step4_gate"]}), flush=True)


if __name__ == "__main__":
    main()
