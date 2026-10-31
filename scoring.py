"""Reusable event scorer. Pure standard library; no inference or model fitting."""
from collections import Counter

LABELS = ("footsteps", "gunshot", "reload", "explosion")


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def match_events(truth, predictions, tolerance=1.0):
    """Maximum-cardinality matching, never credit one caption to two targets."""
    assigned = {}
    edges = {i: sorted((j for j, prediction in enumerate(predictions)
                        if event["label"] == prediction["label"]
                        and abs(event["start_s"] - prediction["start_s"]) <= tolerance + 1e-9),
                       key=lambda j: (abs(event["start_s"] - predictions[j]["start_s"]), j))
             for i, event in enumerate(truth)}

    def visit(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or visit(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(truth)):
        visit(i, set())
    return sorted((i, j) for j, i in assigned.items())


def summarize(counts):
    minutes = counts["duration_s"] / 60
    return {**counts, "caught_pct": percent(counts["caught"], counts["targets"]),
            "false_per_min": counts["false_captions"] / minutes if minutes else None,
            "unmatched_per_min": counts["unmatched_captions"] / minutes if minutes else None,
            "arrow_correct_pct": percent(counts["arrows_correct"], counts["arrows_scored"]),
            "arrow_coverage_pct": percent(counts["arrows_scored"], counts["matched_directional_targets"]),
            "per_label": {label: {**value, "caught_pct": percent(value["caught"], value["targets"])}
                          for label, value in counts["per_label"].items()}}


def score_clip(truth, predictions, duration, tolerance=1.0, hide_centered=True):
    sounds = [row for row in truth if row["label"] in LABELS]
    targets = [row for row in sounds if not hide_centered or row["direction"] != "center"]
    matches = match_events(targets, predictions, tolerance)
    matched_truth = {i for i, _ in matches}
    matched_predictions = {j for _, j in matches}
    per_label = {label: {"targets": sum(row["label"] == label for row in targets),
                          "caught": sum(targets[i]["label"] == label for i, _ in matches)} for label in LABELS}
    confusion = {label: Counter() for label in LABELS}
    for i, j in matches:
        confusion[targets[i]["label"]][predictions[j]["label"]] += 1
    unmatched = set(range(len(predictions))) - matched_predictions
    # Pair remaining wrong labels with missed targets only for a descriptive table.
    for i, event in enumerate(targets):
        if i in matched_truth:
            continue
        nearby = [j for j in unmatched if abs(predictions[j]["start_s"] - event["start_s"]) <= tolerance + 1e-9
                  and predictions[j]["label"] != event["label"]]
        if nearby:
            j = min(nearby, key=lambda j: (abs(predictions[j]["start_s"] - event["start_s"]), j))
            confusion[event["label"]][predictions[j]["label"]] += 1
            unmatched.remove(j)
        else:
            confusion[event["label"]]["missed"] += 1
    directional = [(i, j) for i, j in matches if targets[i]["direction"] in {"left", "right"}]
    arrows = [(i, j) for i, j in directional if predictions[j]["direction"] in {"left", "right"}]
    return summarize({"duration_s": duration, "targets": len(targets), "caught": len(matches),
                      "predictions": len(predictions), "centered_targets_excluded": len(sounds) - len(targets),
                      "false_captions": sum(not any(abs(row["start_s"] - p["start_s"]) <= tolerance + 1e-9
                                                     for row in sounds) for p in predictions),
                      "unmatched_captions": len(predictions) - len(matches),
                      "matched_directional_targets": len(directional), "arrows_scored": len(arrows),
                      "arrows_correct": sum(targets[i]["direction"] == predictions[j]["direction"] for i, j in arrows),
                      "per_label": per_label, "confusion": {k: dict(v) for k, v in confusion.items()}})


def aggregate(results):
    keys = ("duration_s", "targets", "caught", "predictions", "centered_targets_excluded", "false_captions",
            "unmatched_captions", "matched_directional_targets", "arrows_scored", "arrows_correct")
    counts = {key: sum(result[key] for result in results) for key in keys}
    counts["per_label"] = {label: {key: sum(r["per_label"][label][key] for r in results)
                                    for key in ("targets", "caught")} for label in LABELS}
    counts["confusion"] = {}
    for label in LABELS:
        counter = Counter()
        for result in results:
            counter.update(result["confusion"][label])
        counts["confusion"][label] = dict(counter)
    return summarize(counts)
