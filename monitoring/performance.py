from __future__ import annotations

from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate(events: list[dict], reference: dict, cfg: dict, *, as_of: str) -> dict:
    mature = [event for event in events if event["horizon_end"] <= as_of]
    labeled = [event for event in mature if event.get("label") is not None
               and event.get("observed_at", "") <= as_of]
    result = {"status": "awaiting_labels" if mature else "awaiting_outcome_horizon",
              "sample_count": len(events), "mature_samples": len(mature), "labeled_samples": len(labeled),
              "label_coverage": len(labeled) / len(mature) if mature else 0.0,
              "positive_samples": sum(int(event["label"]) for event in labeled),
              "negative_samples": sum(int(event["label"]) == 0 for event in labeled),
              "threshold": reference["decision_threshold"], "metrics": None,
              "baseline": reference["performance"], "trigger_candidate": False, "drops": {}}
    if not labeled:
        return result
    y = [int(event["label"]) for event in labeled]
    p = [event["probability"] for event in labeled]
    predicted = [value >= reference["decision_threshold"] for value in p]
    both = len(set(y)) == 2
    metrics = {"roc_auc": float(roc_auc_score(y, p)) if both else None,
               "average_precision": float(average_precision_score(y, p)) if both else None,
               "precision": float(precision_score(y, predicted, zero_division=0)),
               "recall": float(recall_score(y, predicted, zero_division=0)),
               "f1": float(f1_score(y, predicted, zero_division=0)),
               "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}
    eligible = (len(labeled) >= cfg["min_labeled_samples"]
                and min(result["positive_samples"], result["negative_samples"]) >= cfg["min_class_samples"]
                and result["label_coverage"] >= cfg["min_label_coverage"])
    result["status"] = "ready" if eligible else "single_class" if not both else "insufficient_labels"
    result["metrics"] = metrics
    if eligible:
        result["drops"] = {"recall": reference["performance"]["Recall"] - metrics["recall"],
                           "average_precision": reference["performance"]["PR_AUC"] - metrics["average_precision"]}
        result["trigger_candidate"] = any(drop >= cfg["performance_drop"] for drop in result["drops"].values())
    return result
