from __future__ import annotations

from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def evaluate(events: list[dict], reference: dict, cfg: dict, *, as_of: str) -> dict:
    mature = [event for event in events if event["horizon_end"] <= as_of]
    labeled = [event for event in mature if event.get("label") is not None
               and event.get("observed_at", "") <= as_of]
    sources = sorted({event.get("label_source", "unknown") for event in labeled})
    result = {"status": "awaiting_labels" if mature else "awaiting_outcome_horizon",
              "sample_count": len(events), "mature_samples": len(mature), "labeled_samples": len(labeled),
              "label_coverage": len(labeled) / len(mature) if mature else 0.0,
              "positive_samples": sum(int(event["label"]) for event in labeled),
              "negative_samples": sum(int(event["label"]) == 0 for event in labeled),
              "threshold": reference["decision_threshold"], "metrics": None,
              "baseline": reference["performance"], "trigger_candidate": False, "drops": {},
              "label_sources": sources, "simulated": "synthetic_stress" in sources,
              "churn_rate": sum(int(event["label"]) for event in labeled) / len(labeled) if labeled else None,
              "baseline_note": "Study holdout labels; synthetic outcomes are a different label source"}
    if not labeled:
        return result
    y = [int(event["label"]) for event in labeled]
    p = [event["probability"] for event in labeled]
    predicted = [value >= reference["decision_threshold"] for value in p]
    both = len(set(y)) == 2
    metrics = {"roc_auc": float(roc_auc_score(y, p)) if both else None,
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
               "log_loss": float(log_loss(y, p, labels=[0, 1])),
               "brier": float(brier_score_loss(y, p)),
               "predicted_positive_rate": sum(predicted) / len(predicted),
               "accuracy": float(accuracy_score(y, predicted)),
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


def measured_labels(y, probabilities, reference, cfg):
    """Synthetic labels have already passed policy/join validation; use the same metric convention."""
    import numpy as np
    y, probabilities = np.asarray(y), np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or probabilities.shape != y.shape or not np.isin(y, [0, 1]).all() or
            not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError("Outcome/probability alignment or values are invalid")
    events = [{"horizon_end": "2000-01-01T00:00:00+00:00", "observed_at": "2000-01-01T00:00:00+00:00",
               "label": int(label), "label_source": "synthetic_stress", "probability": float(p)}
              for label, p in zip(y, probabilities)]
    return evaluate(events, reference, cfg, as_of="2000-01-02T00:00:00+00:00")
