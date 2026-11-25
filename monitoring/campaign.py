"""Preflight a fair five-model campaign without opening or mutating a database."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from api import module5, segmentation_service as seg
from api.challenger_models import load as load_challenger
from api.scoring import score_probabilities
from demo_data.generate import BATCHES
from demo_data.labels import read_labels
from demo_data.validate import check_dataset, model_paths, read_outputs, segmentation_frame
from monitoring import clustering_metrics as cm, drift, performance
from monitoring.settings import ROOT, digest, now_utc, timestamp

ALGORITHMS = ("xgboost", "lr", "rf", "kmeans", "hierarchical")
VERSION = "five_model_campaign_v1"


def config():
    value = yaml.safe_load((ROOT / "config/multi_model_monitoring.yaml").read_text(encoding="utf-8"))
    if value["version"] != 1 or value["silhouette_sample_size"] < 2 or value["min_cluster_profile_samples"] < 1:
        raise ValueError("Invalid multi-model monitoring configuration")
    for name in ("silhouette_drop", "dbi_relative_increase", "p95_relative_increase", "outlier_share_increase"):
        if not 0 < value[name] <= 1:
            raise ValueError("Invalid clustering review threshold")
    return value


@dataclass
class PreparedCampaign:
    manifest: dict
    frame: pd.DataFrame
    labels: object
    models: dict
    probabilities: dict
    class_baselines: dict
    input_reports: dict
    cluster_reports: dict
    assignments: dict
    distances: dict
    margins: dict


def prepare(service, *, data_dir, labels_path=None, anchor=None):
    if service.model is None or service.segmentation is None or service.reference is None:
        raise ValueError("Prepared champion, frozen segmentation and monitoring reference required")
    model, bundle, cfg = service.model, service.segmentation, service.config
    experiment = config()
    manifest, frame, examples = read_outputs(Path(data_dir))
    if (Path(data_dir) / ".publishing").exists() or manifest["inference"]["status"] != "passed":
        raise ValueError("Demo publication/validation is incomplete")
    if manifest["input_columns"] != model.input_columns or manifest["input_dtypes"] != model.manifest["input_dtypes"]:
        raise ValueError("Campaign classifier input schema mismatch")
    for name, path in model_paths().items():
        if not path.is_file() or module5.file_hash(path) != manifest["models"][name]["sha256"]:
            raise ValueError(f"Campaign artifact mismatch: {name}")
    source_ids = set(manifest["source_customer_id_by_demo_id"].values())
    if len(source_ids) != manifest["N_base"]:
        source_ids = pd.read_csv(module5.DATA_PATH, usecols=["customer_id"]).customer_id
    check_dataset(frame, examples, manifest, source_ids)
    frame = frame.sort_values("customer_id").reset_index(drop=True)
    if set(frame.batch_id) != set(BATCHES) or len(frame) > 10000:
        raise ValueError("Campaign requires all five nonempty batches, capped at 10,000 rows")
    data_hash = manifest["output_sha256"]["customer_features.csv"]
    labels, label_meta = read_labels(Path(labels_path), frame, data_hash) if labels_path is not None else (None, None)
    y = labels.loc[frame.customer_id, "churn"].to_numpy() if labels is not None else None
    first = datetime.fromisoformat(timestamp(anchor or model.manifest["created_at"])).replace(microsecond=0)
    if anchor is None:
        first = first.replace(hour=0, minute=0, second=0) - timedelta(days=cfg["outcome_horizon_days"] + 7)
    if cfg["window_hours"] != 24 or first.timestamp() % 86400:
        raise ValueError("Campaigns require 24-hour windows and a UTC midnight anchor")
    if first < now_utc() - timedelta(days=cfg["retention_days"]) or first + timedelta(days=5) > now_utc():
        raise ValueError("Campaign anchor must be closed and within retention")
    if labels is not None and first + timedelta(days=5 + cfg["outcome_horizon_days"]) > now_utc():
        raise ValueError("Synthetic labels require mature outcome horizons")
    masks = {name: frame.batch_id.eq(name).to_numpy() for name in BATCHES}
    normal = np.logical_or.reduce([masks[name] for name in BATCHES[:3]])
    encoded = model.transform(frame[model.input_columns])
    models, probabilities, class_baselines = {}, {}, {}
    common_hash = digest([model.manifest["preprocessor_sha256"], service.reference["features"],
                          service.reference["training_subjects_sha256"]])
    for algorithm in ALGORITHMS[:3]:
        if algorithm == "xgboost":
            p = score_probabilities(model, frame[model.input_columns])
            ref = service.reference
            descriptor = {"model_id": service.model_id, "algorithm": algorithm, "task": "classification", "role": "champion",
                          "study_reproduced": model.manifest["study_reproduced"],
                          "checkpoint_sha256": model.manifest["checkpoint_sha256"], "holdout_metrics": ref["performance"]}
        else:
            adapter, validated = load_challenger(algorithm, model)
            p, ref = adapter.predict_encoded(encoded), validated["reference"]
            descriptor = {key: validated[key] for key in ("model_id", "algorithm", "role", "study_reproduced", "checkpoint_sha256",
                                                          "holdout_metrics", "comparisons", "compatibility_warnings")}
            descriptor["task"] = "classification"
            del adapter
        if digest(ref["features"]) != digest(service.reference["features"]) or ref["decision_threshold"] != service.reference["decision_threshold"]:
            raise ValueError("Classifier feature space/threshold parity failed")
        descriptor.update(reference=ref, reference_sha256=digest(ref), feature_space_id=common_hash)
        models[algorithm], probabilities[algorithm] = descriptor, p
        measured = performance.measured_labels(y[normal], p[normal], ref, cfg) if y is not None else None
        class_baselines[algorithm] = ({"batches": list(BATCHES[:3]), "sample_count": int(normal.sum()), "metrics": measured["metrics"]}
                                      if measured and measured["status"] == "ready" else None)

    classifier_input = model.profiles(frame[model.input_columns])
    input_reports = {"classification": {name: drift.evaluate(classifier_input.loc[mask], service.reference, cfg) for name, mask in masks.items()}}
    current = seg.feature_frame(segmentation_frame(frame))
    official = seg.feature_frame(pd.read_csv(seg.DATA_PATH))
    z, ref_z = bundle["preprocessor"].transform(current), bundle["preprocessor"].transform(official)
    cluster_cfg = {**cfg, **experiment,
                   "numeric_features": [name for name in current if pd.api.types.is_numeric_dtype(current[name])],
                   "categorical_features": [name for name in current if not pd.api.types.is_numeric_dtype(current[name])]}
    cluster_cfg["critical_features"] = [name for name in cfg["critical_features"] if name in current]
    cluster_input = cm.input_reference(official, cfg)
    cluster_input_hash = digest(cluster_input)
    cluster_space = digest(["segmentation_input_v1", manifest["models"]["clustering_bundle"]["sha256"], cluster_input_hash])
    input_reports["clustering"] = {name: drift.evaluate(current.loc[mask], {"features": cluster_input}, cluster_cfg) for name, mask in masks.items()}
    samples = {name: np.sort(np.random.default_rng(experiment["sampling_seed"]).choice(int(mask.sum()),
                 size=min(int(mask.sum()), experiment["silhouette_sample_size"]), replace=False)) for name, mask in masks.items()}
    cluster_reports, assignments, distances, margins = {}, {}, {}, {}
    for algorithm in ALGORITHMS[3:]:
        centers = bundle["kmeans"].cluster_centers_ if algorithm == "kmeans" else bundle["hierarchical_centers"]
        assigned = bundle["kmeans"].predict(z) if algorithm == "kmeans" else np.square(z[:, None, :] - centers).sum(axis=2).argmin(axis=1)
        reference_labels = bundle["kmeans"].predict(ref_z) if algorithm == "kmeans" else np.square(ref_z[:, None, :] - centers).sum(axis=2).argmin(axis=1)
        # Reference and current both use predict / nearest fixed Ward centroid; never mix native Ward labels.
        ref = cm.frozen_reference(official, ref_z, reference_labels, centers, cluster_cfg)
        method = "kmeans_predict" if algorithm == "kmeans" else "ward_nearest_centroid_surrogate"
        cluster_id = digest(["cluster_adapter_v1", algorithm, method, manifest["models"]["clustering_bundle"]["sha256"],
                             centers.tolist(), digest(ref)])
        models[algorithm] = {"model_id": cluster_id, "algorithm": algorithm, "task": "clustering", "role": "segmentation",
                             "assignment_method": method, "reference": ref, "reference_sha256": digest(ref),
                             "feature_space_id": cluster_space, "input_reference_sha256": cluster_input_hash}
        group = {}
        for name, mask in masks.items():
            quality = cm.evaluate(z[mask], assigned[mask], centers, ref, cfg, indices=samples[name])
            quality["profiles"] = cm.profile_drift(current.loc[mask].reset_index(drop=True), assigned[mask], ref, cluster_cfg)
            group[name] = quality
        baseline = {key: float(np.median([group[name][key] for name in BATCHES[:3] if group[name][key] is not None]))
                    if any(group[name][key] is not None for name in BATCHES[:3]) else None
                    for key in ("silhouette", "dbi", "chi", "p95_distance", "outlier_share")}
        cluster_reports[algorithm] = {name: cm.compare_baseline(row, baseline, experiment, building=name in BATCHES[:3])
                                       for name, row in group.items()}
        assignments[algorithm] = assigned
        distances[algorithm] = np.linalg.norm(z - centers[assigned], axis=1)
        ordered = np.sort(np.linalg.norm(z[:, None, :] - centers, axis=2), axis=1)
        margins[algorithm] = ordered[:, 1] - ordered[:, 0]
    identity = {"version": VERSION, "schema_version": 1, "dataset_sha256": data_hash,
                "generation_seed": manifest["seed"], "labels_sha256": label_meta["output_sha256"] if label_meta else None,
                "label_policy_version": label_meta["label_policy_version"] if label_meta else None,
                "label_policy_sha256": label_meta["policy_sha256"] if label_meta else None,
                "label_seed": label_meta["label_seed"] if label_meta else None,
                "feature_snapshot": manifest["snapshot"], "anchor": timestamp(first), "config": {**cfg, **experiment},
                "mode": "simulated_labels" if labels is not None else "drift_only",
                "models": {name: {key: value for key, value in item.items() if key != "reference"} for name, item in models.items()},
                "batches": [{"batch_id": name, "index": i + 1, "N": int(mask.sum()),
                             "cohort_sha256": digest(sorted(frame.loc[mask, "customer_id"].astype(str).tolist())),
                             "window_start": timestamp(first + timedelta(days=i)), "window_end": timestamp(first + timedelta(days=i + 1))}
                            for i, (name, mask) in enumerate(masks.items())]}
    identity["campaign_id"] = digest(identity)
    return PreparedCampaign(identity, frame, labels, models, probabilities, class_baselines, input_reports,
                            cluster_reports, assignments, distances, margins)
