"""Verify demo outputs, provenance and parity with the actual FastAPI routes."""
from __future__ import annotations

import argparse
import json
import math
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import pandas as pd

from api import module5, segmentation_service
from api.scoring import score_probabilities
from demo_data.generate import (
    AUDIT_COLUMNS,
    BATCHES,
    META,
    OUTPUT,
    SEG_BEHAVIOR,
    SNAPSHOT_DATE,
    SOURCE,
    aggregate,
    batch_sizes,
    load_sources,
    sample_count,
    summarize,
)
from segmentation.features import ORIGINAL_FEATURES

SEGMENTATION_DATA_PATH = Path(os.getenv("SEGMENTATION_CUSTOMERS_CSV_PATH", "data/official/customer_features.csv"))

NULLABLE = {
    "avg_resolution_delay_days", "initiated_failure_rate", "login_failure_rate",
}


def model_paths():
    return {
        "classifier_checkpoint": module5.CHECKPOINT_PATH,
        "classifier_preprocessor": module5.PREPROCESSOR_PATH,
        "classifier_manifest": module5.MANIFEST_PATH,
        "classifier_official_snapshot": module5.DATA_PATH,
        "clustering_bundle": segmentation_service.MODEL_DIR / "cluster_bundle.joblib",
        "clustering_metadata": segmentation_service.MODEL_DIR / "metadata.json",
        "clustering_official_snapshot": SEGMENTATION_DATA_PATH,
    }


def json_scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def request_for(record, input_columns, *, segmentation=False):
    from api.main import CustomerInput
    profile = {name: json_scalar(record[name]) for name in module5.PROFILE_COLUMNS}
    if segmentation:
        profile.update({name: json_scalar(record[f"seg_{name}"]) for name in SEG_BEHAVIOR})
    else:
        profile["raw_features"] = {name: json_scalar(record[name]) for name in input_columns}
    return CustomerInput.model_validate(profile).model_dump(exclude_unset=True)


def build_examples(frame, input_columns):
    chosen = frame.groupby("batch_id", sort=False).head(10)
    return [request_for(record, input_columns) for record in chosen.to_dict("records")]


def segmentation_frame(frame):
    result = frame[list(segmentation_service.FEATURES)].copy()
    result[list(SEG_BEHAVIOR)] = frame[[f"seg_{name}" for name in SEG_BEHAVIOR]].to_numpy()
    return result


def check_novelty(frame, reference, input_columns):
    """Look for copied feature vectors independently of customer IDs and batch annotations."""
    def matching_rows(left, right, columns):
        # Compare at eight decimal places to ignore float serialization noise.
        keys = []
        for table in (left, right):
            selected = table.loc[:, columns]
            numeric = selected.select_dtypes(include="number").columns
            keys.append(selected.astype(dict.fromkeys(numeric, float)).round(8))
        return len(keys[0].merge(keys[1].drop_duplicates(), on=columns, how="inner"))

    official = (pd.read_csv(module5.DATA_PATH, usecols=lambda name: name in input_columns)
                if module5.DATA_PATH.is_file() else pd.DataFrame())
    report = {
        "matches_source_profiles": matching_rows(frame, reference, ORIGINAL_FEATURES),
        "matches_source_classifier_inputs": matching_rows(frame, reference, input_columns),
        "matches_official_classifier_inputs": (
            matching_rows(frame, official, input_columns) if set(official) == set(input_columns) else None),
    }
    if any(report.values()):
        raise ValueError(f"Copied source feature vectors found: {report}")
    return report


def check_dataset(frame, examples, manifest, source_ids):
    from api.main import BATCH_CUSTOMER_ADAPTER
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    source_ids = set(source_ids)
    columns = manifest["input_columns"]
    dtypes = manifest["input_dtypes"]
    require(set(dtypes) == set(columns) and len(columns) == len(set(columns)), "Invalid classifier schema")
    required = set(META) | set(columns) | set(module5.PROFILE_COLUMNS) | set(AUDIT_COLUMNS) | {
        f"seg_{name}" for name in SEG_BEHAVIOR
    }
    require(set(frame) == required and len(frame.columns) == len(required), "Unexpected feature CSV columns")
    expected = sample_count(len(source_ids), manifest["multiplier"], manifest["max_samples"])
    require(len(source_ids) == manifest["N_base"] and len(frame) == manifest["N_demo"] == expected,
            "Demo/source sample count mismatch")
    sizes = batch_sizes(expected)
    actual = {key: int(frame.batch_id.eq(key).sum()) for key in BATCHES}
    require(sizes == actual == manifest["batch_rows"] and sum(actual.values()) == len(frame),
            "Batch size or assignment mismatch")
    scenario = frame.batch_id.where(~frame.batch_id.str.startswith("normal_"), "normal")
    require(frame.scenario.eq(scenario).all() and frame.feature_snapshot.eq(SNAPSHOT_DATE).all()
            and manifest["snapshot"] == SNAPSHOT_DATE, "Invalid snapshot/scenario")
    ids = frame.customer_id
    require(pd.api.types.is_integer_dtype(ids) and ids.between(1, 2**53 - 1).all()
            and ids.is_unique and not set(ids).intersection(source_ids), "Invalid or colliding demo IDs")
    require(manifest["version"] == 2, "Regenerate demo data with the synthetic profile generator")
    demo_keys = set(map(str, ids))
    for mapping in ("source_customer_id_by_demo_id", "profile_source_customer_id_by_demo_id"):
        donors = manifest[mapping]
        require(set(donors) == demo_keys and set(donors.values()).issubset(source_ids), "Invalid template mapping")
    require(manifest["labels"] == {"source": None, "available": False,
                                   "performance_metrics": "insufficient_labeled_data"},
            "No observed labels are available for synthetic IDs")
    require(not frame.drop(columns=list(NULLABLE)).isna().any().any(), "Missing required feature value")
    require(all(dtypes[c] == "numeric" for c in columns if c not in {
        "country", "gender", "preferred_channel", "primary_device"
    }), "Unsupported classifier input type")
    require(frame.preferred_channel.isin(("Mobile App", "ATM", "Branch", "No_Channel")).all()
            and frame.primary_device.isin(("Android", "iOS", "Desktop", "No_Device")).all(),
            "Invalid Module 5 channel/device")

    categorical = {"country", "gender", "preferred_channel", "primary_device"}
    numeric = sorted(required - categorical - {"batch_id", "scenario", "feature_snapshot"})
    require(all(pd.api.types.is_numeric_dtype(frame[c]) for c in numeric), "Non-numeric feature")
    values = frame[numeric].to_numpy(dtype=float)
    require(not np.isinf(values).any() and not (values < 0).any(), "Infinite or negative feature")
    count_fields = {"total_accounts", "total_tx_attempts", "total_transactions", "failed_transactions",
                    "cancelled_transactions", "customer_initiated_tx_count", "initiated_tx_attempts",
                    "initiated_failed_transactions", "seg_customer_service_complaints", "seg_has_unresolved_complaint"}
    count_fields.update(c for c in numeric if c.startswith(("num_", "has_")) or c.endswith(
        ("_count", "_complaints", "_logins", "_logins_total", "_q2", "_q3")))
    count_fields -= {"days_observed_q2", "days_observed_q3", "monthly_logins"}
    require(frame[sorted(count_fields)].mod(1).eq(0).all().all(), "Counts and flags must be integers")

    relationships = {
        "accounts": (frame.num_everyday_accounts + frame.num_savings_accounts).eq(frame.total_accounts),
        "account balances": np.isclose(frame.everyday_balance + frame.savings_balance, frame.balance, atol=.011, rtol=0),
        "transaction outcomes": (frame.total_transactions + frame.failed_transactions +
                                 frame.cancelled_transactions).eq(frame.total_tx_attempts),
        "initiated transactions": (frame.customer_initiated_tx_count + frame.initiated_failed_transactions)
                                  .le(frame.initiated_tx_attempts) & frame.initiated_tx_attempts.le(frame.total_tx_attempts)
                                  & frame.customer_initiated_tx_count.le(frame.total_transactions)
                                  & (frame.trans_count_3m + frame.trans_count_prior_3m).le(frame.customer_initiated_tx_count),
        "login outcomes": (frame.successful_logins + frame.failed_logins).eq(frame.total_logins)
                          & (frame.app_logins_total + frame.desktop_logins_total).eq(frame.successful_logins)
                          & frame.mobile_logins_total.eq(frame.app_logins_total)
                          & (frame.login_count_q2 + frame.login_count_q3).eq(frame.successful_logins),
        "complaints": (frame.resolved_complaints_count + frame.unresolved_complaints_count)
                       .eq(frame.customer_service_complaints)
                       & frame.has_unresolved_complaint.eq(frame.unresolved_complaints_count.gt(0))
                       & frame.seg_customer_service_complaints.eq(frame.customer_service_complaints)
                       & frame.seg_has_unresolved_complaint.eq(frame.has_unresolved_complaint),
        "exposure": frame.exposure_days.gt(0)
                    & np.isclose(frame.days_observed_q2 + frame.days_observed_q3, frame.exposure_days, atol=1e-8)
                    & (frame.days_observed_q2.gt(0) | (frame.login_count_q2.eq(0) & frame.trans_count_prior_3m.eq(0))),
        "recency": frame[["recency_days", "complaint_recency_days", "login_recency_days", "seg_recency_days"]]
                        .le(frame.exposure_days + 1e-8, axis=0).all(axis=1)
                   & (frame.customer_initiated_tx_count.gt(0) | np.isclose(frame.recency_days, frame.exposure_days))
                   & (frame.successful_logins.gt(0) | np.isclose(frame.login_recency_days, frame.exposure_days)),
        "resolution delay null": frame.avg_resolution_delay_days.isna().eq(
            frame.resolved_complaints_count.eq(0) & frame.customer_service_complaints.gt(0)),
    }
    for name, valid in relationships.items():
        require(valid.all(), f"Invalid {name} in {int((~valid).sum())} demo rows")
    for rate, count, total in (
        ("initiated_failure_rate", "initiated_failed_transactions", "initiated_tx_attempts"),
        ("login_failure_rate", "failed_logins", "total_logins"),
    ):
        require(frame[rate].dropna().between(0, 1).all() and np.allclose(
            frame[rate], frame[count] / frame[total].replace(0, np.nan), equal_nan=True), f"Invalid {rate}")
    for rate, numerator in (
        ("monthly_initiated_transactions", "customer_initiated_tx_count"),
        ("monthly_transactions", "total_transactions"),
        ("monthly_logins", "successful_logins"),
        ("app_logins_monthly", "app_logins_total"),
    ):
        require(np.allclose(frame[rate], frame[numerator] / frame.exposure_months), f"Invalid {rate}")
    BATCH_CUSTOMER_ADAPTER.validate_python(frame[list(module5.PROFILE_COLUMNS)].to_dict("records"))
    BATCH_CUSTOMER_ADAPTER.validate_python(frame[list(module5.PROFILE_COLUMNS)].assign(
        **{name: frame[f"seg_{name}"] for name in SEG_BEHAVIOR}).to_dict("records"))
    require(examples == build_examples(frame, columns) and len(examples) <= 50,
            "API examples must be exact requests from the demo feature CSV")
    return {"status": "passed", "validated_rows": len(frame), "validated_examples": len(examples),
            "nullable_columns": sorted(NULLABLE), "null_counts": {c: int(frame[c].isna().sum()) for c in sorted(NULLABLE)}}


def score_batch(frame, input_columns, classifier=None, cluster_bundle=None):
    """Score model-specific projections, with identity/metadata excluded from the matrices."""
    probabilities = (score_probabilities(classifier, frame[input_columns])
                     if classifier is not None else None)
    clusters = ({name: segmentation_service.predict_clusters(cluster_bundle, segmentation_frame(frame), name)
                 for name in segmentation_service.ALGORITHMS} if cluster_bundle is not None else {})
    return probabilities, clusters


def inference_report(frame, examples, manifest, *, verify_api=False):
    paths = model_paths()
    report, models = {"by_batch": {}}, {}
    expected = {
        "classifier": ("classifier_checkpoint", "classifier_preprocessor",
                       "classifier_manifest", "classifier_official_snapshot"),
        "segmentation": ("clustering_bundle", "clustering_official_snapshot"),
    }
    for name, load in (("classifier", module5.load),
                       ("segmentation", lambda: segmentation_service.load(SEGMENTATION_DATA_PATH))):
        try:
            missing = [str(paths[key]) for key in expected[name] if not paths[key].is_file()]
            if missing:
                raise FileNotFoundError(", ".join(missing))
            model = load()
            if model is None:
                raise ValueError("Bundle version or official snapshot hash rejected by serving loader")
            if name == "classifier":
                profiles = model.profiles(frame[model.input_columns])
                for column in set(module5.PROFILE_COLUMNS) - set(model.input_columns) - {"customer_id"}:
                    if not np.allclose(profiles[column], frame[column]):
                        raise ValueError(f"Demo profile formula differs from fitted serving profile: {column}")
            models[name] = model
            report[name] = {"status": "loaded"}
        except (AttributeError, EOFError, ImportError, KeyError, OSError, TypeError, UnpicklingError, ValueError) as exc:
            report[name] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    for name, model in list(models.items()):
        try:
            probs, clusters = score_batch(frame, manifest["input_columns"],
                                          model if name == "classifier" else None,
                                          model if name == "segmentation" else None)
            for batch, group in frame.groupby("batch_id", sort=False):
                result = report["by_batch"].setdefault(batch, {})
                if probs is not None:
                    result["mean_churn_probability"] = float(probs[group.index].mean())
                else:
                    result["cluster_counts"] = {key: {str(label): int(count) for label, count in
                                                      zip(*np.unique(values[group.index], return_counts=True))}
                                                for key, values in clusters.items()}
            report[name] = {"status": "passed", "scored_rows": len(frame)}
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            del models[name]
            report[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    report["status"] = "passed" if len(models) == len(expected) else "partial" if models else "unavailable"
    if verify_api and models:
        report["api_parity"] = check_api(frame, examples, manifest["input_columns"], models)
    return report


def check_api(frame, examples, input_columns, models):
    from fastapi.testclient import TestClient

    from api.main import app

    rows = frame.set_index("customer_id", drop=False).loc[[example["customer_id"] for example in examples]]
    records = rows.to_dict("records")
    probabilities, clusters = score_batch(rows, input_columns, models.get("classifier"), models.get("segmentation"))
    parity = {"classifier": 0, "kmeans": 0, "hierarchical": 0, "max_probability_delta": None}
    with TestClient(app) as client:
        if probabilities is not None:
            parity["max_probability_delta"] = 0.0
            for request, local in zip(examples, probabilities):
                response = client.post("/api/predict", json=request)
                if response.status_code != 200:
                    raise ValueError(f"/api/predict [{response.status_code}]: {response.text}")
                body = response.json()
                delta = abs(body["churn_probability"] - float(local))
                if delta > 1e-7 or body["customer_id"] != str(request["customer_id"]):
                    raise ValueError(f"Churn API parity failed for {request['customer_id']}: delta={delta}")
                parity["max_probability_delta"] = max(parity["max_probability_delta"], delta)
                parity["classifier"] += 1
        for algorithm, expected in clusters.items():
            for row, cluster in zip(records, expected):
                response = client.post("/api/segments/predict", params={"algorithm": algorithm},
                                       json=request_for(row, input_columns, segmentation=True))
                if response.status_code != 200 or response.json()["cluster_id"] != cluster:
                    raise ValueError(f"{algorithm} API parity failed for {row['customer_id']}: {response.text}")
                parity[algorithm] += 1
    return parity


def read_outputs(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def checked_content(name):
        content = (output / name).read_bytes()
        if sha256(content).hexdigest() != manifest["output_sha256"][name]:
            raise ValueError(f"Demo artifact hash mismatch: {name}")
        return content

    frame = pd.read_csv(BytesIO(checked_content("customer_features.csv")), float_precision="round_trip")

    def reject_nonfinite(token):
        raise ValueError(f"Invalid JSON value: {token}")

    examples = [json.loads(line, parse_constant=reject_nonfinite) for line in
                checked_content("api_examples.jsonl").decode("utf-8").splitlines()]
    return manifest, frame, examples


def validate(source=SOURCE, output=OUTPUT):
    manifest, frame, examples = read_outputs(Path(output))
    tables, observed_sources = load_sources(Path(source))
    for name in manifest["sources"].keys() | observed_sources.keys():
        if observed_sources.get(name) != manifest["sources"].get(name):
            raise ValueError(f"Source audit mismatch: {name}")
    for name, path in model_paths().items():
        if path.is_file() and module5.file_hash(path) != manifest["models"][name]["sha256"]:
            raise ValueError(f"Model artifact hash mismatch: {name}")
    source_ids = tables["kaggle_bank_churn"].customer_id
    result = check_dataset(frame, examples, manifest, source_ids)
    columns = manifest["input_columns"]
    result["novelty"] = check_novelty(frame, aggregate(tables, columns), columns)
    if result["novelty"] != manifest["validation"]["novelty"]:
        raise ValueError("Synthetic profile novelty disagrees with the manifest")
    for batch in BATCHES:
        actual = summarize(frame.loc[frame.batch_id.eq(batch)])
        for name, record in actual.items():
            expected = manifest["monitoring"]["by_batch"][batch]["demo"][name]
            for key, value in record.items():
                other = expected[key]
                if value != other and (value is None or other is None or not math.isclose(value, other, abs_tol=1e-12)):
                    raise ValueError(f"Monitoring statistic mismatch: {batch}.{name}.{key}")
    return {"data": result, "inference": inference_report(frame, examples, manifest, verify_api=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = validate(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Demo validation failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["inference"]["status"] != "passed":
        parser.exit(1, "Data checks passed. See inference report for unavailable checks.\n")


if __name__ == "__main__":
    main()
