"""Independent, reproducible illustrative outcomes; never classifier ground truth."""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from monitoring.settings import ROOT, digest

POLICY_PATH = ROOT / "config/synthetic_outcome_v1.json"
VERSION = "synthetic_outcome_v1"
COLUMNS = ["customer_id", "batch_id", "churn", "label_source", "label_policy_version", "label_seed"]
FEATURES = ["recency_days", "login_recency_days", "trans_count_3m", "trans_count_prior_3m",
            "login_count_q2", "login_count_q3", "days_observed_q2", "days_observed_q3",
            "has_unresolved_complaint", "failed_logins", "total_logins"]


def simulate_labels(frame: pd.DataFrame, *, seed: int = 4242, policy: dict | None = None) -> pd.DataFrame:
    policy = policy or json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    # Versioned code and config together define the policy. A changed policy needs a new version.
    if policy != json.loads(POLICY_PATH.read_text(encoding="utf-8")) or policy["version"] != VERSION:
        raise ValueError("Unsupported outcome policy; version the simulator before changing assumptions")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Label seed must be an integer in [0, 2**32)")
    if frame.customer_id.isna().any() or not frame.customer_id.is_unique:
        raise ValueError("Outcome IDs must be unique and non-null")
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Outcome policy requires finite nonnegative features")
    if not values.has_unresolved_complaint.isin([0, 1]).all() or (values.failed_logins > values.total_logins).any():
        raise ValueError("Invalid complaint flag or login counts")

    def rate(count, days):
        return np.divide(count * 30, days, out=np.zeros(len(frame)), where=days.to_numpy() > 0)

    smooth = policy["monthly_rate_smoothing"]
    tx_ratio = ((rate(values.trans_count_3m, values.days_observed_q3) + smooth) /
                (rate(values.trans_count_prior_3m, values.days_observed_q2) + smooth))
    login_ratio = ((rate(values.login_count_q3, values.days_observed_q3) + smooth) /
                   (rate(values.login_count_q2, values.days_observed_q2) + smooth))
    terms = {
        "recency": values.recency_days.to_numpy(),
        "login_recency": values.login_recency_days.to_numpy(),
        "engagement_decline": (np.clip(1 - tx_ratio, 0, 1) + np.clip(1 - login_ratio, 0, 1)) / 2,
        "unresolved_complaint": values.has_unresolved_complaint.to_numpy(),
        "login_failure": np.divide(values.failed_logins, values.total_logins,
                                   out=np.zeros(len(frame)), where=values.total_logins.to_numpy() > 0),
    }
    z = np.full(len(frame), policy["intercept"])
    for name, definition in policy["terms"].items():
        z += definition["weight"] * np.clip(terms[name] / definition["scale"], 0, definition["cap"])
    # Per-customer randomness makes outcomes invariant to sorting, subsetting and batch execution order.
    uniforms = np.array([int(digest([VERSION, seed, str(person)])[:13], 16) / 2**52
                         for person in frame.customer_id])
    result = frame[["customer_id", "batch_id"]].copy()
    result["churn"] = (uniforms < expit(z)).astype(int)
    result["label_source"] = "synthetic_stress"
    result["label_policy_version"] = VERSION
    result["label_seed"] = seed
    return result[COLUMNS]


def generate_labels(input_path: Path, output_path: Path, *, seed: int = 4242) -> dict:
    content = input_path.read_bytes()
    frame = pd.read_csv(BytesIO(content), float_precision="round_trip")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    labels = simulate_labels(frame, seed=seed, policy=policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = labels.to_csv(index=False, lineterminator="\n").encode("utf-8")
    metadata = {"dataset_sha256": sha256(content).hexdigest(), "output_sha256": sha256(encoded).hexdigest(),
                "label_source": "synthetic_stress", "label_policy_version": VERSION, "label_seed": seed,
                "policy": policy, "policy_sha256": digest(policy), "row_count": len(labels),
                "by_batch": {name: {"samples": len(group), "positives": int(group.churn.sum())}
                             for name, group in labels.groupby("batch_id", sort=False)}}
    # Metadata is published last; readers reject an interrupted or mixed pair.
    stage = output_path.with_suffix(output_path.suffix + ".tmp")
    stage.write_bytes(encoded)
    stage.replace(output_path)
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def read_labels(path: Path, frame: pd.DataFrame, dataset_sha256: str):
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    content = path.read_bytes()
    if (sha256(content).hexdigest() != metadata["output_sha256"] or
            metadata["dataset_sha256"] != dataset_sha256 or digest(metadata["policy"]) != metadata["policy_sha256"]):
        raise ValueError("Synthetic outcome checksum/dataset/policy mismatch")
    labels = pd.read_csv(BytesIO(content))
    expected = simulate_labels(frame, seed=metadata["label_seed"], policy=metadata["policy"])
    if list(labels) != COLUMNS or len(labels) != len(frame) or not labels.customer_id.is_unique:
        raise ValueError("Synthetic outcomes require an exact one-to-one join")
    labels = labels.set_index("customer_id").sort_index()
    expected = expected.set_index("customer_id").sort_index()
    if not labels.equals(expected) or metadata["label_policy_version"] != VERSION or metadata["row_count"] != len(frame):
        raise ValueError("Synthetic outcome provenance or labels do not match the independent policy")
    return labels, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/demo/customer_features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/demo/synthetic_outcomes.csv")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    print(json.dumps(generate_labels(args.input, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
