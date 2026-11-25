"""Generate five synthetic demo cohorts from audited reference tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from api import module5, segmentation_service
from api.official_data import assemble_profiles
from demo_data.synthesize import (
    MONEY_CHANGE,
    TIME_WARP_SECONDS,
    match_profiles,
    synthesize,
)
from model_development.preprocessing.behavioral_momentum import BehavioralMomentumEngineer
from model_development.preprocessing.customer_friction import CustomerFrictionEngineer
from preprocessing.ingestion.etl import BankDataETL
from segmentation.data import REQUIRED, SNAPSHOT, START, guard_tables

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/full_dataset"
OUTPUT = ROOT / "data/demo"
BATCHES = ("normal_01", "normal_02", "normal_03", "engagement_shift", "friction_shift")
SNAPSHOT_DATE = SNAPSHOT.date().isoformat()
Q3 = pd.Timestamp(BankDataETL.DEFAULT_Q3_START)
QUIET_DAYS = 21
ENGAGEMENT_DROP = .55
FRICTION_SHARE = .65
BASELINE_DROP = .04
HARD_CAP = 10_000
META = ("customer_id", "batch_id", "scenario", "feature_snapshot")
SEG_BEHAVIOR = segmentation_service.BEHAVIORAL_NUMERIC
AUDIT_COLUMNS = (
    "days_observed_q2", "days_observed_q3", "customer_initiated_tx_count",
    "initiated_tx_attempts", "initiated_failed_transactions", "monthly_initiated_transactions",
    "monthly_logins", "initiated_failure_rate", "login_failure_rate",
)
MONITOR_COLUMNS = (
    "monthly_initiated_transactions", "monthly_logins", "recency_days", "login_recency_days",
    "customer_service_complaints", "unresolved_complaints_count", "login_failure_rate",
    "initiated_failure_rate",
)


def sample_count(n_base, multiplier=10, max_samples=HARD_CAP):
    if min(n_base, multiplier, max_samples) <= 0:
        raise ValueError("Customer count, multiplier and maximum must be positive")
    return min(multiplier * n_base, max_samples, HARD_CAP)


def batch_sizes(total):
    quotient, remainder = divmod(total, len(BATCHES))
    return {batch: quotient + (i < remainder) for i, batch in enumerate(BATCHES)}


def load_sources(root):
    tables, manifest = {}, {}
    for name, required in REQUIRED.items():
        path = root / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source table: {path}")
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Unreadable source table: {path}: {exc}") from exc
        if frame.empty:
            raise ValueError(f"Empty source table: {path}")
        missing = set(required) - set(frame)
        if name == "accounts" and "payout_account_id" not in frame:
            missing.add("payout_account_id")
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        if "customer_id" in frame and (
            not pd.api.types.is_integer_dtype(frame.customer_id)
            or frame.customer_id.isna().any() or frame.customer_id.le(0).any()
            or frame.customer_id.gt(np.iinfo(np.int64).max).any()
        ):
            raise ValueError(f"{path.name}: customer_id must be a positive int64")
        retained = [*required, "payout_account_id"] if name == "accounts" else required
        tables[name] = frame.loc[:, retained].copy()
        manifest[path.name] = {"rows": len(frame), "sha256": module5.file_hash(path)}
    owner = tables["accounts"].set_index("account_id").customer_id
    if owner.index.has_duplicates:
        raise ValueError("accounts.csv: account_id must be unique")
    for name, frame in tables.items():
        ids = frame.customer_id if "customer_id" in frame else frame.account_id.map(owner)
        manifest[f"{name}.csv"]["unique_customer_ids"] = int(ids.nunique())
    tables, audit = guard_tables(tables)
    accounts = tables["accounts"].set_index("account_id")
    savings = accounts.account_type.eq("savings")
    payout = accounts.loc[savings, "payout_account_id"]
    if (not accounts.loc[savings, "renewal_type"].isin(("manual", "auto")).all()
            or accounts.loc[~savings, ["payout_account_id", "renewal_type"]].notna().any().any()
            or not payout.map(accounts.account_type).eq("everyday").all()
            or not payout.map(accounts.customer_id).eq(accounts.loc[savings, "customer_id"]).all()):
        raise ValueError("accounts.csv: invalid savings renewal or payout account")
    for name, result in audit.items():
        manifest[f"{name}.csv"]["excluded_outside_observation"] = result["excluded_outside_observation"]
    tables["kaggle_bank_churn"].drop(columns="churn", inplace=True)
    source_manifest = root / "generation_manifest.json"
    if source_manifest.is_file():
        manifest[source_manifest.name] = {"sha256": module5.file_hash(source_manifest)}
    return tables, manifest


def donors_and_ids(ids, count, rng):
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    donor_ids = np.concatenate([rng.permutation(ids) for _ in range(-(-count // len(ids)))])[:count]
    # Start above source IDs and within JavaScript's exact integer range for UI use.
    start = max(100_000_000, int(ids[-1]) + 1)
    if start + count - 1 > 2**53 - 1:
        raise ValueError("Cannot allocate demo IDs above the source within the UI integer range")
    return donor_ids, np.arange(start, start + count, dtype=np.int64)


def initiated_transactions(tables):
    tx = tables["transactions"]
    return (tx.channel.ne("system") & ~tx.transaction_type.isin(["bank_fee", "interest_credit"])
            & (tx.direction.eq("out") | tx.transaction_type.eq("cash_deposit")))


def shift_engagement(tables, rng, *, probability=ENGAGEMENT_DROP, quiet_days=QUIET_DAYS, start=Q3):
    tx, log = (tables[name] for name in ("transactions", "login"))
    accounts = tables["accounts"].set_index("account_id")
    owner = tx.account_id.map(accounts.customer_id)
    # Keep both sides of linked transfers and all activity on closed or savings accounts.
    initiated = initiated_transactions(tables)
    removable = (initiated & tx.transfer_id.isna() & tx.direction.eq("out")
                 & tx.account_id.map(accounts.account_type).eq("everyday")
                 & tx.account_id.map(accounts.closed_at).isna())
    tx_time, login_time = tx.transaction_datetime, log.sign_in_datetime
    keys = ("customer_id", "day")
    tx_keys = pd.MultiIndex.from_arrays([owner, tx_time.dt.normalize()], names=keys)
    login_keys = pd.MultiIndex.from_arrays([log.customer_id, login_time.dt.normalize()], names=keys)
    days = tx_keys[removable & tx_time.ge(start)].union(login_keys[login_time.ge(start)]).unique().sort_values()
    quiet = days[(rng.random(len(days)) < probability) |
                 (days.get_level_values(1) >= SNAPSHOT - pd.Timedelta(days=quiet_days))]
    remove_tx = removable & tx_keys.isin(quiet)
    tables["transactions"] = tx.loc[~remove_tx].copy()

    # Keep prior observed successful sessions for retained mobile actions, including across midnight.
    mobile = ~remove_tx & initiated & tx.channel.eq("mobile_app")
    actions = pd.DataFrame({"customer_id": owner[mobile], "time": tx_time[mobile]}).sort_values("time")
    sessions = (log.loc[log.login_status.eq("success") & log.device_type.isin(("android", "ios")),
                        ["customer_id", "sign_in_datetime", "login_id"]]
                .rename(columns={"sign_in_datetime": "time"}).sort_values("time"))
    matches = pd.merge_asof(actions, sessions, on="time", by="customer_id",
                            tolerance=pd.Timedelta(minutes=50), direction="backward")
    remove_login = login_keys.isin(quiet) & ~log.login_id.isin(matches.login_id.dropna())
    tables["login"] = log.loc[~remove_login].copy()
    return {"removed_transactions": int(remove_tx.sum()), "removed_login_attempts": int(remove_login.sum())}


def shift_friction(tables, rng):
    login, complaints, context = (tables[name] for name in ("login", "complaints", "customer_context"))
    start = context.set_index("customer_id").observation_start
    candidates = login.loc[login.login_status.eq("success") &
                           login.sign_in_datetime.ge(Q3) &
                           login.sign_in_datetime.sub(pd.Timedelta(seconds=1)).ge(login.customer_id.map(start))]
    candidates = candidates.sort_values("login_id").copy()
    candidates["_draw"] = rng.random(len(candidates))
    anchors = candidates.sort_values("_draw").drop_duplicates("customer_id").sort_values("customer_id")
    anchors = anchors.loc[rng.random(len(anchors)) < FRICTION_SHARE]
    extra = anchors.drop(columns="_draw").assign(
        login_id="DEMO-FAIL-" + anchors.login_id.astype(str), login_status="failed",
        sign_in_datetime=anchors.sign_in_datetime - pd.Timedelta(seconds=1),
    )
    new_complaints = pd.DataFrame({
        "customer_id": anchors.customer_id, "complaint_id": "DEMO-COMPLAINT-" + anchors.login_id.astype(str),
        "complaint_datetime": anchors.sign_in_datetime,
        "resolved_datetime": pd.Series(pd.NaT, index=anchors.index, dtype="datetime64[ns]"),
    })
    tables["login"] = pd.concat([login, extra], ignore_index=True)
    tables["complaints"] = pd.concat([complaints, new_complaints], ignore_index=True)
    return {"added_failed_logins": len(extra), "added_unresolved_complaints": len(new_complaints)}


class InMemoryETL(BankDataETL):
    def __init__(self, tables):
        super().__init__()
        self.tables = tables

    def _load_tables(self):
        return {name: frame.copy() for name, frame in self.tables.items()}


def aggregate(tables, input_columns):
    raw = InMemoryETL(tables).process(verbose=False).set_index("customer_id")
    # Build diagnostic engagement columns independent of the classifier's posted-transaction volume.
    tx = tables["transactions"]
    actions = tx.loc[initiated_transactions(tables)]
    ids = actions.account_id.map(tables["accounts"].set_index("account_id").customer_id)
    raw["initiated_tx_attempts"] = ids.value_counts().reindex(raw.index, fill_value=0)
    raw["initiated_failed_transactions"] = ids[actions.transaction_status.eq("failed")].value_counts().reindex(
        raw.index, fill_value=0)
    raw["monthly_initiated_transactions"] = raw.customer_initiated_tx_count / raw.exposure_months
    raw["initiated_failure_rate"] = raw.initiated_failed_transactions / raw.initiated_tx_attempts.replace(0, np.nan)
    # Use only the classifier's committed inputs so profile formulas match its serving path.
    main = raw.loc[:, input_columns]
    profiles = CustomerFrictionEngineer().transform(BehavioralMomentumEngineer().transform(main))
    # Diagnostic successful engagement is independent of the legacy serving formula.
    raw["monthly_logins"] = raw.successful_logins / raw.exposure_months
    raw["login_failure_rate"] = raw.failed_logins / raw.total_logins.replace(0, np.nan)
    for name in set(module5.PROFILE_COLUMNS) - set(raw) - {"customer_id"}:
        raw[name] = profiles[name]
    segmentation = assemble_profiles(tables["kaggle_bank_churn"], {
        "account": tables["accounts"], "transaction": tx, "login": tables["login"],
        "complaint": tables["complaints"], "customercontext": tables["customer_context"],
    }, demo=True).set_index("customer_id")
    display = [name for name in module5.PROFILE_COLUMNS if name not in {"customer_id", *input_columns}]
    return raw[[*input_columns, *display, *AUDIT_COLUMNS]].join(
        segmentation[list(SEG_BEHAVIOR)].add_prefix("seg_"), validate="one_to_one")


def summarize(frame):
    summary = {}
    for name in MONITOR_COLUMNS:
        values = frame[name].dropna()
        quantiles = values.quantile([.1, .5, .9]).tolist() if len(values) else [None] * 3
        summary[name] = {
            "mean": float(values.mean()) if len(values) else None,
            "p10": quantiles[0], "median": quantiles[1], "p90": quantiles[2],
            "missing": len(frame) - len(values),
        }
    return summary


def build_demo(tables, input_columns, multiplier, max_samples, seed):
    reference = aggregate(tables, input_columns)
    count = sample_count(len(reference), multiplier, max_samples)
    rng = np.random.default_rng(seed)
    donor_ids, demo_ids = donors_and_ids(reference.index, count, rng)
    profile_ids = match_profiles(tables["kaggle_bank_churn"], rng).loc[donor_ids].to_numpy()
    warp_seconds = float(rng.choice((-1, 1)) * rng.uniform(*TIME_WARP_SECONDS))
    frames, batch_details = [], {}
    offset = 0
    for batch, size in batch_sizes(count).items():
        end = offset + size
        ids, donors = demo_ids[offset:end], donor_ids[offset:end]
        details = {"rows": size, "matched_donor": summarize(reference.loc[donors])}
        before = reference.iloc[:0]
        if size:
            histories = synthesize(tables, ids, donors, profile_ids[offset:end], warp_seconds, rng)
            details["baseline_event_edits"] = shift_engagement(
                histories, rng, probability=BASELINE_DROP, quiet_days=0, start=START)
            before = aggregate(histories, input_columns).loc[ids]
        details["synthetic_baseline"] = summarize(before)
        after = before
        if size and batch in ("engagement_shift", "friction_shift"):
            shift = shift_engagement if batch == "engagement_shift" else shift_friction
            details["changed_events_in_synthetic_histories"] = shift(histories, rng)
            after = aggregate(histories, input_columns).loc[ids]
        frame = after.reset_index(drop=True)
        frame.insert(0, "feature_snapshot", SNAPSHOT_DATE)
        frame.insert(0, "scenario", "normal" if batch.startswith("normal_") else batch)
        frame.insert(0, "batch_id", batch)
        frame.insert(0, "customer_id", ids)
        frames.append(frame)
        details["demo"] = summarize(frame)
        details["delta_mean"] = {
            name: after_metric["mean"] - details["synthetic_baseline"][name]["mean"]
            if after_metric["mean"] is not None and details["synthetic_baseline"][name]["mean"] is not None else None
            for name, after_metric in details["demo"].items()
        }
        batch_details[batch] = details
        offset = end
    mapping = {
        "source_customer_id_by_demo_id": dict(zip(map(str, demo_ids), map(int, donor_ids))),
        "profile_source_customer_id_by_demo_id": dict(zip(map(str, demo_ids), map(int, profile_ids))),
    }
    monitoring = {"source_reference": summarize(reference), "by_batch": batch_details}
    return pd.concat(frames, ignore_index=True), reference, mapping, monitoring, warp_seconds


def generate(source=SOURCE, output=OUTPUT, multiplier=10, max_samples=HARD_CAP, seed=42):
    from demo_data.validate import (
        build_examples,
        check_dataset,
        check_novelty,
        inference_report,
        model_paths,
    )

    source, output = Path(source), Path(output)
    protected = (source, ROOT / "data/official", ROOT / "data/processed", ROOT / "artifacts",
                 ROOT / "models", ROOT / "results")
    destination = output.resolve()
    if any(
        destination.is_relative_to(path.resolve()) or path.resolve().is_relative_to(destination) for path in protected
    ):
        raise ValueError("Use a dedicated demo output directory outside source and official data")
    filenames = ("customer_features.csv", "api_examples.jsonl", "manifest.json")
    if {output.joinpath(name).resolve() for name in filenames}.intersection(
        path.resolve() for path in model_paths().values()
    ):
        raise ValueError("Demo output conflicts with a serving artifact")
    model_manifest = json.loads(module5.MANIFEST_PATH.read_text(encoding="utf-8"))
    input_columns = model_manifest["input_columns"]
    tables, sources = load_sources(source)
    n_base = len(tables["kaggle_bank_churn"])
    frame, reference, mapping, monitoring, warp_seconds = build_demo(
        tables, input_columns, multiplier, max_samples, seed)
    donors = mapping["source_customer_id_by_demo_id"]
    manifest = {
        "version": 2, "N_base": n_base, "formula": "min(multiplier * N_base, max_samples, 10000)",
        "N_demo": len(frame), "multiplier": multiplier, "max_samples": max_samples, "seed": seed,
        "snapshot": SNAPSHOT_DATE, "snapshot_exclusive": True, "simulated_ingestion_batches": True,
        "batch_rows": batch_sizes(len(frame)), "sources": sources,
        "models": {name: {"sha256": module5.file_hash(path) if path.is_file() else None}
                   for name, path in model_paths().items()},
        "input_columns": input_columns, "input_dtypes": model_manifest["input_dtypes"],
        **mapping,
        "generation": {
            "method": "Recombine matched demographic and history templates, then synthesize money, times and activity.",
            "baseline": {"portfolio_money_relative_change": list(MONEY_CHANGE), "clock_warp_seconds": warp_seconds,
                         "customer_day_drop_probability": BASELINE_DROP,
                         "distinct_profile_template_rate": sum(
                             value != mapping["profile_source_customer_id_by_demo_id"][key]
                             for key, value in donors.items()) / len(donors)},
            "engagement_shift": {"q3_customer_day_drop_probability": ENGAGEMENT_DROP, "quiet_days_before_snapshot": QUIET_DAYS,
                                 "protected": "linked transfers, closed/savings accounts, passive postings and sessions for retained mobile actions"},
            "friction_shift": {"eligible_customer_probability": FRICTION_SHARE,
                               "events": "one failed login before a successful Q3 login and one unresolved complaint at its timestamp"},
            "donor_reuse_rate": 1 - len(set(donors.values())) / len(donors),
            "duplicate_feature_vector_rate": float(frame.drop(columns=list(META)).duplicated().mean()),
        },
        "labels": {"source": None, "available": False, "performance_metrics": "insufficient_labeled_data"},
        "monitoring": monitoring,
    }
    examples = build_examples(frame, input_columns)
    manifest["validation"] = check_dataset(frame, examples, manifest, tables["kaggle_bank_churn"].customer_id)
    manifest["validation"]["novelty"] = check_novelty(frame, reference, input_columns)
    manifest["inference"] = inference_report(frame, examples, manifest)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as work:
        stage = Path(work)
        frame.to_csv(stage / "customer_features.csv", index=False, lineterminator="\n")
        (stage / "api_examples.jsonl").write_text(
            "".join(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n" for example in examples),
            encoding="utf-8", newline="\n")
        manifest["output_sha256"] = {name: module5.file_hash(stage / name)
                                     for name in ("customer_features.csv", "api_examples.jsonl")}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8", newline="\n")
        lock = output / ".publishing"
        lock.mkdir()
        try:
            for name in filenames:
                (stage / name).replace(output / name)
        finally:
            lock.rmdir()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=HARD_CAP)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = generate(args.source, args.output, args.multiplier, args.max_samples, args.seed)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Demo generation failed: {exc}\n")
    print(f"N_base={result['N_base']}; N_demo={result['N_demo']}; batches={result['batch_rows']}")
    print(f"Inference: {result['inference']['status']}; output: {args.output}")


if __name__ == "__main__":
    main()
