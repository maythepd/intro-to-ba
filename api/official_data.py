"""Prepare model-ready customer profiles from the group's public six-table Kaggle release.

Source: maythepd/full-dataset-intro2ba, snapshot: 2026-10-01 (exclusive).
Run: python -m api.official_data --download
"""
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

DATASET = "maythepd/full-dataset-intro2ba"
BOUNDARY = pd.Timestamp("2026-10-01")
WINDOW_START = pd.Timestamp("2026-04-01")
OUTPUT = Path(os.getenv("OFFICIAL_FEATURES_PATH", "data/official/customer_features.csv"))
BASE = Path("data/Bank Customer Churn Prediction.csv")

def find_table(root: Path, keyword: str) -> Path:
    matches = [p for p in root.rglob("*") if p.suffix.lower() in (".csv", ".parquet") and
               keyword in re.sub("[^a-z]", "", p.stem.lower())]
    if not matches:
        raise FileNotFoundError(f"Missing '{keyword}' table in {root}; check Kaggle dataset contents")
    matches.sort(key=lambda p: (len(p.parts), len(p.name)))
    return matches[0]

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

def date_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame:
        raise ValueError(f"Required timestamp column not found: {column}")
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None)

def merge_metric(df: pd.DataFrame, series: pd.Series, column: str, default: float = 0) -> pd.DataFrame:
    if series.empty:
        df[column] = default
    else:
        df[column] = df["customer_id"].map(series).fillna(default)
    return df

def assemble(root: Path, base_path: Path = BASE) -> pd.DataFrame:
    if not base_path.exists():
        base_path = find_table(root, "bankcustomerchurnprediction")
    base = read_table(base_path)
    required_base = ["customer_id", "credit_score", "country", "gender", "age", "tenure",
                     "balance", "products_number", "credit_card", "active_member",
                     "estimated_salary", "churn"]
    missing = set(required_base) - set(base.columns)
    if missing:
        raise ValueError(f"Base customer data missing: {sorted(missing)}")
    out = base[required_base].copy().drop_duplicates("customer_id")
    if len(out) != 10000:
        raise ValueError(f"Expected 10,000 unique base customers, found {len(out)}")
    out["customer_id"] = pd.to_numeric(out["customer_id"]).astype("int64")

    # Verify all six tables from the dataset release, not the older synthetic feature CSV.
    tables = {k: read_table(find_table(root, k)) for k in
              ("account", "transaction", "login", "complaint", "savingscycle", "customercontext")}
    accounts = tables["account"].copy()
    trans = tables["transaction"].copy()
    login = tables["login"].copy()
    complaints = tables["complaint"].copy()
    context = tables["customercontext"].copy()
    for frame, col in ((trans, "transaction_datetime"), (login, "sign_in_datetime"),
                       (complaints, "complaint_datetime"), (context, "observation_start")):
        date_column(frame, col)

    # Customer exposure starts when they enter the observation period.
    if context["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_context.customer_id")
    exp = context.set_index("customer_id")["observation_start"]
    out["exposure_start"] = out["customer_id"].map(exp).fillna(WINDOW_START).clip(lower=WINDOW_START)
    if (out["exposure_start"] >= BOUNDARY).any():
        raise ValueError("Invalid customer exposure period")

    if accounts["account_id"].duplicated().any():
        raise ValueError("Duplicate account_id in accounts")
    trans = trans.merge(accounts[["account_id", "customer_id"]], on="account_id",
                        validate="many_to_one", how="inner")
    if "customer_id" not in trans:
        raise ValueError("Transactions are missing account ownership")
    # Observation-only events; exclude system postings and failed attempts.
    trans = trans.loc[
        (trans["transaction_datetime"] >= WINDOW_START) &
        (trans["transaction_datetime"] < BOUNDARY) &
        (trans["transaction_status"].astype(str).str.lower() == "success")
    ].copy()
    trans["amount"] = pd.to_numeric(trans["amount"], errors="coerce").fillna(0)
    trans["channel"] = trans["channel"].astype(str).str.lower().str.strip()
    trans["transaction_type"] = trans["transaction_type"].astype(str).str.lower().str.strip()
    trans["direction"] = trans["direction"].astype(str).str.lower().str.strip()
    initiated = trans.loc[
        trans["channel"].ne("system") &
        ~trans["transaction_type"].isin(["interest_credit", "bank_fee"])
    ].copy()
    # Count linked internal transfer records once within each customer's activity.
    if "transfer_id" in initiated:
        transfer_key = initiated["transfer_id"].astype("string")
        initiated["_event_key"] = np.where(transfer_key.notna(),
                        "transfer_" + transfer_key.fillna(""), initiated["transaction_id"].astype(str))
        initiated = initiated.drop_duplicates(["customer_id", "_event_key"])

    after = pd.Timestamp("2026-07-01")
    recent = initiated.loc[initiated["transaction_datetime"] >= after]
    prior = initiated.loc[(initiated["transaction_datetime"] >= "2026-05-01") &
                           (initiated["transaction_datetime"] < after)]
    counts_recent = recent.groupby("customer_id").size()
    counts_prior = prior.groupby("customer_id").size()
    out = merge_metric(out, counts_recent, "_count_3m")
    out = merge_metric(out, counts_prior, "_count_prior_2m")
    out["monthly_transactions"] = (out["_count_3m"] / 3).round().astype(int)
    recent_monthly = out["_count_3m"] / 3
    prior_monthly = out["_count_prior_2m"] / 2
    out["trans_trend_ratio"] = np.where(
        prior_monthly > 0, recent_monthly / prior_monthly, 1.0
    ).clip(0, 2.5).round(3)
    outgoing = recent.loc[recent["direction"].eq("out")]
    out = merge_metric(out, outgoing.groupby("customer_id")["amount"].sum(), "trans_amount_3m")
    # Last observed activity: successful customer-initiated transaction or successful login.
    login = login.loc[
        (login["sign_in_datetime"] >= WINDOW_START) &
        (login["sign_in_datetime"] < BOUNDARY) &
        login["login_status"].astype(str).str.lower().eq("success")
    ].copy()
    trans_last = initiated.groupby("customer_id")["transaction_datetime"].max()
    login_last = login.groupby("customer_id")["sign_in_datetime"].max()
    last_activity = pd.concat([trans_last.rename("transactions"),login_last.rename("logins")],axis=1).max(axis=1)
    out["last_activity"] = out["customer_id"].map(last_activity)
    out["recency_days"] = ((BOUNDARY - out["last_activity"].fillna(out["exposure_start"]))
                           .dt.days.clip(1, 180)).astype(int)
    last30 = login.loc[login["sign_in_datetime"] >= "2026-09-01"]
    out = merge_metric(out, last30.groupby("customer_id").size(), "app_logins_monthly")
    out["app_logins_monthly"] = out["app_logins_monthly"].astype(int)

    complaints = complaints.loc[
        (complaints["complaint_datetime"] >= WINDOW_START) &
        (complaints["complaint_datetime"] < BOUNDARY)
    ].copy()
    date_column(complaints, "resolved_datetime")
    out = merge_metric(out, complaints.groupby("customer_id").size(), "customer_service_complaints")
    pending = complaints.loc[complaints["resolved_datetime"].isna() |
                             (complaints["resolved_datetime"] >= BOUNDARY)]
    out["has_unresolved_complaint"] = out["customer_id"].isin(pending["customer_id"]).astype(int)
    out["customer_service_complaints"] = out["customer_service_complaints"].astype(int)

    channels = {"mobile_app":"Mobile App","online_web":"Online Web", "atm":"ATM","branch":"Branch"}
    init_channel = initiated.loc[initiated["channel"].isin(channels)]
    if not init_channel.empty:
        most_common = (init_channel.groupby(["customer_id","channel"]).size()
                       .reset_index(name="n").sort_values(["customer_id","n"], ascending=[True,False])
                       .drop_duplicates("customer_id").set_index("customer_id")["channel"])
    else:
        most_common = pd.Series(dtype="object")
    out["preferred_channel"] = out["customer_id"].map(most_common).map(channels)
    # Fallback is a documented inference based on observed logins, not observed channel preference.
    out["preferred_channel"] = out["preferred_channel"].fillna(
        pd.Series(np.where(out["app_logins_monthly"] > 0, "Mobile App", "Online Web"),index=out.index)
    )
    if "device_type" in login:
        latest = login.sort_values("sign_in_datetime").drop_duplicates("customer_id",keep="last")
        raw_device = out["customer_id"].map(latest.set_index("customer_id")["device_type"]).astype("string").str.lower()
        out["primary_device"] = raw_device.map({"ios":"iOS","android":"Android","desktop":"Web/Desktop"}).fillna("Unknown")
    else:
        out["primary_device"] = "Unknown"
    out = out.drop(columns=["_count_3m","_count_prior_2m","exposure_start","last_activity"])
    if out["customer_id"].duplicated().any() or out.isnull().any().any():
        raise ValueError("Derived profiles contain duplicate IDs or nulls")
    return out

def main():
    parser=argparse.ArgumentParser(description="Materialize customer features from the official six-table dataset")
    parser.add_argument("--download",action="store_true",help="Download official Kaggle dataset")
    parser.add_argument("--source",type=Path,default=Path(os.getenv("OFFICIAL_DATA_DIR","data/official/raw")))
    args=parser.parse_args()
    root=args.source
    if args.download:
        try:
            import kagglehub
        except ImportError as exc:
            raise SystemExit("Install kagglehub: pip install kagglehub") from exc
        root=Path(kagglehub.dataset_download(DATASET))
    frame=assemble(root)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUTPUT,index=False)
    print(f"Official dataset: {root}; derived {len(frame)} profiles -> {OUTPUT}")

if __name__ == "__main__":
    main()
