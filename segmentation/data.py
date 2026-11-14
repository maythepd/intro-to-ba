"""Guard the existing relational ETL; never trust an unverified feature cache."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from preprocessing.etl import BankDataETL
from .features import ORIGINAL_FEATURES

START = pd.Timestamp("2026-04-01")
SNAPSHOT = pd.Timestamp("2026-10-01")
REQUIRED = {
    "kaggle_bank_churn": ["customer_id", "churn", *ORIGINAL_FEATURES],
    "customer_context": ["customer_id", "relationship_start_date", "observation_start", "snapshot_boundary"],
    "accounts": ["account_id", "customer_id", "account_type", "created_at", "closed_at", "balance", "renewal_type"],
    "transactions": ["transaction_id", "account_id", "transaction_datetime", "amount", "direction",
                     "transaction_type", "transaction_status", "channel", "transfer_id"],
    "login": ["login_id", "customer_id", "sign_in_datetime", "login_status", "device_type"],
    "complaints": ["complaint_id", "customer_id", "complaint_datetime", "resolved_datetime"],
    "savings_cycles": ["account_id", "started_at", "term_months", "init_balance", "interest_rate", "early_withdrawal_rate"],
}
KEYS = {"kaggle_bank_churn": ["customer_id"], "customer_context": ["customer_id"],
        "accounts": ["account_id"], "transactions": ["transaction_id"], "login": ["login_id"],
        "complaints": ["complaint_id"], "savings_cycles": ["account_id", "started_at"]}
DATES = {"customer_context": ["relationship_start_date", "observation_start", "snapshot_boundary"],
         "accounts": ["created_at", "closed_at"], "transactions": ["transaction_datetime"],
         "login": ["sign_in_datetime"], "complaints": ["complaint_datetime", "resolved_datetime"],
         "savings_cycles": ["started_at"]}


def guard_tables(source):
    """Validate keys/types and restrict event histories to each customer's exposure."""
    tables = {name: frame.copy() for name, frame in source.items()}
    audit = {}
    for name, required in REQUIRED.items():
        if name not in tables or set(required) - set(tables[name]):
            raise ValueError(f"Missing required schema in {name}: {required}")
        frame = tables[name]
        keys = KEYS[name]
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: keys must be non-null and unique: {keys}")
        for col in DATES.get(name, []):
            frame[col] = pd.to_datetime(frame[col], format="ISO8601", errors="raise")
            if col not in ("closed_at", "resolved_datetime") and frame[col].isna().any():
                raise ValueError(f"{name}.{col}: required dates cannot be missing")
        audit[name] = {"input_rows": len(frame), "excluded_outside_observation": 0}

    cust, ctx, acc = (tables[n] for n in ("kaggle_bank_churn", "customer_context", "accounts"))
    if set(cust.customer_id) != set(ctx.customer_id):
        raise ValueError("customer_context must cover exactly the original customer population")
    if not ctx.snapshot_boundary.eq(SNAPSHOT).all():
        raise ValueError("customer_context snapshot must be 2026-10-01 exclusive")
    ctx["observation_start"] = pd.concat([ctx.relationship_start_date, ctx.observation_start], axis=1).max(axis=1).clip(lower=START)
    if ctx.observation_start.ge(SNAPSHOT).any():
        raise ValueError("Every customer must have positive observed exposure before snapshot")
    for name in ("accounts", "login", "complaints"):
        if not tables[name].customer_id.isin(cust.customer_id).all():
            raise ValueError(f"{name}: unknown customer foreign key")
    for name in ("transactions", "savings_cycles"):
        if not tables[name].account_id.isin(acc.account_id).all():
            raise ValueError(f"{name}: unknown account foreign key")
    for name, cols in {"accounts": ["balance"], "transactions": ["amount"],
                       "savings_cycles": ["init_balance", "term_months", "interest_rate", "early_withdrawal_rate"]}.items():
        for col in cols:
            values = pd.to_numeric(tables[name][col], errors="raise")
            if not np.isfinite(values).all() or values.lt(0).any():
                raise ValueError(f"{name}.{col}: values must be finite and nonnegative")
            tables[name][col] = values
    sav = tables["savings_cycles"]
    if not sav.term_months.isin([1, 3, 6]).all() or (sav.early_withdrawal_rate > sav.interest_rate).any() or sav.interest_rate.gt(1).any():
        raise ValueError("Invalid savings terms or rates")
    if not cust.churn.isin([0, 1]).all():
        raise ValueError("churn must be a complete binary label for post-fit evaluation")
    for name, col, allowed in (
        ("accounts", "account_type", ["everyday", "savings"]),
        ("transactions", "direction", ["in", "out"]),
        ("transactions", "transaction_status", ["success", "failed", "cancelled"]),
        ("transactions", "channel", ["system", "mobile_app", "branch", "atm"]),
        ("transactions", "transaction_type", ["external_transfer", "internal_transfer", "payment", "cash_deposit", "cash_withdrawal", "interest_credit", "bank_fee"]),
        ("login", "login_status", ["success", "failed"]),
        ("login", "device_type", ["ios", "android", "desktop"]),
    ):
        if not tables[name][col].isin(allowed).all():
            raise ValueError(f"{name}.{col}: unknown categorical value")
    rel = ctx.set_index("customer_id").relationship_start_date
    if (acc.created_at < acc.customer_id.map(rel)).any() or (acc.closed_at < acc.created_at).any():
        raise ValueError("Account lifecycle precedes relationship or opening")
    start_map = ctx.set_index("customer_id").observation_start
    acc = acc.loc[acc.created_at < SNAPSHOT].copy()
    acc.loc[acc.closed_at >= SNAPSHOT, "closed_at"] = pd.NaT
    tables["accounts"] = acc
    owner = acc.set_index("account_id").customer_id
    for name, date_col in (("transactions", "transaction_datetime"), ("login", "sign_in_datetime"), ("complaints", "complaint_datetime")):
        frame = tables[name]
        ids = frame.customer_id if "customer_id" in frame else frame.account_id.map(owner)
        keep = frame[date_col].ge(ids.map(start_map)) & frame[date_col].lt(SNAPSHOT)
        if name == "transactions":
            opening = frame.account_id.map(acc.set_index("account_id").created_at)
            closing = frame.account_id.map(acc.set_index("account_id").closed_at)
            keep &= frame[date_col].ge(opening) & (closing.isna() | frame[date_col].le(closing))
        tables[name] = frame.loc[keep].copy()
    cmp = tables["complaints"]
    if (cmp.resolved_datetime < cmp.complaint_datetime).any():
        raise ValueError("Complaint resolved before it was filed")
    cmp.loc[cmp.resolved_datetime >= SNAPSHOT, "resolved_datetime"] = pd.NaT
    # A cycle is visible only if started before snapshot and overlaps exposure.
    end = sav.started_at.copy()
    for term in (1, 3, 6):
        mask = sav.term_months.eq(term)
        end.loc[mask] = sav.loc[mask, "started_at"] + pd.DateOffset(months=term)
    opening = sav.account_id.map(acc.set_index("account_id").created_at)
    closing = sav.account_id.map(acc.set_index("account_id").closed_at)
    end = pd.concat([end, closing], axis=1).min(axis=1)
    observed_start = sav.account_id.map(owner).map(start_map)
    keep = sav.started_at.lt(SNAPSHOT) & sav.started_at.ge(opening) & end.gt(observed_start)
    keep &= closing.isna() | sav.started_at.le(closing)
    tables["savings_cycles"] = sav.loc[keep].copy()
    # Paired internal transfers must have one debit and one credit for one owner.
    tx = tables["transactions"]
    internal = tx.loc[tx.transaction_type.eq("internal_transfer") & tx.transaction_status.eq("success")]
    if internal.transfer_id.isna().any():
        raise ValueError("Successful internal transfers require a transfer_id")
    for _, pair in internal.groupby("transfer_id"):
        if (len(pair) != 2 or set(pair.direction) != {"in", "out"} or pair.amount.nunique() != 1
                or pair.transaction_datetime.nunique() != 1 or pair.account_id.nunique() != 2
                or pair.account_id.map(owner).nunique() != 1):
            raise ValueError("Invalid paired internal transfer")
    for name, frame in tables.items():
        audit[name]["excluded_outside_observation"] = audit[name]["input_rows"] - len(frame)
    return tables, audit


class SegmentationETL(BankDataETL):
    """Same ETL joins, guarded raw tables, and initiated-only activity denominators."""

    def _load_tables(self):
        self.tables_, self.audit_ = guard_tables(super()._load_tables())
        return {name: frame.copy() for name, frame in self.tables_.items()}

    def process(self, verbose=True):
        out = super().process(verbose=verbose)
        tx = self.tables_["transactions"].copy()
        owner = self.tables_["accounts"].set_index("account_id").customer_id
        tx["customer_id"] = tx.account_id.map(owner)
        initiated = tx.channel.ne("system") & ~tx.transaction_type.isin(["bank_fee", "interest_credit"])
        initiated &= tx.direction.eq("out") | tx.transaction_type.eq("cash_deposit")
        actions = tx.loc[initiated]
        attempts = actions.groupby("customer_id").size()
        failed = actions.loc[actions.transaction_status.eq("failed")].groupby("customer_id").size()
        posted = actions.loc[actions.transaction_status.eq("success")]
        external = posted.loc[posted.direction.eq("out") & posted.transaction_type.ne("internal_transfer")]
        out["initiated_tx_attempts"] = out.customer_id.map(attempts).fillna(0)
        out["initiated_failed_transactions"] = out.customer_id.map(failed).fillna(0)
        out["initiated_outflow"] = out.customer_id.map(external.groupby("customer_id").amount.sum()).fillna(0)
        if out.customer_id.isna().any() or out.customer_id.duplicated().any():
            raise ValueError("ETL must preserve exactly one record per customer")
        return out


def load_segmentation_data(raw_dir, verbose=True):
    raw_dir = Path(raw_dir)
    missing = [name + ".csv" for name in REQUIRED if not (raw_dir / (name + ".csv")).is_file()]
    if missing:
        raise FileNotFoundError(f"Segmentation requires raw tables in {raw_dir}: {', '.join(missing)}. "
                                "See docs/segmentation.md for the official dataset download.")
    etl = SegmentationETL(raw_dir)
    frame = etl.process(verbose=verbose)
    sources = {}
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix in (".csv", ".json"):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    manifest = raw_dir / "generation_manifest.json"
    return frame, {"tables": etl.audit_, "sources": sources,
                   "generation_manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
