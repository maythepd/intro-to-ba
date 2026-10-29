"""
ETL & Schema Ingestion Module for Digital Bank Churn Prediction.
Ingests multi-table relational data from data/full_dataset/ and performs
schema-level joins and raw aggregations per customer.

Raw source tables:
  1. kaggle_bank_churn.csv   (Demographics, baseline products, credit score, churn target)
  2. customer_context.csv     (Relationship start date, observation start, snapshot boundary)
  3. accounts.csv             (Deposit accounts, types, open/close dates, balances, renewals)
  4. savings_cycles.csv       (Savings terms, interest rates, early withdrawal rates)
  5. complaints.csv           (Customer complaints, resolution status and timestamps)
  6. login.csv                (Digital login attempts, device types, success/failure)
  7. transactions.csv         (Account-level financial transactions, channels, directions, types)

NOTE: This module strictly performs ETL and schema integration (counts, sums, raw dates, min/max).
Dynamic behavioral features (momentum ratios, collapsing flags, friction index, etc.)
are engineered downstream in DomainFeatureEngineer so that ablation flags function properly.
"""

from datetime import datetime
import os
from pathlib import Path
import time
from typing import Optional, Union
import numpy as np
import pandas as pd


class BankDataETL:
    """
    Ingests and joins multi-table relational banking events into a consolidated base customer table.
    Strictly performs schema integration and base raw aggregations per customer.
    """

    DEFAULT_SNAPSHOT_BOUNDARY = "2026-10-01"
    DEFAULT_Q3_START = "2026-07-01"

    def __init__(
        self,
        raw_data_dir: Union[str, Path] = "data/full_dataset",
        snapshot_boundary: Optional[str] = None,
        q3_start: Optional[str] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.snapshot_boundary = pd.Timestamp(snapshot_boundary or self.DEFAULT_SNAPSHOT_BOUNDARY)
        self.q3_start = pd.Timestamp(q3_start or self.DEFAULT_Q3_START)

    def _resolve_file(self, filename: str) -> Path:
        """Resolve file path relative to raw_data_dir or repository root."""
        candidate = self.raw_data_dir / filename
        if candidate.exists():
            return candidate

        repo_root = Path(__file__).resolve().parent.parent
        candidate_repo = repo_root / self.raw_data_dir / filename
        if candidate_repo.exists():
            return candidate_repo

        candidate_root = repo_root / "data" / "full_dataset" / filename
        if candidate_root.exists():
            return candidate_root

        raise FileNotFoundError(f"Required table '{filename}' not found in '{self.raw_data_dir}'.")

    def process(self, verbose: bool = True) -> pd.DataFrame:
        """
        Executes schema-level joining and raw aggregations across all 7 tables.
        Returns base customer dataframe with raw metrics (no dynamic ratios/transformations).
        """
        t0 = time.time()
        if verbose:
            print("[INFO] Starting schema-level ingestion and join across full_dataset tables...")

        # 1. Load Raw Tables
        cust = pd.read_csv(self._resolve_file("kaggle_bank_churn.csv"))
        ctx = pd.read_csv(self._resolve_file("customer_context.csv"))
        acc = pd.read_csv(self._resolve_file("accounts.csv"))
        sav = pd.read_csv(self._resolve_file("savings_cycles.csv"))
        cmp = pd.read_csv(self._resolve_file("complaints.csv"))
        log = pd.read_csv(self._resolve_file("login.csv"))
        tx = pd.read_csv(self._resolve_file("transactions.csv"))

        # 2. Customer Context & Exposure Boundaries
        obs_start = pd.to_datetime(ctx["observation_start"], format="ISO8601")
        rel_start = pd.to_datetime(ctx["relationship_start_date"], format="ISO8601")

        effective_start = np.maximum(obs_start, rel_start)
        ctx["exposure_days"] = (self.snapshot_boundary - effective_start).dt.total_seconds() / 86400.0
        ctx["exposure_months"] = np.maximum(ctx["exposure_days"] / 30.4375, 0.1)
        ctx["relationship_tenure_days"] = (self.snapshot_boundary - rel_start).dt.total_seconds() / 86400.0

        df = cust.merge(
            ctx[["customer_id", "exposure_days", "exposure_months", "relationship_tenure_days"]],
            on="customer_id",
            how="left",
        )

        # 3. Account Schema Aggregation
        acc["is_everyday"] = (acc["account_type"] == "everyday").astype(int)
        acc["is_savings"] = (acc["account_type"] == "savings").astype(int)
        acc["is_closed"] = acc["closed_at"].notna().astype(int)
        acc["is_auto_renewal"] = (acc["renewal_type"] == "auto").astype(int)
        acc["is_manual_renewal"] = (acc["renewal_type"] == "manual").astype(int)

        acc["savings_bal"] = np.where(acc["account_type"] == "savings", acc["balance"], 0.0)
        acc["everyday_bal"] = np.where(acc["account_type"] == "everyday", acc["balance"], 0.0)

        acc_feat = acc.groupby("customer_id").agg(
            total_accounts=("account_id", "count"),
            num_everyday_accounts=("is_everyday", "sum"),
            num_savings_accounts=("is_savings", "sum"),
            num_closed_accounts=("is_closed", "sum"),
            savings_balance=("savings_bal", "sum"),
            everyday_balance=("everyday_bal", "sum"),
            has_auto_renewal=("is_auto_renewal", "max"),
            has_manual_renewal=("is_manual_renewal", "max"),
        ).reset_index()

        df = df.merge(acc_feat, on="customer_id", how="left")
        for col in [
            "total_accounts",
            "num_everyday_accounts",
            "num_savings_accounts",
            "num_closed_accounts",
            "savings_balance",
            "everyday_balance",
            "has_auto_renewal",
            "has_manual_renewal",
        ]:
            df[col] = df[col].fillna(0)

        # 4. Savings Cycles Schema Aggregation
        sav_acc = sav.merge(acc[["account_id", "customer_id"]], on="account_id", how="left")
        sav_feat = sav_acc.groupby("customer_id").agg(
            num_savings_cycles=("term_months", "count"),
            avg_savings_term_months=("term_months", "mean"),
            max_savings_term_months=("term_months", "max"),
            avg_interest_rate=("interest_rate", "mean"),
            avg_early_withdrawal_rate=("early_withdrawal_rate", "mean"),
        ).reset_index()

        df = df.merge(sav_feat, on="customer_id", how="left")
        for col, default_val in {
            "num_savings_cycles": 0,
            "avg_savings_term_months": 0.0,
            "max_savings_term_months": 0,
            "avg_interest_rate": 0.0,
            "avg_early_withdrawal_rate": 0.0,
        }.items():
            df[col] = df[col].fillna(default_val)

        # 5. Complaints Schema Aggregation
        cmp["complaint_datetime"] = pd.to_datetime(cmp["complaint_datetime"], format="ISO8601")
        cmp["resolved_datetime"] = pd.to_datetime(cmp["resolved_datetime"], format="ISO8601")
        cmp["is_unresolved"] = cmp["resolved_datetime"].isna() | (cmp["resolved_datetime"] >= self.snapshot_boundary)
        cmp["is_resolved"] = (~cmp["is_unresolved"]).astype(int)
        cmp["resolution_delay_days"] = np.where(
            cmp["is_resolved"] == 1,
            (cmp["resolved_datetime"] - cmp["complaint_datetime"]).dt.total_seconds() / 86400.0,
            np.nan,
        )

        cmp_feat = cmp.groupby("customer_id").agg(
            customer_service_complaints=("complaint_id", "count"),
            unresolved_complaints_count=("is_unresolved", "sum"),
            resolved_complaints_count=("is_resolved", "sum"),
            avg_resolution_delay_days=("resolution_delay_days", "mean"),
        ).reset_index()

        last_cmp = cmp.groupby("customer_id")["complaint_datetime"].max()
        cmp_rec = (self.snapshot_boundary - last_cmp).dt.total_seconds() / 86400.0
        cmp_feat["complaint_recency_days"] = cmp_feat["customer_id"].map(cmp_rec)

        df = df.merge(cmp_feat, on="customer_id", how="left")
        df["customer_service_complaints"] = df["customer_service_complaints"].fillna(0).astype(int)
        df["unresolved_complaints_count"] = df["unresolved_complaints_count"].fillna(0).astype(int)
        df["resolved_complaints_count"] = df["resolved_complaints_count"].fillna(0).astype(int)
        # Note: avg_resolution_delay_days is deliberately kept as NaN/None for customers without resolved complaints
        # (avoiding conflating instant 0-day resolution with unresolved complaints or zero complaints).
        df["complaint_recency_days"] = df["complaint_recency_days"].fillna(df["exposure_days"])

        # 6. Digital Logins Schema Aggregation
        log["sign_in_datetime"] = pd.to_datetime(log["sign_in_datetime"], format="ISO8601")
        log["is_success"] = (log["login_status"] == "success").astype(int)
        log["is_failed"] = (log["login_status"] == "failed").astype(int)
        log["is_app"] = (log["is_success"] & log["device_type"].isin(["ios", "android"])).astype(int)
        log["is_desktop"] = (log["is_success"] & (log["device_type"] == "desktop")).astype(int)
        log["is_mobile"] = (log["is_success"] & log["device_type"].isin(["ios", "android"])).astype(int)
        # Engagement counts strictly measure successful logins per data dictionary (synth_data_dict.md: L33)
        log["is_q3"] = (log["is_success"] & (log["sign_in_datetime"] >= self.q3_start)).astype(int)
        log["is_q2"] = (log["is_success"] & (log["sign_in_datetime"] < self.q3_start)).astype(int)

        log_feat = log.groupby("customer_id").agg(
            total_logins=("login_id", "count"),
            successful_logins=("is_success", "sum"),
            failed_logins=("is_failed", "sum"),
            app_logins_total=("is_app", "sum"),
            desktop_logins_total=("is_desktop", "sum"),
            mobile_logins_total=("is_mobile", "sum"),
            login_count_q3=("is_q3", "sum"),
            login_count_q2=("is_q2", "sum"),
        ).reset_index()

        log_succ = log[log["is_success"] == 1]
        last_login = log_succ.groupby("customer_id")["sign_in_datetime"].max()
        rec_login = (self.snapshot_boundary - last_login).dt.total_seconds() / 86400.0
        log_feat = log_feat.merge(rec_login.rename("login_recency_days"), on="customer_id", how="left")

        # Dominant primary device (determined strictly from successful logins)
        dev_counts = log_succ.groupby(["customer_id", "device_type"]).size().unstack(fill_value=0)
        dev_map = {"ios": "iOS", "android": "Android", "desktop": "Desktop"}
        primary_dev_series = dev_counts.idxmax(axis=1).map(dev_map)
        log_feat = log_feat.merge(primary_dev_series.rename("primary_device"), on="customer_id", how="left")

        df = df.merge(log_feat, on="customer_id", how="left")
        for col in [
            "total_logins",
            "successful_logins",
            "failed_logins",
            "app_logins_total",
            "desktop_logins_total",
            "mobile_logins_total",
            "login_count_q3",
            "login_count_q2",
        ]:
            df[col] = df[col].fillna(0).astype(int)
        df["login_recency_days"] = df["login_recency_days"].fillna(df["exposure_days"])
        df["primary_device"] = df["primary_device"].fillna("None")

        # 7. Financial Transactions Schema Aggregation
        acc_map = acc.set_index("account_id")["customer_id"].to_dict()
        tx["customer_id"] = tx["account_id"].map(acc_map)
        tx["transaction_datetime"] = pd.to_datetime(tx["transaction_datetime"], format="ISO8601")

        tx["is_failed"] = (tx["transaction_status"] == "failed").astype(int)
        tx["is_cancelled"] = (tx["transaction_status"] == "cancelled").astype(int)
        tx["is_posted"] = (tx["transaction_status"] == "success").astype(int)

        posted_tx = tx[tx["is_posted"] == 1].copy()

        # In banking transactions, internal transfers between two accounts of the same customer
        # produce 2 linked records (1 debit 'out' and 1 credit 'in' with the same transfer_id).
        # To count distinct customer transaction actions/attempts (avoiding double-counting
        # internal transfers for the same customer), we take 1 record per internal transfer action (direction == 'out').
        action_tx = tx[(tx["transaction_type"] != "internal_transfer") | (tx["direction"] == "out")]

        # Group by customer for overall counts and attempts
        tx_all = action_tx.groupby("customer_id").agg(
            total_tx_attempts=("transaction_id", "count"),
            failed_transactions=("is_failed", "sum"),
            cancelled_transactions=("is_cancelled", "sum"),
        ).reset_index()

        # Inflow / Outflow aggregations:
        # Customer-portfolio flows exclude internal transfers between a customer's own accounts
        # (which are internal balance reallocations rather than external spending or incoming funds).
        is_ext_out = (posted_tx["direction"] == "out") & (posted_tx["transaction_type"] != "internal_transfer")
        is_ext_in = (posted_tx["direction"] == "in") & (posted_tx["transaction_type"] != "internal_transfer")

        posted_tx["outflow_amount"] = np.where(is_ext_out, posted_tx["amount"], 0.0)
        posted_tx["inflow_amount"] = np.where(is_ext_in, posted_tx["amount"], 0.0)
        posted_tx["is_q3"] = (
            (posted_tx["transaction_datetime"] >= self.q3_start) & is_ext_out
        ).astype(int)
        posted_tx["is_q2"] = (
            (posted_tx["transaction_datetime"] < self.q3_start) & is_ext_out
        ).astype(int)

        posted_tx["q3_amount"] = np.where(posted_tx["is_q3"] == 1, posted_tx["amount"], 0.0)
        posted_tx["q2_amount"] = np.where(posted_tx["is_q2"] == 1, posted_tx["amount"], 0.0)

        # Transaction types aggregations
        for t_type in ["payment", "cash_withdrawal", "cash_deposit", "bank_fee"]:
            posted_tx[f"{t_type}_amt"] = np.where(posted_tx["transaction_type"] == t_type, posted_tx["amount"], 0.0)
            posted_tx[f"{t_type}_cnt"] = (posted_tx["transaction_type"] == t_type).astype(int)

        is_ext_out_type = (posted_tx["transaction_type"] == "external_transfer") & (posted_tx["direction"] == "out")
        posted_tx["external_transfer_out_amt"] = np.where(is_ext_out_type, posted_tx["amount"], 0.0)
        posted_tx["external_transfer_out_cnt"] = is_ext_out_type.astype(int)

        # Action-level posted transactions for count
        action_posted = posted_tx[(posted_tx["transaction_type"] != "internal_transfer") | (posted_tx["direction"] == "out")]

        tx_feat = action_posted.groupby("customer_id").agg(
            total_transactions=("transaction_id", "count"),
            total_outflow=("outflow_amount", "sum"),
            total_inflow=("inflow_amount", "sum"),
            trans_amount_3m=("q3_amount", "sum"),
            trans_amount_prior_3m=("q2_amount", "sum"),
            trans_count_3m=("is_q3", "sum"),
            trans_count_prior_3m=("is_q2", "sum"),
            payment_amount=("payment_amt", "sum"),
            payment_count=("payment_cnt", "sum"),
            cash_withdrawal_amount=("cash_withdrawal_amt", "sum"),
            cash_withdrawal_count=("cash_withdrawal_cnt", "sum"),
            cash_deposit_amount=("cash_deposit_amt", "sum"),
            cash_deposit_count=("cash_deposit_cnt", "sum"),
            external_transfer_out_amount=("external_transfer_out_amt", "sum"),
            external_transfer_out_count=("external_transfer_out_cnt", "sum"),
            bank_fee_amount=("bank_fee_amt", "sum"),
            bank_fee_count=("bank_fee_cnt", "sum"),
        ).reset_index()

        # Recency: Calculated strictly on customer-initiated transactions (channel != "system"),
        # preventing automated system postings (fees, interest, passive receipts) from faking active engagement.
        cust_tx = posted_tx[posted_tx["channel"] != "system"]
        last_tx = cust_tx.groupby("customer_id")["transaction_datetime"].max()
        rec_tx = (self.snapshot_boundary - last_tx).dt.total_seconds() / 86400.0
        tx_feat = tx_feat.merge(rec_tx.rename("recency_days"), on="customer_id", how="left")

        # Preferred channel (customer-initiated channels only, excluding automated system postings)
        channel_counts = cust_tx.groupby(["customer_id", "channel"]).size().unstack(fill_value=0)
        chan_map = {
            "mobile_app": "Mobile App",
            "atm": "ATM",
            "branch": "Branch",
        }
        pref_chan_series = channel_counts.idxmax(axis=1).map(chan_map)
        tx_feat = tx_feat.merge(pref_chan_series.rename("preferred_channel"), on="customer_id", how="left")

        # Merge tx features
        tx_summary = tx_all.merge(tx_feat, on="customer_id", how="left")
        df = df.merge(tx_summary, on="customer_id", how="left")

        # Fill missing tx fields
        for col in [
            "total_tx_attempts",
            "failed_transactions",
            "cancelled_transactions",
            "total_transactions",
            "trans_count_3m",
            "trans_count_prior_3m",
            "payment_count",
            "cash_withdrawal_count",
            "cash_deposit_count",
            "external_transfer_out_count",
            "bank_fee_count",
        ]:
            df[col] = df[col].fillna(0).astype(int)

        for col in [
            "total_outflow",
            "total_inflow",
            "trans_amount_3m",
            "trans_amount_prior_3m",
            "payment_amount",
            "cash_withdrawal_amount",
            "cash_deposit_amount",
            "external_transfer_out_amount",
            "bank_fee_amount",
        ]:
            df[col] = df[col].fillna(0.0)

        df["recency_days"] = df["recency_days"].fillna(df["exposure_days"])
        df["preferred_channel"] = df["preferred_channel"].fillna("None")

        # Ensure correct column ordering and types
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        if verbose:
            elapsed = time.time() - t0
            print(f"[OK] Ingestion and join completed in {elapsed:.2f}s | Output: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    def save(self, df: pd.DataFrame, output_path: Union[str, Path], verbose: bool = True) -> str:
        """Saves joined dataframe to CSV."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"[OK] Saved joined base dataset to: {out_path.resolve()}")
        return str(out_path.resolve())


def load_or_build_dataset(
    raw_data_dir: Union[str, Path] = "data/full_dataset",
    output_path: Union[str, Path] = "data/full_bank_data.csv",
    force_rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Loads the joined full dataset from cached CSV if available,
    otherwise performs schema join from raw_data_dir and caches it.
    """
    out = Path(output_path)
    if not force_rebuild and out.exists():
        if verbose:
            print(f"[OK] Loading cached joined dataset from: {out.resolve()}")
        df = pd.read_csv(out)
        return df

    etl = BankDataETL(raw_data_dir=raw_data_dir)
    df = etl.process(verbose=verbose)
    etl.save(df, output_path=output_path, verbose=verbose)
    return df


# Backward-compatible aliases
FullDatasetProcessor = BankDataETL
load_or_build_full_dataset = load_or_build_dataset
