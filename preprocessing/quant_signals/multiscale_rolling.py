"""
Multi-Scale High-Resolution Rolling Window Feature Extraction.
Computes fine-grained behavioral signals across 7d, 14d, 30d, 60d, and 90d windows.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd


def extract_multiscale_rolling_features(
    raw_data_dir: Union[str, Path],
    customer_ids: Union[List[int], np.ndarray],
    ch_net_flow: Optional[np.ndarray] = None,
    ch_tx_count: Optional[np.ndarray] = None,
    ch_login_active: Optional[np.ndarray] = None,
    cutoff_date: str = "2026-10-01",
) -> Dict[str, np.ndarray]:
    """
    Computes high-resolution rolling window metrics over recent intervals [7d, 14d, 30d, 60d, 90d]
    and cross-scale momentum/decay ratios.
    """
    features: Dict[str, np.ndarray] = {}
    cust_idx = pd.Index(customer_ids)
    raw_path = Path(raw_data_dir)
    cutoff = pd.Timestamp(cutoff_date)

    # 1. 3-Month Longitudinal Rolling Metrics (from 6-month matrices if provided)
    if ch_net_flow is not None and ch_tx_count is not None and ch_login_active is not None:
        w3_nf = ch_net_flow[:, 3:]  # Months 4, 5, 6
        w3_tx = ch_tx_count[:, 3:]
        w3_log = ch_login_active[:, 3:]

        features["quant_rolling_mean_3m_net_flow"] = np.mean(w3_nf, axis=1)
        features["quant_rolling_std_3m_net_flow"] = np.std(w3_nf, axis=1)
        features["quant_rolling_slope_3m_net_flow"] = (w3_nf[:, 2] - w3_nf[:, 0]) / 2.0
        features["quant_rolling_min_3m_net_flow"] = np.min(w3_nf, axis=1)

        features["quant_rolling_mean_3m_tx_count"] = np.mean(w3_tx, axis=1)
        features["quant_rolling_slope_3m_tx_count"] = (w3_tx[:, 2] - w3_tx[:, 0]) / 2.0

        features["quant_rolling_mean_3m_login_days"] = np.mean(w3_log, axis=1)
        features["quant_rolling_slope_3m_login_days"] = (w3_log[:, 2] - w3_log[:, 0]) / 2.0

    # 2. Fine-grained Event-Level Windows from transactions.csv and login.csv
    tx_csv = raw_path / "transactions.csv"
    acc_csv = raw_path / "accounts.csv"
    log_csv = raw_path / "login.csv"

    if tx_csv.exists() and acc_csv.exists():
        acc = pd.read_csv(acc_csv, usecols=["account_id", "customer_id"])
        tx = pd.read_csv(
            tx_csv,
            usecols=[
                "account_id",
                "transaction_datetime",
                "amount",
                "direction",
                "transaction_status",
            ],
        )
        tx = tx[tx["transaction_status"] == "success"].merge(acc, on="account_id", how="inner")
        tx["dt"] = pd.to_datetime(tx["transaction_datetime"], format="ISO8601")
        tx["days_to_cutoff"] = (cutoff - tx["dt"]).dt.total_seconds() / 86400.0

        # Multi-scale windows for transactions
        windows = [7, 14, 30, 60, 90]
        tx_agg: Dict[str, pd.Series] = {}

        for w in windows:
            sub = tx[tx["days_to_cutoff"] <= w]
            tx_agg[f"tx_count_{w}d"] = sub.groupby("customer_id")["amount"].count().reindex(cust_idx, fill_value=0.0)
            tx_agg[f"tx_amount_{w}d"] = sub.groupby("customer_id")["amount"].sum().reindex(cust_idx, fill_value=0.0)

            in_sub = sub[sub["direction"] == "in"]
            out_sub = sub[sub["direction"] == "out"]
            in_sum = in_sub.groupby("customer_id")["amount"].sum().reindex(cust_idx, fill_value=0.0)
            out_sum = out_sub.groupby("customer_id")["amount"].sum().reindex(cust_idx, fill_value=0.0)

            tx_agg[f"inflow_{w}d"] = in_sum
            tx_agg[f"outflow_{w}d"] = out_sum
            tx_agg[f"net_flow_{w}d"] = in_sum - out_sum

            features[f"quant_multiscale_tx_count_{w}d"] = tx_agg[f"tx_count_{w}d"].values
            features[f"quant_multiscale_tx_amount_{w}d"] = tx_agg[f"tx_amount_{w}d"].values
            features[f"quant_multiscale_inflow_{w}d"] = in_sum.values
            features[f"quant_multiscale_outflow_{w}d"] = out_sum.values
            features[f"quant_multiscale_net_flow_{w}d"] = (in_sum - out_sum).values

        # Dynamic cross-scale ratios (Velocity & Activity drop indicators)
        c7 = tx_agg["tx_count_7d"].values
        c14 = tx_agg["tx_count_14d"].values
        c30 = tx_agg["tx_count_30d"].values
        c90 = tx_agg["tx_count_90d"].values

        a7 = tx_agg["tx_amount_7d"].values
        a14 = tx_agg["tx_amount_14d"].values
        a30 = tx_agg["tx_amount_30d"].values
        a90 = tx_agg["tx_amount_90d"].values

        out7 = tx_agg["outflow_7d"].values
        out30 = tx_agg["outflow_30d"].values

        nf7 = tx_agg["net_flow_7d"].values
        nf14 = tx_agg["net_flow_14d"].values

        # Standardized pace ratios: compare recent week/fortnight against monthly average rate
        features["quant_multiscale_ratio_tx_count_7d_to_30d"] = (c7 * (30.0 / 7.0)) / (c30 + 1.0)
        features["quant_multiscale_ratio_tx_count_14d_to_30d"] = (c14 * (30.0 / 14.0)) / (c30 + 1.0)
        features["quant_multiscale_ratio_tx_amount_7d_to_30d"] = (a7 * (30.0 / 7.0)) / (a30 + 1.0)
        features["quant_multiscale_ratio_tx_amount_14d_to_30d"] = (a14 * (30.0 / 14.0)) / (a30 + 1.0)
        features["quant_multiscale_ratio_tx_amount_30d_to_90d"] = (a30 * 3.0) / (a90 + 1.0)

        # Sudden outflow concentration in final week
        features["quant_multiscale_outflow_share_7d_of_30d"] = out7 / (out30 + 1.0)

        # Recent net flow velocity & acceleration
        features["quant_multiscale_net_flow_velocity_7d_vs_14d"] = nf7 - (nf14 - nf7)

    # 3. Multi-scale login engagement from login.csv
    if log_csv.exists():
        log = pd.read_csv(
            log_csv,
            usecols=["customer_id", "sign_in_datetime", "login_status"],
        )
        log["dt"] = pd.to_datetime(log["sign_in_datetime"], format="ISO8601")
        log["days_to_cutoff"] = (cutoff - log["dt"]).dt.total_seconds() / 86400.0

        for w in [7, 14, 30, 60, 90]:
            sub_log = log[log["days_to_cutoff"] <= w]
            l_cnt = sub_log.groupby("customer_id")["login_status"].count().reindex(cust_idx, fill_value=0.0)
            features[f"quant_multiscale_login_count_{w}d"] = l_cnt.values

            if w in [7, 14, 30]:
                fail_cnt = (
                    sub_log[sub_log["login_status"] != "success"]
                    .groupby("customer_id")["login_status"]
                    .count()
                    .reindex(cust_idx, fill_value=0.0)
                )
                features[f"quant_multiscale_login_failed_count_{w}d"] = fail_cnt.values

        # Login drop-off ratio: 7d vs 30d pace
        l7 = features["quant_multiscale_login_count_7d"]
        l14 = features["quant_multiscale_login_count_14d"]
        l30 = features["quant_multiscale_login_count_30d"]
        features["quant_multiscale_ratio_login_7d_to_30d"] = (l7 * (30.0 / 7.0)) / (l30 + 1.0)
        features["quant_multiscale_ratio_login_14d_to_30d"] = (l14 * (30.0 / 14.0)) / (l30 + 1.0)

    return features
