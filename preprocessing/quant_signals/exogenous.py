"""
Exogenous and High-Frequency Event Signal Extraction.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def extract_exogenous_features(
    df_bank: Optional[pd.DataFrame],
    tx_df: Optional[pd.DataFrame],
    customer_ids: Union[List[int], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Extracts macroeconomic rate spreads, fee friction, and event-level burstiness."""
    features: Dict[str, np.ndarray] = {}
    cust_idx = pd.Index(customer_ids)

    # 1. Macro and Portfolio Signals from df_bank
    if df_bank is not None:
        bank_map = df_bank.set_index("customer_id") if "customer_id" in df_bank.columns else df_bank
        idx = cust_idx.intersection(bank_map.index)

        pos_rates = bank_map.loc[bank_map["avg_interest_rate"] > 0, "avg_interest_rate"]
        med_rate = float(pos_rates.median()) if not pos_rates.empty else 0.02

        rate_spread = (bank_map.loc[idx, "avg_interest_rate"] - med_rate).reindex(cust_idx, fill_value=0.0)
        friction = (
            bank_map.loc[idx, "avg_early_withdrawal_rate"]
            / (bank_map.loc[idx, "avg_interest_rate"] + 1e-4)
        ).reindex(cust_idx, fill_value=0.0)
        fee_pressure = (
            bank_map.loc[idx, "bank_fee_amount"]
            / (bank_map.loc[idx, "total_outflow"] + 1.0)
        ).reindex(cust_idx, fill_value=0.0)
        q3_to_q2 = np.log1p(
            bank_map.loc[idx, "trans_amount_3m"]
            / (bank_map.loc[idx, "trans_amount_prior_3m"] + 1.0)
        ).reindex(cust_idx, fill_value=0.0)

        features["quant_exogenous_savings_rate_spread"] = rate_spread.values
        features["quant_exogenous_early_withdrawal_friction"] = friction.values
        features["quant_exogenous_fee_pressure_ratio"] = fee_pressure.values
        features["quant_exogenous_q3_to_q2_activity_ratio"] = q3_to_q2.values

    # 2. Event-level Signals from tx_df
    if tx_df is not None and not tx_df.empty:
        tx = tx_df[tx_df["transaction_status"] == "success"].copy()
        if "day" not in tx.columns or "dow" not in tx.columns:
            if not pd.api.types.is_datetime64_any_dtype(tx["transaction_datetime"]):
                tx["dt"] = pd.to_datetime(tx["transaction_datetime"])
            else:
                tx["dt"] = tx["transaction_datetime"]
            tx["day"] = tx["dt"].dt.day
            tx["dow"] = tx["dt"].dt.dayofweek

        tx["is_weekend"] = (tx["dow"] >= 5).astype(float)
        tx["is_payday"] = ((tx["day"] >= 25) & (tx["day"] <= 31)).astype(float)
        tx["is_month_end"] = (tx["day"] >= 26).astype(float)

        in_mask = tx["direction"] == "in"
        payday_inflows = tx[in_mask].groupby("customer_id")["is_payday"].mean().reindex(cust_idx, fill_value=0.0)
        weekend_tx_share = tx.groupby("customer_id")["is_weekend"].mean().reindex(cust_idx, fill_value=0.0)
        out_mask = tx["direction"] == "out"
        month_end_withdraw = tx[out_mask].groupby("customer_id")["is_month_end"].mean().reindex(cust_idx, fill_value=0.0)

        # Inter-event Jitter and Burstiness
        tx = tx.sort_values(["customer_id", "dt"])
        tx["delta_hours"] = tx.groupby("customer_id")["dt"].diff().dt.total_seconds() / 3600.0
        inter_mean = tx.groupby("customer_id")["delta_hours"].mean().reindex(cust_idx, fill_value=0.0)
        inter_std = tx.groupby("customer_id")["delta_hours"].std().reindex(cust_idx, fill_value=0.0).fillna(0.0)
        burstiness = (inter_std / (inter_mean + 1.0)).clip(0, 10.0)

        features["quant_exogenous_payday_inflow_share"] = payday_inflows.values
        features["quant_exogenous_weekend_tx_share"] = weekend_tx_share.values
        features["quant_exogenous_month_end_cash_drain"] = month_end_withdraw.values
        features["quant_volatility_inter_tx_hours_mean"] = np.log1p(inter_mean.clip(0, 1000).values)
        features["quant_volatility_inter_tx_hours_std"] = np.log1p(inter_std.clip(0, 1000).values)
        features["quant_volatility_inter_tx_burstiness"] = burstiness.values

    return features
