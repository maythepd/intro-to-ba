"""
Sin/Cos Cyclical Calendar Harmonic Embeddings.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict, List, Union
import numpy as np
import pandas as pd


def extract_cyclic_features(
    tx_df: pd.DataFrame,
    customer_ids: Union[List[int], np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Computes harmonic trigonometric day-in-month coordinates and Rayleigh concentration metrics.
    theta = 2 * pi * (day - 1) / 31.0
    """
    features: Dict[str, np.ndarray] = {}
    cust_idx = pd.Index(customer_ids)

    # Filter to success transactions
    tx = tx_df[tx_df["transaction_status"] == "success"].copy()
    if "day" not in tx.columns:
        if not pd.api.types.is_datetime64_any_dtype(tx["transaction_datetime"]):
            tx["dt"] = pd.to_datetime(tx["transaction_datetime"])
        else:
            tx["dt"] = tx["transaction_datetime"]
        tx["day"] = tx["dt"].dt.day

    theta = 2.0 * np.pi * (tx["day"] - 1.0) / 31.0
    tx["sin_day"] = np.sin(theta)
    tx["cos_day"] = np.cos(theta)

    # All transactions harmonic center & concentration
    grp_all = tx.groupby("customer_id")
    sin_all = grp_all["sin_day"].mean().reindex(cust_idx, fill_value=0.0)
    cos_all = grp_all["cos_day"].mean().reindex(cust_idx, fill_value=0.0)
    disp_all = np.sqrt(sin_all**2 + cos_all**2)

    features["quant_cyclic_tx_day_sin"] = sin_all.values
    features["quant_cyclic_tx_day_cos"] = cos_all.values
    features["quant_cyclic_tx_day_dispersion"] = disp_all.values

    # Inflow transactions (Payday / Salary cycle)
    in_tx = tx[tx["direction"] == "in"]
    grp_in = in_tx.groupby("customer_id")
    sin_in = grp_in["sin_day"].mean().reindex(cust_idx, fill_value=0.0)
    cos_in = grp_in["cos_day"].mean().reindex(cust_idx, fill_value=0.0)
    disp_in = np.sqrt(sin_in**2 + cos_in**2)

    features["quant_cyclic_inflow_day_sin"] = sin_in.values
    features["quant_cyclic_inflow_day_cos"] = cos_in.values
    features["quant_cyclic_inflow_day_dispersion"] = disp_in.values

    return features
