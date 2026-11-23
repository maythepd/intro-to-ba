"""
Multiplicative Cross-Feature Interaction Signals.
Exploiting empirically discovered churn decision boundaries.
Pure Classical ML (Zero Deep Learning).
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def extract_interaction_features(
    df_bank: pd.DataFrame,
    customer_ids: Union[List[int], np.ndarray],
    q3_to_q2_ratio: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Computes cross-product multiplicative terms between key risk drivers:
      - Age x Products
      - Germany x Balance
      - Inactive x Age / Products
      - Germany x Single Product x Age >= 45
    """
    features: Dict[str, np.ndarray] = {}
    cust_idx = pd.Index(customer_ids)
    bank_map = df_bank.set_index("customer_id") if "customer_id" in df_bank.columns else df_bank
    idx = cust_idx.intersection(bank_map.index)

    b_age = bank_map.loc[idx, "age"].reindex(cust_idx, fill_value=40.0).values
    b_prod = bank_map.loc[idx, "products_number"].reindex(cust_idx, fill_value=1.0).values
    b_act = bank_map.loc[idx, "active_member"].reindex(cust_idx, fill_value=0.0).values
    b_bal = bank_map.loc[idx, "balance"].reindex(cust_idx, fill_value=0.0).values
    b_cntry = bank_map.loc[idx, "country"].reindex(cust_idx, fill_value="France").values
    b_rec = bank_map.loc[idx, "recency_days"].reindex(cust_idx, fill_value=90.0).values
    b_cmp = bank_map.loc[idx, "customer_service_complaints"].reindex(cust_idx, fill_value=0.0).values
    is_germany = (b_cntry == "Germany").astype(float)

    features["quant_inter_age_x_products"] = b_age * b_prod
    features["quant_inter_age_x_danger_products"] = b_age * (b_prod >= 3).astype(float)
    features["quant_inter_age_x_inactive"] = b_age * (1.0 - b_act)
    features["quant_inter_germany_x_balance"] = is_germany * b_bal
    features["quant_inter_germany_x_age"] = is_germany * b_age
    features["quant_inter_germany_x_danger_single"] = (
        is_germany * (b_prod == 1).astype(float) * (b_age >= 45).astype(float)
    )
    features["quant_inter_products_x_inactive"] = b_prod * (1.0 - b_act)

    if q3_to_q2_ratio is not None and len(q3_to_q2_ratio) == len(cust_idx):
        features["quant_inter_balance_x_q3_drop"] = b_bal * q3_to_q2_ratio
    else:
        q3_q2 = np.log1p(
            bank_map.loc[idx, "trans_amount_3m"]
            / (bank_map.loc[idx, "trans_amount_prior_3m"] + 1.0)
        ).reindex(cust_idx, fill_value=0.0).values
        features["quant_inter_balance_x_q3_drop"] = b_bal * q3_q2

    features["quant_inter_recency_x_inactive"] = b_rec * (1.0 - b_act)
    features["quant_inter_complaints_x_balance"] = b_cmp * b_bal

    return features
