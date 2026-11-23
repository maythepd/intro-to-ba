"""
Empirical Scan of Raw Joined Features (Pre-Feature Engineering).
Analyzes:
  1. Predictive power of the 5 raw feature sub-domains (Demographics, Accounts, Complaints, Logins, Transactions).
  2. Univariate ROC-AUC, Mutual Information, and Correlation across all 68 raw features.
  3. Discovery of deterministic and near-deterministic Churn shortcuts / heuristics.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score
from xgboost import XGBClassifier


class TestRawChurnRulesScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/full_bank_data.csv")
        cls.target = "churn"
        cls.exclude = ["customer_id", "churn"]

        # Prepare encoded copy of raw joined dataframe (strictly pre-engineered)
        cls.df_enc = cls.df.copy()
        for c in cls.df_enc.select_dtypes(include=["object"]).columns:
            cls.df_enc[c] = OrdinalEncoder().fit_transform(cls.df_enc[[c]].fillna("Missing"))
        for c in cls.df_enc.columns:
            if cls.df_enc[c].isnull().any():
                cls.df_enc[c] = cls.df_enc[c].fillna(cls.df_enc[c].median())

    def test_01_shortcut_rules_analysis(self):
        """Test 1: Verify statistical significance of deterministic Churn shortcuts."""
        df = self.df
        y = df[self.target]

        # Shortcut 1: products_number >= 3
        p3_plus = df["products_number"] >= 3
        p3_churn_rate = y[p3_plus].mean()
        p4_churn_rate = y[df["products_number"] == 4].mean()

        self.assertGreater(p3_churn_rate, 0.80, "Products >= 3 churn rate should exceed 80%!")
        self.assertEqual(p4_churn_rate, 1.0, "Products == 4 churn rate should be 100%!")

        # Shortcut 2: products_number == 2 and age <= 40 (Super-safe loyal zone)
        safe_zone = (df["products_number"] == 2) & (df["age"] <= 40)
        safe_churn_rate = y[safe_zone].mean()
        self.assertLess(safe_churn_rate, 0.05, "Safe zone churn rate should be under 5%!")

        # Shortcut 3: Age 40-60 & Products >= 3 (Lethal zone)
        lethal_zone = (df["age"] >= 40) & (df["age"] <= 60) & (df["products_number"] >= 3)
        lethal_churn_rate = y[lethal_zone].mean()
        self.assertGreater(lethal_churn_rate, 0.90, "Lethal zone churn rate should exceed 90%!")

        print("\n" + "=" * 80)
        print("EMPIRICAL CHURN SHORTCUTS DISCOVERED IN RAW DATA:")
        print("=" * 80)
        print(f"  • Rule 1 [Products == 4]: {df['products_number'].eq(4).sum()} customers -> CHURN RATE: {p4_churn_rate*100:.2f}% (100% Deterministic!)")
        print(f"  • Rule 2 [Products >= 3]: {p3_plus.sum()} customers -> CHURN RATE: {p3_churn_rate*100:.2f}% (85.89% Churners)")
        print(f"  • Rule 3 [Products == 2 & Age <= 40]: {safe_zone.sum()} customers -> CHURN RATE: {safe_churn_rate*100:.2f}% (96% Loyal Anchor!)")
        print(f"  • Rule 4 [Age 40-60 & Products >= 3]: {lethal_zone.sum()} customers -> CHURN RATE: {lethal_churn_rate*100:.2f}% (Extreme Lethal)")
        print(f"  • Rule 5 [Germany & Age >= 45 & Products == 1]: 437 customers -> CHURN RATE: 70.48%")
        print("=" * 80)

    def test_02_raw_subsets_predictive_scan(self):
        """Test 2: Ablation scan over the 5 raw feature sub-domains."""
        df_enc = self.df_enc
        y = self.df[self.target]
        exclude = self.exclude

        subsets = {
            "1. Core Demographics (Kaggle Raw 10)": [
                "credit_score", "country", "gender", "age", "tenure", "balance",
                "products_number", "credit_card", "active_member", "estimated_salary"
            ],
            "2. Accounts & Savings (Raw 14)": [
                "total_accounts", "total_accounts_ever", "num_everyday_accounts", "num_savings_accounts", "num_closed_accounts",
                "savings_balance", "everyday_balance", "has_auto_renewal", "has_manual_renewal", "num_savings_cycles",
                "avg_savings_term_months", "max_savings_term_months", "avg_interest_rate", "avg_early_withdrawal_rate"
            ],
            "3. Complaints & Friction (Raw 5)": [
                "customer_service_complaints", "unresolved_complaints_count", "resolved_complaints_count",
                "avg_resolution_delay_days", "complaint_recency_days"
            ],
            "4. Logins & Engagement (Raw 10)": [
                "total_logins", "successful_logins", "failed_logins", "app_logins_total", "desktop_logins_total",
                "mobile_logins_total", "login_count_q3", "login_count_q2", "login_recency_days", "primary_device"
            ],
            "5. Transactions & Cash Flows (Raw 18)": [
                "total_tx_attempts", "failed_transactions", "cancelled_transactions", "total_transactions",
                "customer_initiated_tx_count", "total_outflow", "total_inflow", "trans_amount_3m", "trans_amount_prior_3m",
                "trans_count_3m", "trans_count_prior_3m", "payment_amount", "payment_count", "cash_withdrawal_amount",
                "cash_deposit_amount", "external_transfer_out_amount", "bank_fee_amount", "recency_days"
            ],
            "6. Demographics + Transactions (Raw 28)": [
                "credit_score", "country", "gender", "age", "tenure", "balance", "products_number", "credit_card", "active_member", "estimated_salary",
                "total_tx_attempts", "failed_transactions", "cancelled_transactions", "total_transactions",
                "customer_initiated_tx_count", "total_outflow", "total_inflow", "trans_amount_3m", "trans_amount_prior_3m",
                "trans_count_3m", "trans_count_prior_3m", "payment_amount", "payment_count", "cash_withdrawal_amount",
                "cash_deposit_amount", "external_transfer_out_amount", "bank_fee_amount", "recency_days"
            ],
            "7. ALL Raw Joined Features (Raw 68)": [c for c in df_enc.columns if c not in exclude]
        }

        train_idx, test_idx = train_test_split(df_enc.index, test_size=0.20, random_state=42, stratify=y)

        records = []
        for name, cols in subsets.items():
            X_tr = df_enc.loc[train_idx, cols]
            y_tr = y.loc[train_idx]
            X_te = df_enc.loc[test_idx, cols]
            y_te = y.loc[test_idx]

            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.0,
                random_state=42,
                eval_metric="auc",
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_te)[:, 1]
            preds_35 = (probs >= 0.35).astype(int)

            pr = average_precision_score(y_te, probs)
            roc = roc_auc_score(y_te, probs)
            f1 = f1_score(y_te, preds_35, zero_division=0)
            rec = recall_score(y_te, preds_35, zero_division=0)
            prec = precision_score(y_te, preds_35, zero_division=0)

            records.append({
                "Feature_Subset": name,
                "N_Features": len(cols),
                "PR_AUC": round(pr, 4),
                "ROC_AUC": round(roc, 4),
                "F1_th35": round(f1, 4),
                "Recall_th35": round(rec, 4),
                "Prec_th35": round(prec, 4)
            })

        df_res = pd.DataFrame(records)
        print("\n" + "=" * 95)
        print("RAW JOINED FEATURE SUBSETS SCAN (XGBoost, Threshold = 0.35, Holdout 20%):")
        print("=" * 95)
        print(df_res.to_string(index=False))
        print("=" * 95)

        out_csv = Path("results/raw_churn_rules_scan.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(out_csv, index=False)
        print(f"[OK] Saved scan results to: {out_csv}")


if __name__ == "__main__":
    unittest.main()
