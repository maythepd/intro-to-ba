"""
Login Behavioral Extractor.
Extracts 15 longitudinal channels capturing digital banking habits, session intensity,
circadian rhythms, weekend patterns, device diversity & entropy, and authentication friction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseSubExtractor


class LoginSnapshotExtractor(BaseSubExtractor):
    """
    Extracts multi-snapshot digital login dynamics from raw event logs.
    """

    FEATURE_NAMES: List[str] = [
        "login_count_log",
        "login_active_days",
        "login_daily_intensity",
        "login_breadth_ratio",
        "login_inactivity_days",
        "login_weekend_share",
        "login_night_share",
        "login_business_hours_share",
        "login_intra_month_decay",
        "login_desktop_share",
        "login_device_diversity",
        "login_device_entropy",
        "login_failed_count_log",
        "login_fail_rate",
        "login_lockout_burst_flag",
    ]

    LEGACY_FEATURE_NAMES: List[str] = [
        "login_count_log",
        "login_fail_rate",
        "login_mobile_share",
    ]

    MONTH_DAYS: Dict[int, int] = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30}

    def extract_legacy(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extracts the 3 original legacy digital login channels."""
        if context is None:
            context = {}
        full_tensor, _ = self.extract(customer_ids, months, context)
        c1 = full_tensor[:, :, 0]   # login_count_log
        c2 = full_tensor[:, :, 13]  # login_fail_rate
        c3 = context["login_mobile_share"]  # login_mobile_share
        legacy_tensor = np.stack([c1, c2, c3], axis=-1).astype(np.float32)
        return legacy_tensor, self.LEGACY_FEATURE_NAMES

    def extract(
        self,
        customer_ids: np.ndarray,
        months: List[int],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        idx = customer_ids
        if context is None:
            context = {}

        log_path = self._resolve_file("login.csv")
        log_df = pd.read_csv(
            log_path,
            usecols=["customer_id", "sign_in_datetime", "login_status", "device_type"],
        )

        dt_str = log_df["sign_in_datetime"]
        log_df["month"] = (
            pd.to_numeric(dt_str.str.slice(5, 7), errors="coerce").fillna(0).astype(int)
        )
        log_df["day"] = (
            pd.to_numeric(dt_str.str.slice(8, 10), errors="coerce").fillna(0).astype(int)
        )
        log_df["hour"] = (
            pd.to_numeric(dt_str.str.slice(11, 13), errors="coerce").fillna(0).astype(int)
        )
        log_df["date"] = dt_str.str.slice(0, 10)

        # 1. Total login count
        log_cnt_df = (
            log_df.groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        log_cnt = log_cnt_df.values.astype(np.float32)
        context["log_cnt_raw"] = log_cnt

        # 2. Active login days
        active_days_df = (
            log_df.groupby(["customer_id", "month"])["date"]
            .nunique()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        active_days = active_days_df.values.astype(np.float32)

        # 3. Daily intensity (logins per active day)
        daily_intensity = self.safe_div(log_cnt, active_days, fill_val=0.0)

        # 4. Habit breadth ratio (active days / days in month)
        days_in_month_arr = np.array([self.MONTH_DAYS.get(m, 30) for m in months], dtype=np.float32).reshape(1, -1)
        breadth_ratio = self.safe_clip(active_days / days_in_month_arr, 0.0, 1.0)

        # 5. Inactivity days in month (days in month - active days)
        inactivity_days = np.maximum(days_in_month_arr - active_days, 0.0).astype(np.float32)

        # 6. Weekend share
        # Approximate weekend via day of month or vectorized dayofweek
        # We can parse date accurately or compute weekday
        dt_dates = pd.to_datetime(log_df["date"], errors="coerce")
        log_df["is_weekend"] = dt_dates.dt.dayofweek.isin([5, 6]).astype(int)
        wk_df = (
            log_df[log_df["is_weekend"] == 1]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        weekend_share = self.safe_clip(self.safe_div(wk_df.values.astype(np.float32), log_cnt), 0.0, 1.0)

        # 7. Night share (22:00 to 05:00)
        log_df["is_night"] = log_df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        night_df = (
            log_df[log_df["is_night"] == 1]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        night_share = self.safe_clip(self.safe_div(night_df.values.astype(np.float32), log_cnt), 0.0, 1.0)

        # 8. Business hours share (08:00 to 17:00 on weekdays)
        log_df["is_business"] = (log_df["hour"].isin(range(8, 18))) & (log_df["is_weekend"] == 0)
        biz_df = (
            log_df[log_df["is_business"]]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        biz_share = self.safe_clip(self.safe_div(biz_df.values.astype(np.float32), log_cnt), 0.0, 1.0)

        # 9. Intra-month decay (Day 16-31 vs Day 1-15)
        h1_df = (
            log_df[log_df["day"] <= 15]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        h2_df = (
            log_df[log_df["day"] > 15]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        decay_ratio = self.safe_clip((h2_df.values.astype(np.float32) + 1.0) / (h1_df.values.astype(np.float32) + 1.0), 0.0, 10.0)

        # 10. Desktop share
        desk_df = (
            log_df[log_df["device_type"] == "desktop"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        desktop_share = self.safe_clip(self.safe_div(desk_df.values.astype(np.float32), log_cnt), 0.0, 1.0)

        # 11. Device diversity count (1, 2, or 3)
        dev_cnt_df = (
            log_df.groupby(["customer_id", "month"])["device_type"]
            .nunique()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        device_diversity = dev_cnt_df.values.astype(np.float32)

        # 12. Device Shannon Entropy over (android, ios, desktop)
        android_df = (
            log_df[log_df["device_type"] == "android"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        ).values.astype(np.float32)

        ios_df = (
            log_df[log_df["device_type"] == "ios"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        ).values.astype(np.float32)

        desk_vals = desk_df.values.astype(np.float32)
        p_and = self.safe_div(android_df, log_cnt)
        p_ios = self.safe_div(ios_df, log_cnt)
        p_desk = self.safe_div(desk_vals, log_cnt)

        def shannon_p(p):
            return np.where(p > 1e-6, -p * np.log(p + 1e-12), 0.0)

        entropy_raw = shannon_p(p_and) + shannon_p(p_ios) + shannon_p(p_desk)
        # Normalize by log(3) so entropy is strictly in [0, 1]
        device_entropy = self.safe_clip(entropy_raw / np.log(3.0), 0.0, 1.0)
        context["login_mobile_share"] = self.safe_clip(p_and + p_ios, 0.0, 1.0)

        # 13. Failed logins count log
        fail_df = (
            log_df[log_df["login_status"] != "success"]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        fail_cnt = fail_df.values.astype(np.float32)
        failed_count_log = self.safe_log1p(fail_cnt)
        context["fail_cnt_raw"] = fail_cnt

        # 14. Failed login rate
        fail_rate = self.safe_clip(self.safe_div(fail_cnt, log_cnt), 0.0, 1.0)

        # 15. Lockout burst flag (>= 3 failed logins on any single day)
        fail_events = log_df[log_df["login_status"] != "success"]
        day_burst = (
            fail_events.groupby(["customer_id", "month", "date"])
            .size()
            .reset_index(name="day_fails")
        )
        max_daily_fails = (
            day_burst[day_burst["day_fails"] >= 3]
            .groupby(["customer_id", "month"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(index=idx, columns=months, fill_value=0.0)
        )
        lockout_burst_flag = (max_daily_fails.values > 0).astype(np.float32)
        context["lockout_burst_flag"] = lockout_burst_flag

        # Assemble channels
        c1 = self.safe_log1p(log_cnt)
        c2 = active_days
        c3 = daily_intensity
        c4 = breadth_ratio
        c5 = inactivity_days
        c6 = weekend_share
        c7 = night_share
        c8 = biz_share
        c9 = decay_ratio
        c10 = desktop_share
        c11 = device_diversity
        c12 = device_entropy
        c13 = failed_count_log
        c14 = fail_rate
        c15 = lockout_burst_flag

        channels = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
        tensor_slice = np.stack(channels, axis=-1).astype(np.float32)

        return tensor_slice, self.FEATURE_NAMES
