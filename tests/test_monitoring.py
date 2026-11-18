"""Monitoring metrics and window contracts, independent of external services."""
from copy import deepcopy
from datetime import timedelta
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from monitoring import drift, performance
from monitoring.report import render_html
from monitoring.runner import run_window
from monitoring.settings import digest, load_config, now_utc, timestamp
from monitoring.store import OutcomeConflict, Store, alerts, events, runs


def reference():
    return {"features": {"recency_days": drift.numeric_reference(np.arange(100))},
            "prediction": drift.numeric_reference(np.linspace(.01, .99, 100)),
            "performance": {"ROC_AUC": .9, "PR_AUC": .9, "Recall": .9, "Precision": .9, "F1_Score": .9},
            "decision_threshold": .5}


class DriftTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_identical_and_shifted_numeric_populations(self):
        ref = drift.numeric_reference(np.arange(1000))
        stable = drift.compare(np.arange(1000), ref, self.cfg, name="x")
        shifted = drift.compare(np.arange(1000) + 10000, ref, self.cfg, name="x")
        self.assertAlmostEqual(stable["score"], 0)
        self.assertEqual(shifted["status"], "critical")
        self.assertAlmostEqual(sum(row["current"] for row in shifted["distribution"]), 1)

    def test_constant_bins_missingness_and_new_categories(self):
        ref = drift.numeric_reference([1] * 100)
        self.assertAlmostEqual(drift.compare([1] * 100, ref, self.cfg, name="x")["score"], 0)
        missing = drift.compare([None] * 100, ref, self.cfg, name="x")
        self.assertEqual(missing["status"], "missing_data")
        self.assertEqual(missing["missing_excess"], 1)
        categorical = drift.compare(["new"] * 100, drift.categorical_reference(["old"] * 100), self.cfg, name="device")
        self.assertEqual(categorical["new_category_rate"], 1)
        self.assertAlmostEqual(categorical["score"], 1)
        many = drift.compare([f"new-{index}" for index in range(100)], drift.categorical_reference(["old"] * 100), self.cfg, name="device")
        self.assertLessEqual(len(many["distribution"]), 20)
        self.assertAlmostEqual(sum(row["current"] for row in many["distribution"]), 1)

    def test_small_samples_are_not_reported_as_stable(self):
        result = drift.evaluate(pd.DataFrame({"recency_days": [1]}), reference(), self.cfg)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["features"], [])


class WindowStoreTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.url = "sqlite:///" + (Path(self.folder.name) / "history.db").as_posix()
        self.store = Store(self.url)
        self.addCleanup(self.store.close)
        self.cfg = {**load_config(), "min_drift_samples": 10, "min_labeled_samples": 10,
                    "min_class_samples": 2, "min_quality_samples": 10}
        self.model = "a" * 64
        self.ref = reference()
        self.store.register_model(self.model, self.ref, {"name": "test"}, digest(self.ref))
        self.anchor = now_utc().replace(hour=0, minute=0, second=0) - timedelta(days=100)

    def append(self, day=0, *, shifted=False, source="study_replay", scenario="stable", invalid=0):
        rows = []
        for index in range(20):
            occurred = self.anchor + timedelta(days=day, hours=1, seconds=index)
            rows.append({"id": digest([day, index, source, scenario, self.model]), "model_id": self.model,
                "occurred_at": timestamp(occurred), "horizon_end": timestamp(occurred + timedelta(days=90)),
                "subject": digest(index), "source": source, "scenario": scenario,
                "features": {"recency_days": index * 5 + (1000 if shifted else 0)},
                "probability": .95 if index % 2 else .05, "latency_ms": 2.0, "valid": int(index >= invalid),
                "error_type": "validation_error" if index < invalid else None, "segment": None, "segment_model_id": None})
        self.store.append_events(rows)
        return rows

    def evaluate_window(self, day=0, source="study_replay", scenario="stable", **kwargs):
        return run_window(self.store, self.model, self.cfg, source=source, scenario=scenario,
            start=timestamp(self.anchor + timedelta(days=day)), end=timestamp(self.anchor + timedelta(days=day + 1)), **kwargs)

    def test_restart_idempotency_and_adjacent_drift_trigger(self):
        first = self.append(0, shifted=True)
        self.assertEqual(self.store.append_events(first), 0)
        self.assertEqual(self.evaluate_window(0)["alerts"], [])
        self.append(1, shifted=True)
        second = self.evaluate_window(1)
        self.assertEqual(second["status"], "review_required")
        self.evaluate_window(1)
        with self.store.engine.connect() as connection:
            self.assertEqual(connection.execute(select(func.count()).select_from(runs)).scalar_one(), 2)
            self.assertEqual(connection.execute(select(func.count()).select_from(alerts)).scalar_one(), 1)
        self.store.close()
        reopened = Store(self.url)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_run(second["id"])["status"], "review_required")

    def test_gap_and_source_isolation_break_the_streak(self):
        self.append(0, shifted=True); self.evaluate_window(0)
        self.append(2, shifted=True)
        self.assertEqual(self.evaluate_window(2)["alerts"], [])
        self.append(1, shifted=True, source="live", scenario="")
        self.assertEqual(self.evaluate_window(1, source="live", scenario="")["alerts"], [])

    def test_label_maturity_join_and_immutable_outcomes(self):
        rows = self.append()
        labels = [{"event_id": row["id"], "label": index % 2, "observed_at": row["horizon_end"],
                   "horizon_end": row["horizon_end"], "source": "study_snapshot"} for index, row in enumerate(rows)]
        bad = deepcopy(labels); bad[0]["observed_at"] = rows[0]["occurred_at"]
        with self.assertRaises(ValueError):
            self.store.add_outcomes(bad)
        self.assertEqual(self.evaluate_window()["performance"]["status"], "awaiting_labels")
        self.assertEqual(self.store.add_outcomes(labels), 20)
        self.assertEqual(self.store.add_outcomes(labels), 0)
        self.assertEqual(self.evaluate_window()["performance"]["status"], "ready")
        changed = deepcopy(labels); changed[0]["label"] = 1
        with self.assertRaises(OutcomeConflict):
            self.store.add_outcomes(changed)

    def test_live_rows_reject_synthetic_labels_and_future_outcomes(self):
        rows = self.append(source="live", scenario="")
        label = {"event_id": rows[0]["id"], "label": 1, "horizon_end": rows[0]["horizon_end"],
                 "observed_at": rows[0]["horizon_end"], "source": "synthetic_stress"}
        with self.assertRaises(ValueError):
            self.store.add_outcomes([label])
        label.update(source="observed", observed_at=timestamp(now_utc() + timedelta(days=1)))
        with self.assertRaises(ValueError):
            self.store.add_outcomes([label])

    def test_duplicate_customer_lookups_do_not_inflate_drift_cohort(self):
        rows = self.append()
        copies = [{**row, "id": digest([row["id"], "repeat"]), "occurred_at": timestamp(self.anchor + timedelta(hours=2))} for row in rows]
        self.store.append_events(copies)
        report = self.evaluate_window()
        self.assertEqual(report["quality"]["request_count"], 40)
        self.assertEqual(report["sample_count"], 20)

    def test_quality_counts_rejections_and_html_escapes_categories(self):
        self.append(invalid=5)
        report = self.evaluate_window()
        self.assertTrue(any(alert["rule"] == "data_quality" for alert in report["alerts"]))
        report["scenario"] = "<script>alert(1)</script>"
        self.assertNotIn("<script>", render_html(report))

    def test_performance_drop_needs_two_mature_eligible_windows(self):
        for day in range(2):
            rows = self.append(day)
            self.store.add_outcomes([{"event_id": row["id"], "label": 1 - index % 2,
                "horizon_end": row["horizon_end"], "observed_at": row["horizon_end"], "source": "synthetic_stress"}
                for index, row in enumerate(rows)])
            report = self.evaluate_window(day)
            self.assertEqual(report["performance"]["status"], "ready")
            self.assertEqual(any(a["rule"] == "performance_drop" for a in report["alerts"]), day == 1)

    def test_single_class_metrics_are_unavailable_and_future_windows_rejected(self):
        rows = self.append()
        self.store.add_outcomes([{"event_id": row["id"], "label": 0, "horizon_end": row["horizon_end"],
            "observed_at": row["horizon_end"], "source": "study_snapshot"} for row in rows])
        result = self.evaluate_window()["performance"]
        self.assertEqual(result["status"], "single_class")
        self.assertIsNone(result["metrics"]["roc_auc"])
        self.assertFalse(result["trigger_candidate"])
        with self.assertRaises(ValueError):
            self.evaluate_window(as_of=timestamp(self.anchor))


if __name__ == "__main__":
    unittest.main()
