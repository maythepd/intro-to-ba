"""Runs against the dedicated disposable PostgreSQL service in Monitoring CI."""
from datetime import timedelta
import os
import unittest
from uuid import uuid4

from sqlalchemy import text

from monitoring.settings import digest, now_utc, timestamp
from monitoring.store import Store


@unittest.skipUnless(os.getenv("MONITORING_TEST_DATABASE_URL"), "Dedicated PostgreSQL test URL is not set")
class PostgreSQLStoreTest(unittest.TestCase):
    def test_private_schema_runtime_role_json_and_persistence(self):
        url = os.environ["MONITORING_TEST_DATABASE_URL"]
        store = Store(url, migrate=True)
        self.addCleanup(store.close)
        model_id = digest(uuid4().hex)
        store.register_model(model_id, {"features":{}}, {"name":"CI fixture"}, digest({"features":{}}))
        with store.engine.connect() as connection:
            self.assertEqual(connection.execute(text("SELECT current_user")).scalar_one(), "bank_monitoring_writer")
        date = now_utc() - timedelta(days=100)
        row = {"id":digest([model_id,"event"]), "model_id":model_id, "occurred_at":timestamp(date),
            "horizon_end":timestamp(date+timedelta(days=90)), "subject":digest("fixture"), "source":"study_replay",
            "scenario":"ci", "features":{"age":40}, "probability":.8, "latency_ms":1.5,
            "valid":1, "error_type":None, "segment":None, "segment_model_id":None}
        self.assertEqual(store.append_events([row]), 1)
        self.assertEqual(store.append_events([row]), 0)
        store.close()
        reopened = Store(url)
        self.addCleanup(reopened.close)
        found = reopened.window_events(model_id, "study_replay", "ci", timestamp(date-timedelta(seconds=1)), timestamp(date+timedelta(seconds=1)), 100)
        self.assertEqual(found[0]["features"], {"age":40})


if __name__ == "__main__":
    unittest.main()
