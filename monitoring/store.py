"""Transactional PostgreSQL/SQLite storage. PostgreSQL tables live in a private schema."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from sqlalchemy import (JSON, CheckConstraint, Column, Float, ForeignKey, Index, Integer,
                        MetaData, String, Table, Text, UniqueConstraint, create_engine,
                        delete, event, select, text, update)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.pool import StaticPool
from sqlalchemy.schema import CreateSchema

from monitoring.settings import SOURCES, now_utc, timestamp

metadata = MetaData(schema="bank_monitoring")
models = Table("models", metadata,
    Column("id", String(64), primary_key=True), Column("created_at", String(32), nullable=False),
    Column("reference_sha256", String(64), nullable=False), Column("reference", JSON, nullable=False),
    Column("details", JSON, nullable=False))
events = Table("prediction_events", metadata,
    Column("id", String(64), primary_key=True),
    Column("model_id", String(64), ForeignKey(models.c.id), nullable=False),
    Column("occurred_at", String(32), nullable=False), Column("horizon_end", String(32), nullable=False),
    Column("subject", String(64), nullable=False), Column("source", String(24), nullable=False),
    Column("scenario", String(60), nullable=False), Column("features", JSON, nullable=False),
    Column("probability", Float), Column("latency_ms", Float, nullable=False),
    Column("valid", Integer, nullable=False), Column("error_type", String(60)),
    Column("segment", String(30)), Column("segment_model_id", String(64)),
    CheckConstraint("valid IN (0, 1)", name="event_valid"),
    CheckConstraint("probability IS NULL OR (probability >= 0 AND probability <= 1)", name="event_probability"))
Index("events_scope_time", events.c.model_id, events.c.source, events.c.scenario, events.c.occurred_at)
outcomes = Table("outcomes", metadata,
    Column("event_id", String(64), ForeignKey(events.c.id, ondelete="CASCADE"), primary_key=True),
    Column("label", Integer, nullable=False), Column("observed_at", String(32), nullable=False),
    Column("horizon_end", String(32), nullable=False), Column("source", String(30), nullable=False),
    CheckConstraint("label IN (0, 1)", name="outcome_binary"))
runs = Table("runs", metadata,
    Column("id", String(64), primary_key=True), Column("model_id", String(64), ForeignKey(models.c.id), nullable=False),
    Column("source", String(24), nullable=False), Column("scenario", String(60), nullable=False),
    Column("window_start", String(32), nullable=False), Column("window_end", String(32), nullable=False),
    Column("config_sha256", String(64), nullable=False), Column("computed_at", String(32), nullable=False),
    Column("report", JSON, nullable=False),
    UniqueConstraint("model_id", "source", "scenario", "window_start", "window_end", "config_sha256", name="run_scope"))
Index("runs_scope_end", runs.c.model_id, runs.c.source, runs.c.scenario, runs.c.window_end)
alerts = Table("alerts", metadata,
    Column("id", String(64), primary_key=True), Column("run_id", String(64), ForeignKey(runs.c.id), nullable=False),
    Column("rule", String(40), nullable=False), Column("active", Integer, nullable=False),
    Column("first_seen", String(32), nullable=False), Column("last_seen", String(32), nullable=False),
    Column("evidence", JSON, nullable=False))
Index("alerts_run", alerts.c.run_id)
versions = Table("schema_version", metadata, Column("version", Integer, primary_key=True))


def access_statements() -> list[str]:
    statements = ["REVOKE ALL ON SCHEMA bank_monitoring FROM PUBLIC",
        "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bank_monitoring_writer') THEN CREATE ROLE bank_monitoring_writer NOLOGIN; END IF; END $$",
        "DO $$ BEGIN EXECUTE format('GRANT bank_monitoring_writer TO %I', current_user); END $$",
        "GRANT USAGE ON SCHEMA bank_monitoring TO bank_monitoring_writer"]
    for table in metadata.sorted_tables:
        name = f'bank_monitoring."{table.name}"'
        privileges = ("SELECT" if table.name == "schema_version" else "SELECT, INSERT" if table.name in ("models", "outcomes")
                      else "SELECT, INSERT, UPDATE, DELETE" if table.name == "prediction_events" else "SELECT, INSERT, UPDATE")
        statements.extend([f"ALTER TABLE {name} ENABLE ROW LEVEL SECURITY",
            f"GRANT {privileges} ON {name} TO bank_monitoring_writer",
            f"DROP POLICY IF EXISTS monitoring_service ON {name}",
            f"CREATE POLICY monitoring_service ON {name} TO bank_monitoring_writer USING (true) WITH CHECK (true)"])
    return statements


class OutcomeConflict(ValueError):
    pass


class Store:
    def __init__(self, database_url: str, *, migrate: bool = False):
        url = make_url(database_url)
        if url.drivername in ("postgres", "postgresql"):
            url = url.set(drivername="postgresql+psycopg")
        self.is_sqlite = url.get_backend_name() == "sqlite"
        if not self.is_sqlite and url.drivername != "postgresql+psycopg":
            raise ValueError("Monitoring supports PostgreSQL (psycopg) or local SQLite")
        options = {"hide_parameters": True, "json_serializer": lambda value: json.dumps(value, allow_nan=False)}
        if self.is_sqlite:
            if url.database and url.database != ":memory:":
                Path(url.database).parent.mkdir(parents=True, exist_ok=True)
            options.update(connect_args={"check_same_thread": False, "timeout": 3},
                           execution_options={"schema_translate_map": {"bank_monitoring": None}})
            if url.database in (None, ":memory:"):
                options["poolclass"] = StaticPool
        else:
            if url.host not in ("localhost", "127.0.0.1", "::1") and "sslmode" not in url.query:
                url = url.update_query_dict({"sslmode": "require"})
            options.update(pool_size=2, max_overflow=1, pool_timeout=3, pool_pre_ping=True,
                           connect_args={"connect_timeout": 5, "prepare_threshold": None,
                                         "options": "-c statement_timeout=5000"})
        self.engine = create_engine(url, **options)
        if self.is_sqlite:
            @event.listens_for(self.engine, "connect")
            def configure(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=3000")
                connection.execute("PRAGMA journal_mode=WAL")
        self.insert = sqlite_insert if self.is_sqlite else pg_insert
        if migrate or self.is_sqlite:
            self.migrate()
        if not self.is_sqlite:
            role = os.getenv("MONITORING_DATABASE_ROLE", "bank_monitoring_writer")
            if role:
                if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", role):
                    raise ValueError("Invalid monitoring database role")
                @event.listens_for(self.engine, "begin")
                def restrict_runtime(connection):
                    connection.exec_driver_sql(f'SET LOCAL ROLE "{role}"')

    def migrate(self):
        with self.engine.begin() as connection:
            if not self.is_sqlite:
                connection.execute(CreateSchema("bank_monitoring", if_not_exists=True))
            metadata.create_all(connection)
            if not self.is_sqlite:
                for statement in access_statements():
                    connection.execute(text(statement))
            connection.execute(self.insert(versions).values(version=1).on_conflict_do_nothing())
            if connection.execute(select(versions.c.version)).scalars().all() != [1]:
                raise ValueError("Unsupported monitoring schema version")

    def close(self):
        self.engine.dispose()

    def ping(self):
        with self.engine.connect() as connection:
            return connection.execute(select(versions.c.version)).scalar_one() == 1

    def register_model(self, model_id: str, reference: dict, details: dict, reference_hash: str):
        with self.engine.begin() as connection:
            connection.execute(self.insert(models).values(id=model_id, created_at=timestamp(now_utc()),
                reference_sha256=reference_hash, reference=reference, details=details).on_conflict_do_nothing(index_elements=["id"]))
            existing = connection.execute(select(models.c.reference_sha256).where(models.c.id == model_id)).scalar_one()
            if existing != reference_hash:
                raise ValueError("Model registry reference mismatch")

    def list_models(self) -> list[dict]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(models).order_by(models.c.created_at.desc()).limit(100)).mappings()
            return [{"id": row["id"], "created_at": row["created_at"], **row["details"]} for row in rows]

    def get_model(self, model_id: str) -> dict | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(models).where(models.c.id == model_id)).mappings().first()
            return dict(row) if row else None

    def append_events(self, rows: list[dict]) -> int:
        if any(row["source"] not in SOURCES for row in rows):
            raise ValueError("Unsupported monitoring source")
        inserted = 0
        with self.engine.begin() as connection:
            for offset in range(0, len(rows), 250):
                result = connection.execute(self.insert(events).values(rows[offset:offset + 250])
                                            .on_conflict_do_nothing(index_elements=["id"]))
                inserted += max(result.rowcount, 0)
        return inserted

    def add_outcomes(self, rows: list[dict], *, as_of: str | None = None) -> int:
        as_of = as_of or timestamp(now_utc())
        if len({row["event_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate prediction IDs in outcome upload")
        with self.engine.begin() as connection:
            found = connection.execute(select(events).where(events.c.id.in_([r["event_id"] for r in rows]))).mappings()
            lookup = {row["id"]: row for row in found}
            existing = connection.execute(select(outcomes).where(outcomes.c.event_id.in_([r["event_id"] for r in rows]))).mappings()
            previous = {row["event_id"]: row for row in existing}
            normalized = []
            for item in rows:
                prediction = lookup.get(item["event_id"])
                if prediction is None or not prediction["valid"]:
                    raise ValueError("Outcome must refer to an existing successful prediction")
                item = {**item, "observed_at": timestamp(item["observed_at"]), "horizon_end": timestamp(item["horizon_end"])}
                if item["label"] not in (0, 1) or isinstance(item["label"], bool):
                    raise ValueError("Churn outcome must be integer 0 or 1")
                if (item["horizon_end"] != prediction["horizon_end"] or item["observed_at"] < item["horizon_end"]
                        or item["observed_at"] > as_of):
                    raise ValueError("Outcome horizon must match the prediction and be fully observed by now")
                if prediction["source"] != "study_replay" and item["source"] != "observed":
                    raise ValueError("Simulated labels cannot be attached to live or snapshot-batch predictions")
                if item["event_id"] in previous and any(previous[item["event_id"]][k] != item[k] for k in item):
                    raise OutcomeConflict("Outcome is immutable; an incompatible label is already recorded")
                normalized.append(item)
            inserted = 0
            for offset in range(0, len(normalized), 250):
                result = connection.execute(self.insert(outcomes).values(normalized[offset:offset + 250])
                                            .on_conflict_do_nothing(index_elements=["event_id"]))
                inserted += max(result.rowcount, 0)
            return inserted

    def window_events(self, model_id: str, source: str, scenario: str, start: str, end: str, limit: int) -> list[dict]:
        query = (select(events, outcomes.c.label, outcomes.c.observed_at, outcomes.c.source.label("label_source"))
                 .outerjoin(outcomes, outcomes.c.event_id == events.c.id)
                 .where(events.c.model_id == model_id, events.c.source == source, events.c.scenario == scenario,
                        events.c.occurred_at >= start, events.c.occurred_at < end)
                 .order_by(events.c.occurred_at, events.c.id).limit(limit + 1))
        with self.engine.connect() as connection:
            result = [dict(row) for row in connection.execute(query).mappings()]
        if len(result) > limit:
            raise ValueError("Window exceeds max_window_events; choose smaller windows")
        return result

    def previous_run(self, model_id: str, source: str, scenario: str, start: str, config_hash: str) -> dict | None:
        with self.engine.connect() as connection:
            return connection.execute(select(runs.c.report).where(runs.c.model_id == model_id,
                runs.c.source == source, runs.c.scenario == scenario, runs.c.window_end == start,
                runs.c.config_sha256 == config_hash).order_by(runs.c.computed_at.desc()).limit(1)).scalar_one_or_none()

    def save_run(self, report: dict):
        row = {key: report[key] for key in ("id", "model_id", "source", "scenario", "window_start", "window_end", "config_sha256", "computed_at")}
        row["report"] = report
        with self.engine.begin() as connection:
            stmt = self.insert(runs).values(row)
            connection.execute(stmt.on_conflict_do_update(index_elements=["id"],
                set_={"report": stmt.excluded.report, "computed_at": stmt.excluded.computed_at}))
            connection.execute(update(alerts).where(alerts.c.run_id == report["id"]).values(active=0))
            for alert in report["alerts"]:
                alert_row = {"id": alert["id"], "run_id": report["id"], "rule": alert["rule"], "active": 1,
                             "first_seen": report["computed_at"], "last_seen": report["computed_at"], "evidence": alert}
                stmt = self.insert(alerts).values(alert_row)
                connection.execute(stmt.on_conflict_do_update(index_elements=["id"],
                    set_={"active": 1, "last_seen": stmt.excluded.last_seen, "evidence": stmt.excluded.evidence}))

    def list_runs(self, model_id: str, source: str, scenario: str, *, limit: int = 200, since: str | None = None) -> list[dict]:
        query = select(runs.c.report).where(runs.c.model_id == model_id, runs.c.source == source, runs.c.scenario == scenario)
        if since:
            query = query.where(runs.c.window_end >= since)
        with self.engine.connect() as connection:
            return connection.execute(query.order_by(runs.c.window_end.desc(), runs.c.computed_at.desc()).limit(limit)).scalars().all()

    def get_run(self, identifier: str) -> dict | None:
        with self.engine.connect() as connection:
            return connection.execute(select(runs.c.report).where(runs.c.id == identifier)).scalar_one_or_none()

    def prediction_scopes(self, identifiers: list[str]) -> list[dict]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(events.c.model_id, events.c.source, events.c.scenario, events.c.occurred_at)
                                      .where(events.c.id.in_(identifiers))).mappings()
            return [dict(row) for row in rows]

    def prune(self, before: str) -> int:
        # Called only by an explicit operator command; report/model history is retained.
        with self.engine.begin() as connection:
            return connection.execute(delete(events).where(events.c.occurred_at < before)).rowcount
