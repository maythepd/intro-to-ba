"""Additive campaign storage, separate clustering assignments and aggregate read APIs."""
from __future__ import annotations

from sqlalchemy import Column, String, Integer, Float, JSON, ForeignKey, CheckConstraint, Table, Index, select, update, func

from monitoring.settings import digest, now_utc, timestamp
from monitoring.store import metadata

campaigns = Table("monitoring_campaigns", metadata,
    Column("id", String(64), primary_key=True), Column("created_at", String(32), nullable=False),
    Column("status", String(20), nullable=False), Column("manifest", JSON, nullable=False))
cluster_models = Table("cluster_model_registry", metadata,
    Column("id", String(64), primary_key=True), Column("reference", JSON, nullable=False), Column("details", JSON, nullable=False))
units = Table("campaign_reports", metadata,
    Column("campaign_id", String(64), ForeignKey(campaigns.c.id), primary_key=True),
    Column("algorithm", String(20), primary_key=True), Column("batch_id", String(30), primary_key=True),
    Column("model_id", String(64), nullable=False), Column("task", String(20), nullable=False),
    Column("cohort_sha256", String(64), nullable=False), Column("report", JSON, nullable=False))
Index("campaign_reports_scope", units.c.campaign_id, units.c.task, units.c.batch_id)
assignments = Table("cluster_assignments", metadata,
    Column("campaign_id", String(64), ForeignKey(campaigns.c.id), primary_key=True),
    Column("model_id", String(64), ForeignKey(cluster_models.c.id), primary_key=True),
    Column("batch_id", String(30), primary_key=True), Column("subject", String(64), primary_key=True),
    Column("cluster_id", Integer, nullable=False), Column("distance", Float, nullable=False), Column("margin", Float, nullable=False),
    CheckConstraint("cluster_id >= 0 AND distance >= 0 AND margin >= 0", name="assignment_values"))
inputs = Table("input_drift_reports", metadata,
    Column("campaign_id", String(64), ForeignKey(campaigns.c.id), primary_key=True),
    Column("batch_id", String(30), primary_key=True), Column("feature_space_id", String(64), primary_key=True),
    Column("reference_sha256", String(64), nullable=False), Column("report", JSON, nullable=False))


class CampaignStore:
    def __init__(self, store):
        self.store = store

    def register(self, manifest):
        identifier = manifest["campaign_id"]
        with self.store.connection(write=True) as connection:
            connection.execute(self.store.insert(campaigns).values(id=identifier, created_at=timestamp(now_utc()),
                status="partial", manifest=manifest).on_conflict_do_nothing(index_elements=["id"]))
            existing = connection.execute(select(campaigns.c.manifest).where(campaigns.c.id == identifier)).scalar_one()
            if digest(existing) != digest(manifest):
                raise ValueError("Campaign identity collision or incompatible provenance")

    def register_cluster(self, model):
        with self.store.connection(write=True) as connection:
            connection.execute(self.store.insert(cluster_models).values(id=model["model_id"], reference=model["reference"],
                details={key: value for key, value in model.items() if key != "reference"}).on_conflict_do_nothing(index_elements=["id"]))
            saved = connection.execute(select(cluster_models.c.reference).where(cluster_models.c.id == model["model_id"])).scalar_one()
            if digest(saved) != digest(model["reference"]):
                raise ValueError("Cluster reference identity collision")

    def save_input(self, campaign_id, batch, space, reference_hash, report):
        with self.store.connection(write=True) as connection:
            connection.execute(self.store.insert(inputs).values(campaign_id=campaign_id, batch_id=batch,
                feature_space_id=space, reference_sha256=reference_hash, report=report).on_conflict_do_nothing())

    def save_assignments(self, rows):
        inserted = 0
        with self.store.connection(write=True) as connection:
            for start in range(0, len(rows), 250):
                result = connection.execute(self.store.insert(assignments).values(rows[start:start+250])
                    .on_conflict_do_nothing().returning(assignments.c.subject))
                inserted += len(result.scalars().all())
        return inserted

    def save_report(self, campaign_id, algorithm, batch, task, model_id, cohort_hash, report):
        with self.store.connection(write=True) as connection:
            manifest = connection.execute(select(campaigns.c.manifest).where(campaigns.c.id == campaign_id)).scalar_one()
            expected_model = manifest["models"].get(algorithm)
            expected_batch = next((item for item in manifest["batches"] if item["batch_id"] == batch), None)
            if (expected_model is None or expected_batch is None or expected_model["model_id"] != model_id or
                    expected_model["task"] != task or expected_batch["cohort_sha256"] != cohort_hash or
                    report.get("sample_count", report.get("N")) != expected_batch["N"]):
                raise ValueError("Campaign model/batch/cohort alignment mismatch")
            statement = self.store.insert(units).values(campaign_id=campaign_id, algorithm=algorithm, batch_id=batch,
                task=task, model_id=model_id, cohort_sha256=cohort_hash, report=report)
            connection.execute(statement.on_conflict_do_update(index_elements=["campaign_id", "algorithm", "batch_id"],
                               set_={"report": statement.excluded.report}))
            count = connection.execute(select(func.count()).select_from(units).where(units.c.campaign_id == campaign_id)).scalar_one()
            connection.execute(update(campaigns).where(campaigns.c.id == campaign_id).values(status="complete" if count == 25 else "partial"))

    def get(self, campaign_id):
        with self.store.connection() as connection:
            row = connection.execute(select(campaigns).where(campaigns.c.id == campaign_id)).mappings().first()
        if row is None:
            return None
        reports = self.reports(campaign_id)
        return {**dict(row), "reports": reports, "report_count": len(reports)}

    def list(self):
        with self.store.connection() as connection:
            rows = connection.execute(select(campaigns).order_by(campaigns.c.created_at.desc()).limit(100)).mappings().all()
        return [{"id": row["id"], "created_at": row["created_at"], "status": row["status"],
                 "mode": row["manifest"]["mode"], "dataset_sha256": row["manifest"]["dataset_sha256"],
                 "label_policy_version": row["manifest"].get("label_policy_version")} for row in rows]

    def reports(self, campaign_id, *, task=None, batch=None):
        query = select(units).where(units.c.campaign_id == campaign_id)
        if task:
            query = query.where(units.c.task == task)
        if batch:
            query = query.where(units.c.batch_id == batch)
        with self.store.connection() as connection:
            rows = connection.execute(query).mappings().all()
        order = {name: i for i, name in enumerate(("xgboost", "lr", "rf", "kmeans", "hierarchical"))}
        return sorted([dict(row) for row in rows], key=lambda row: (row["report"].get("window_start", ""), order.get(row["algorithm"], 99)))

    def input_reports(self, campaign_id, batch=None):
        query = select(inputs).where(inputs.c.campaign_id == campaign_id)
        if batch:
            query = query.where(inputs.c.batch_id == batch)
        with self.store.connection() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]
