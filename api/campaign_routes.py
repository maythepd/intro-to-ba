"""Aggregate-only campaign reads; explicit authenticated replay with fixed server artifacts."""
from __future__ import annotations

import csv
from datetime import datetime
from html import escape
from io import StringIO
import json
import os
from pathlib import Path
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, field_validator

from api.monitoring_routes import require_admin, service_for
from monitoring.campaign_store import CampaignStore
from monitoring.settings import digest

router = APIRouter(prefix="/api/monitoring", tags=["Multi-model study replay"])
Batch = Literal["normal_01", "normal_02", "normal_03", "engagement_shift", "friction_shift"]
Algorithm = Literal["xgboost", "lr", "rf", "kmeans", "hierarchical"]


def store_for(request):
    service = service_for(request)
    if service.store is None or not service.status()["storage_ready"]:
        raise HTTPException(503, "Monitoring storage unavailable")
    return CampaignStore(service.store)


def detail(request, identifier):
    try:
        result = store_for(request).get(identifier)
    except HTTPException:
        raise
    except Exception as exc:
        service_for(request).failed(exc)
        raise HTTPException(503, "Campaign schema/history unavailable; check migration") from exc
    if result is None:
        raise HTTPException(404, "Campaign not found")
    return result


@router.get("/campaigns")
def list_campaigns(request: Request):
    try:
        return {"items": store_for(request).list()}
    except HTTPException:
        raise
    except Exception as exc:
        service_for(request).failed(exc)
        raise HTTPException(503, "Campaign schema/history unavailable; check migration") from exc


@router.get("/campaigns/{campaign_id}")
def campaign_detail(request: Request, campaign_id: str = PathParam(pattern=r"^[a-f0-9]{64}$")):
    return detail(request, campaign_id)


@router.get("/campaigns/{campaign_id}/classification")
def classification(request: Request, campaign_id: str = PathParam(pattern=r"^[a-f0-9]{64}$"), batch: Batch | None = None):
    result = detail(request, campaign_id)
    return {"campaign_id": campaign_id, "status": result["status"], "items": [item for item in result["reports"]
        if item["task"] == "classification" and (batch is None or item["batch_id"] == batch)]}


@router.get("/campaigns/{campaign_id}/clustering")
def clustering(request: Request, campaign_id: str = PathParam(pattern=r"^[a-f0-9]{64}$"), batch: Batch | None = None):
    result = detail(request, campaign_id)
    return {"campaign_id": campaign_id, "status": result["status"], "items": [item for item in result["reports"]
        if item["task"] == "clustering" and (batch is None or item["batch_id"] == batch)]}


@router.get("/campaigns/{campaign_id}/input-drift")
def input_drift(request: Request, campaign_id: str = PathParam(pattern=r"^[a-f0-9]{64}$"), batch: Batch | None = None):
    detail(request, campaign_id)
    return {"campaign_id": campaign_id, "items": store_for(request).input_reports(campaign_id, batch)}


@router.get("/campaigns/{campaign_id}/export")
def download(request: Request, campaign_id: str = PathParam(pattern=r"^[a-f0-9]{64}$"),
             format: Literal["json", "csv", "html"] = "json", task: Literal["classification", "clustering"] = "classification"):
    result = detail(request, campaign_id)
    headers = {"Content-Disposition": f'attachment; filename="campaign-{campaign_id[:12]}-{task}.{format}"',
               "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}
    if format == "json":
        return Response(json.dumps(result, allow_nan=False), media_type="application/json", headers=headers)
    if format == "html":
        return HTMLResponse("<!doctype html><meta charset='utf-8'><title>BaBank campaign</title>"
            "<style>pre{white-space:pre-wrap;overflow-wrap:anywhere}body{font:14px system-ui;margin:2rem}</style>"
            "<h1>Simulated five-model study replay</h1><pre>" + escape(json.dumps(result, indent=2, allow_nan=False)) + "</pre>", headers=headers)
    rows = []
    for item in result["reports"]:
        if item["task"] != task:
            continue
        report = item["report"]
        common = {"campaign_id": campaign_id, "algorithm": item["algorithm"], "batch": item["batch_id"],
                  "model_id": item["model_id"], "N": report["sample_count"], "status": report["status"]}
        if task == "classification":
            measured = report["performance"]
            common.update({"status": measured["status"], "prevalence": measured.get("churn_rate"), "coverage": measured["label_coverage"],
                           "threshold": measured["threshold"], **(measured["metrics"] or {})})
            common.update({"mature_N": measured["mature_samples"], "labeled_N": measured["labeled_samples"],
                           "positive_N": measured["positive_samples"], "probability_psi": (report.get("prediction_drift") or {}).get("score")})
            matrix = common.pop("confusion_matrix", None)
            if matrix is not None:
                common.update(TN=matrix[0][0], FP=matrix[0][1], FN=matrix[1][0], TP=matrix[1][1])
            common.update({"baseline_" + key: value for key, value in
                ((measured.get("synthetic_baseline") or {}).get("metrics") or {}).items() if key != "confusion_matrix"})
            common.update({"drop_" + key: value for key, value in measured.get("drops", {}).items()})
        else:
            common.update({key: report.get(key) for key in ("assignment_method", "eval_N", "silhouette", "dbi", "chi",
                "mean_distance", "p95_distance", "distortion", "outlier_share", "margin", "active_clusters")})
            common.update({"baseline_" + key: value for key, value in report.get("baseline", {}).items()})
            common.update({"drop_" + key: value for key, value in report.get("drops", {}).items()})
        rows.append(common)
    output = StringIO()
    if rows:
        columns = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return Response(output.getvalue(), media_type="text/csv", headers=headers)


class ReplayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch: Batch | None = None
    algorithm: Algorithm | None = None
    with_labels: bool | None = None
    anchor: datetime | None = None
    campaign_id: str | None = None

    @field_validator("anchor")
    @classmethod
    def timezone_required(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Timezone required")
        return value

    @field_validator("campaign_id")
    @classmethod
    def valid_identifier(cls, value):
        import re
        if value is not None and not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("Invalid campaign identity")
        return value


@router.post("/multi-model-replay")
def replay(payload: ReplayInput, service=Depends(require_admin)):
    if not service.settings.demo_enabled:
        raise HTTPException(403, "Study replay is disabled")
    directory = os.getenv("DEMO_DATA_DIR")
    if not directory:
        raise HTTPException(503, "DEMO_DATA_DIR is not configured")
    from monitoring import campaign
    from monitoring.multi_model_replay import persist
    from api import module5
    try:
        anchor, labeled = payload.anchor, bool(payload.with_labels)
        if payload.campaign_id:
            old = CampaignStore(service.store).get(payload.campaign_id)
            if old is None:
                raise HTTPException(404, "Campaign not found")
            expected = old["manifest"]["mode"] == "simulated_labels"
            if payload.with_labels is not None and payload.with_labels != expected:
                raise HTTPException(422, "Resume must preserve the stored outcome mode")
            labeled, anchor = expected, old["manifest"]["anchor"]
        root = Path(directory)
        labels = root / "synthetic_outcomes.csv" if labeled else None
        hashes = [module5.file_hash(root / "manifest.json"), module5.file_hash(root / "customer_features.csv"),
                  module5.file_hash(campaign.ROOT / "config/multi_model_monitoring.yaml"), service.model_id,
                  str(anchor), labeled]
        from api.challenger_models import ARTIFACTS
        from demo_data.validate import model_paths
        hashes.extend(module5.file_hash(path) for path in model_paths().values())
        hashes.extend(module5.file_hash(ARTIFACTS / name / "manifest.json") for name in ("lr", "rf"))
        if labels:
            hashes.extend([module5.file_hash(labels), module5.file_hash(labels.with_suffix(".csv.json"))])
        key = digest(hashes)
        # Bounded, short-lived preflight cache avoids repeating expensive RF/silhouette work for 25 admin calls.
        with service.lock:
            cached = getattr(service, "campaign_preflight", None)
            if not cached or cached[0] != key or time.monotonic() - cached[1] > 300:
                prepared = campaign.prepare(service, data_dir=root, labels_path=labels, anchor=anchor)
                service.campaign_preflight = (key, time.monotonic(), prepared)
            else:
                prepared = cached[2]
        if payload.campaign_id and prepared.manifest["campaign_id"] != payload.campaign_id:
            raise HTTPException(422, "Stored campaign artifacts/config no longer match")
        return persist(service, prepared, algorithm=payload.algorithm, batch=payload.batch, execution_origin="authenticated_admin_api")
    except HTTPException:
        raise
    except (FileNotFoundError, KeyError) as exc:
        service.failed(exc)
        raise HTTPException(503, "Approved campaign artifacts are missing; run preparation") from exc
    except ValueError as exc:
        service.failed(exc)
        raise HTTPException(422, "Campaign inputs/artifacts/provenance are invalid; inspect operator configuration") from exc
    except Exception as exc:
        service.failed(exc)
        raise HTTPException(503, "Campaign unit failed; completed units are retained and can be resumed") from exc
