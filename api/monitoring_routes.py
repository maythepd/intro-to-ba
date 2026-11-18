from __future__ import annotations

from datetime import datetime, timedelta
from hmac import compare_digest
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from monitoring.report import render_html
from monitoring.runner import closed_window, summarize
from monitoring.settings import now_utc, timestamp
from monitoring.store import OutcomeConflict

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
Source = Literal["live", "snapshot_batch", "study_replay"]


def service_for(request: Request):
    return request.app.state.monitoring


def require_admin(request: Request, authorization: str | None = Header(None)):
    service = service_for(request)
    token = service.settings.admin_token
    if not token:
        raise HTTPException(503, "Monitoring admin operations are not configured")
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not authorization or not authorization.startswith("Bearer ") or not compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "Monitoring admin authorization required")
    if not service.store or not service.status()["storage_ready"]:
        raise HTTPException(503, "Monitoring database is unavailable")
    return service


@router.get("")
def summary(request: Request, source: Source = "live", scenario: str = Query("", max_length=60, pattern=r"^[a-zA-Z0-9_-]*$"),
            model_id: str | None = Query(None, pattern=r"^[a-f0-9]{64}$"), days: int = Query(30, ge=1, le=180)):
    service = service_for(request)
    if (source == "study_replay" and not scenario) or (source != "study_replay" and scenario):
        raise HTTPException(422, "Select a scenario for study_replay; other sources have no scenario")
    try:
        # Replay uses synthetic historical dates; its day range is anchored to the
        # latest stored replay window rather than pretending it was observed today.
        result = service.summary(source=source, scenario=scenario, model_id=model_id,
                                 since=None if source == "study_replay" else timestamp(now_utc() - timedelta(days=days)))
        if source == "study_replay" and result["history"]:
            end = datetime.fromisoformat(result["history"][-1]["window_end"])
            result["history"] = [item for item in result["history"] if datetime.fromisoformat(item["window_end"]) >= end - timedelta(days=days)]
        return {**result, "model_loaded": request.app.state.churn_model is not None,
                "predictions_this_process": request.app.state.inference_count,
                "policy": {key: service.config[key] for key in ("min_labeled_samples", "min_class_samples", "min_label_coverage",
                           "consecutive_windows", "psi_critical", "js_critical", "performance_drop", "quality_error_rate")}}
    except Exception as exc:
        service.failed(exc)
        raise HTTPException(503, "Monitoring history is temporarily unavailable") from exc


@router.get("/models")
def model_versions(request: Request):
    service = service_for(request)
    if not service.store:
        raise HTTPException(503, "Monitoring database is not configured")
    return {"items": service.store.list_models()}


@router.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str):
    service = service_for(request)
    if not service.store:
        raise HTTPException(503, "Monitoring database is unavailable")
    report = service.store.get_run(run_id)
    if report is None:
        raise HTTPException(404, "Monitoring report not found")
    return report


@router.get("/runs/{run_id}/report")
def download_report(request: Request, run_id: str, format: Literal["html", "json"] = "html"):
    report = run_detail(request, run_id)
    # Identifier comes from the stored report, not arbitrary header input.
    headers = {"Content-Disposition": f'attachment; filename="monitoring-{report["id"][:16]}.{format}"',
               "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}
    return JSONResponse(report, headers=headers) if format == "json" else HTMLResponse(render_html(report), headers=headers)


class RunInput(BaseModel):
    source: Source = "live"
    scenario: str = Field("", max_length=60, pattern=r"^[a-zA-Z0-9_-]*$")
    model_id: str | None = Field(None, pattern=r"^[a-f0-9]{64}$")
    start: datetime | None = None
    end: datetime | None = None
    days: int = Field(2, ge=1, le=7)

    @field_validator("start", "end")
    @classmethod
    def timezone_required(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("Timezone is required")
        return value


@router.post("/runs")
def run_monitoring(payload: RunInput, service=Depends(require_admin)):
    try:
        if (payload.start is None) != (payload.end is None):
            raise ValueError("Provide both start and end")
        if payload.start:
            windows = [(timestamp(payload.start), timestamp(payload.end))]
        else:
            if payload.days > service.config["max_job_windows"]:
                raise ValueError("Job exceeds the configured window limit")
            start, _ = closed_window(now_utc(), service.config["window_hours"])
            first = datetime.fromisoformat(start) - timedelta(hours=service.config["window_hours"] * (payload.days - 1))
            windows = [(timestamp(first + timedelta(hours=service.config["window_hours"] * i)),
                        timestamp(first + timedelta(hours=service.config["window_hours"] * (i + 1)))) for i in range(payload.days)]
        reports = [service.run(source=payload.source, scenario=payload.scenario, model_id=payload.model_id,
                               start=start, end=end) for start, end in windows]
        return {"items": [summarize(report) for report in reports]}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class OutcomeInput(BaseModel):
    event_id: str = Field(min_length=32, max_length=64, pattern=r"^[a-f0-9]+$")
    label: int = Field(strict=True, ge=0, le=1)
    observed_at: datetime
    horizon_end: datetime
    source: Literal["observed", "study_snapshot", "synthetic_stress"] = "observed"

    @field_validator("observed_at", "horizon_end")
    @classmethod
    def timezone_required(cls, value):
        if value.tzinfo is None:
            raise ValueError("Timezone is required")
        return value


class OutcomeUpload(BaseModel):
    items: list[OutcomeInput] = Field(min_length=1, max_length=1000)


@router.post("/labels")
def add_labels(payload: OutcomeUpload, service=Depends(require_admin)):
    try:
        rows = [{**item.model_dump(), "observed_at": timestamp(item.observed_at), "horizon_end": timestamp(item.horizon_end)} for item in payload.items]
        inserted = service.store.add_outcomes(rows)
        updated = service.refresh_labels([row["event_id"] for row in rows])
        return {"inserted": inserted, "received": len(rows), "reports_updated": updated}
    except OutcomeConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/demo")
def run_demo(service=Depends(require_admin)):
    if not service.settings.demo_enabled:
        raise HTTPException(403, "Study replay is disabled on this server")
    from monitoring.simulate import simulate
    return simulate(service)
