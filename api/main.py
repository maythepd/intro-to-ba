"""Bank customer churn inference and staff lookup API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
import pandas as pd
from pydantic import ConfigDict, TypeAdapter, ValidationError
from api.scoring import score_probabilities, risk_level
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from api import module5, segmentation_service
from api.monitoring_routes import router as monitoring_router
from monitoring.service import MonitoringService

DATA_PATH = module5.DATA_PATH
SEGMENTATION_DATA_PATH = Path(os.environ.get("SEGMENTATION_CUSTOMERS_CSV_PATH", "data/official/customer_features.csv"))

class CustomerInput(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    customer_id: int | None = None
    credit_score: float = Field(ge=300,le=850)
    country: Literal["France","Germany","Spain"]
    gender: Literal["Male","Female"]
    age: int = Field(ge=18,le=100)
    tenure: int = Field(ge=0,le=50)
    balance: float = Field(ge=0)
    products_number: int = Field(ge=1,le=4)
    credit_card: int = Field(ge=0,le=1)
    active_member: int = Field(ge=0,le=1)
    estimated_salary: float = Field(ge=0)
    recency_days: float = Field(ge=0)
    monthly_transactions: float = Field(ge=0)
    trans_amount_3m: float = Field(ge=0)
    trans_trend_ratio: float = Field(ge=0)
    app_logins_monthly: float = Field(ge=0)
    customer_service_complaints: int = Field(ge=0)
    has_unresolved_complaint: int = Field(ge=0,le=1)
    preferred_channel: Literal["Mobile App","Online Web","ATM","Branch","No_Channel"]
    primary_device: str = Field(min_length=1,max_length=60)
    raw_features: dict[str, float | str | None] | None = None

BATCH_CUSTOMER_ADAPTER = TypeAdapter(list[CustomerInput])

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = module5.load()
    app.state.customers = (app.state.churn_model.profiles(pd.read_csv(DATA_PATH))
                           if app.state.churn_model is not None else None)
    app.state.inference_count = 0
    app.state.segmentation = segmentation_service.load(SEGMENTATION_DATA_PATH)
    app.state.risk_summary = None
    app.state.batch_predictions = None
    app.state.batch_lock = Lock()
    app.state.monitoring = MonitoringService(app.state.churn_model, segmentation=app.state.segmentation)
    try:
        yield
    finally:
        app.state.monitoring.close()

app = FastAPI(title="BaBank Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("CORS_ORIGINS","http://localhost:5173").split(",") if x.strip()],
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(monitoring_router)


@app.middleware("http")
async def monitor_prediction_errors(request, call_next):
    started = perf_counter()
    response = await call_next(request)
    service = getattr(request.app.state, "monitoring", None)
    if (request.method == "POST" and request.url.path == "/api/predict" and response.status_code >= 400
            and service is not None and service.model is request.app.state.churn_model):
        # Sync database work stays off the event loop.
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(service.record_failure, "validation_error" if response.status_code == 422 else "inference_error",
                                (perf_counter() - started) * 1000)
    return response

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    model = app.state.churn_model
    metadata = getattr(model, "manifest", {})
    return {"status":"ready" if model is not None else "model_unavailable",
            "model_loaded":model is not None,"customers_loaded":app.state.customers is not None,
            "segmentation_loaded":app.state.segmentation is not None,
            "model_source":"module5_checkpoint", "checkpoint_sha256":metadata.get("checkpoint_sha256"),
            "feature_count":len(metadata.get("features", [])), "study_reproduced":metadata.get("study_reproduced", False),
            "monitoring":app.state.monitoring.status()}

def signals(p: CustomerInput) -> list[dict[str,str]]:
    result=[]
    for flag,title,description in [
        (p.trans_trend_ratio < .75,"Declining Transactions","Transaction activity has fallen by more than 25%."),
        (p.customer_service_complaints >= 2 or p.has_unresolved_complaint == 1,"Service Complaints","Customer complaints require resolution."),
        (p.recency_days >= 45,"Low Engagement","No recorded interaction for at least 45 days."),
        (p.products_number >= 3,"Multiple Products","Review the customer's experience with multiple products."),
    ]:
        if flag: result.append({"title":title,"description":description,"direction":"risk"})
    return result

def require_churn_model():
    model = app.state.churn_model
    if model is None:
        raise HTTPException(503, "Module 5 checkpoint/preprocessor unavailable. Run python -m api.prepare_module5 --download")
    if not callable(getattr(model, "predict_proba", None)):
        raise HTTPException(503, "Invalid Module 5 prediction model")
    return model


@app.post("/api/predict/batch")
def predict_batch(
    include_predictions: bool = Query(False, description="Include every customer's probability and risk level."),
    force: bool = Query(False, description="Recompute instead of returning the current process cache."),
    batch_size: int = Query(1000, ge=1, le=5000),
):
    """Score the entire official customer snapshot; cache only completed runs."""
    model = require_churn_model()
    customers = app.state.customers
    if customers is None:
        raise HTTPException(503, "Official customer feature table unavailable")
    if customers.empty or "customer_id" not in customers.columns:
        raise HTTPException(503, "Official customer dataset is empty or has no customer IDs")
    if customers.customer_id.isna().any() or customers.customer_id.duplicated().any():
        raise HTTPException(503, "Official customer IDs must be unique and non-null")

    with app.state.batch_lock:
        if app.state.risk_summary is not None and not force:
            result = dict(app.state.risk_summary, cached=True)
            if include_predictions:
                result["predictions"] = app.state.batch_predictions
            return result

        required_columns = [name for name, field in CustomerInput.model_fields.items() if field.is_required()]
        missing = set(required_columns) - set(customers.columns)
        if missing:
            raise HTTPException(503, "Official dataset is missing columns required by the prediction API")

        scores = []
        telemetry_chunks = []
        counts = {"low": 0, "medium": 0, "high": 0}
        for start in range(0, len(customers), batch_size):
            chunk = customers.iloc[start:start + batch_size]
            records = chunk.loc[:, required_columns].to_dict(orient="records")
            try:
                TypeAdapter(list[CustomerInput]).validate_python(records)
            except ValidationError as exc:
                raise HTTPException(503, "Official customer features failed input validation") from exc
            try:
                started = perf_counter()
                probabilities = score_probabilities(model, chunk)
            except Exception as exc:
                raise HTTPException(500, "Batch scoring failed: verify fitted model compatibility") from exc
            for identifier, probability in zip(chunk["customer_id"], probabilities):
                p = float(probability)
                level = risk_level(p)
                counts[level] += 1
                scores.append({
                    "customer_id": str(int(identifier)),
                    "churn_probability": p,
                    "predicted_churn": p >= 0.5,
                    "risk_level": level,
                })
            telemetry_chunks.append((chunk, probabilities, (perf_counter() - started) * 1000 / len(chunk)))
        if sum(counts.values()) != len(customers):
            raise HTTPException(500, "Batch scoring was incomplete")
        # Log only after every chunk succeeds, using one atomic, idempotent batch.
        service = app.state.monitoring
        if service.model is model and service.store:
            try:
                telemetry = []
                for chunk, probabilities, elapsed in telemetry_chunks:
                    telemetry.extend(service.event_rows(chunk, probabilities, source="snapshot_batch",
                        namespace="snapshot:" + model.manifest["data_sha256"], latency_ms=elapsed))
                service.store.append_events(telemetry)
            except Exception as exc:
                service.failed(exc)
        summary = {
            "status": "scored",
            "source": "Committed Module 5 XGBoost checkpoint on the Module 5 customer feature table",
            "data_kind": "synthetic-extended study snapshot; not live bank activity",
            "model_name": "XGBoost (Module 5)",
            "model_source": "module5_checkpoint",
            "total_customers": len(customers),
            "scored_customers": len(scores),
            "counts": counts,
            "customers_requiring_attention": counts["high"],
            "risk_thresholds": {"medium_from": 0.3, "high_from": 0.7},
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        # An interrupted or failed batch must never replace the previous completed result.
        app.state.batch_predictions = scores
        app.state.risk_summary = summary
        app.state.inference_count += len(scores)
        result = dict(summary, cached=False)
        if include_predictions:
            result["predictions"] = scores
        return result


@app.get("/api/dashboard/risk-summary")
def dashboard_risk_summary():
    require_churn_model()
    if app.state.customers is None:
        raise HTTPException(503, "Official customer feature table unavailable")
    summary = app.state.risk_summary
    if summary is None:
        raise HTTPException(503, "Batch scoring not run yet. POST /api/predict/batch to compute actual risk counts")
    return summary


@app.post("/api/predict")
def predict(p: CustomerInput):
    started = perf_counter()
    model = require_churn_model()
    try:
        inputs = (module5.prediction_inputs(model, p.model_dump(), app.state.customers)
                  if isinstance(model, module5.Module5Model) else pd.DataFrame([p.model_dump()]))
        prob = float(score_probabilities(model, inputs)[0])
    except module5.Module5InputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "Model inference failed: verify training artifact compatibility") from exc
    app.state.inference_count += 1
    level = risk_level(prob)
    signal_input = p
    if isinstance(model, module5.Module5Model):
        profile = model.profiles(inputs).iloc[0].to_dict()
        signal_input = CustomerInput.model_validate({**profile, "customer_id":p.customer_id})
    service = app.state.monitoring
    telemetry = (service.record(inputs, [prob], source="live", customer_ids=[p.customer_id],
                                    latency_ms=(perf_counter() - started) * 1000)[0]
                     if service.model is model else None)
    return {"customer_id": str(p.customer_id or ""),"churn_probability":prob,
            "predicted_churn":prob>=.5,"risk_level":level,"model_name":"XGBoost (Module 5)",
            "model_source":"module5_checkpoint", "signals":signals(signal_input),
            "prediction_id":telemetry["prediction_id"] if telemetry else None,
            "prediction_timestamp":telemetry["prediction_timestamp"] if telemetry else None,
            "outcome_horizon_end":telemetry["outcome_horizon_end"] if telemetry else None,
            "monitoring_recorded":telemetry is not None}

@app.get("/api/customers")
def customers(search: str="",limit: int=Query(20,ge=1,le=100)):
    df=app.state.customers
    if df is None: raise HTTPException(503,"Customer dataset unavailable")
    matches=df[df.customer_id.astype(str).str.contains(search,regex=False)] if search else df
    records=matches.head(limit).loc[:, module5.PROFILE_COLUMNS].to_dict("records")
    for record in records:
        record["customer_id"]=str(record["customer_id"])
    return {"total":int(len(matches)),"items":records}

@app.get("/api/customers/{customer_id}")
def customer(customer_id:int):
    df=app.state.customers
    if df is None: raise HTTPException(503,"Customer dataset unavailable")
    found=df.loc[df.customer_id==customer_id]
    if found.empty: raise HTTPException(404,"Customer not found")
    record=found.loc[:, module5.PROFILE_COLUMNS].iloc[0].to_dict()
    record["customer_id"]=str(record["customer_id"])
    return record

@app.get("/api/metrics")
def metrics():
    """Return the committed Module 5 comparison, never deployment telemetry."""
    report = Path(__file__).resolve().parent.parent / "model_development/results/comparison/model_comparison_metrics.csv"
    if not report.is_file():
        raise HTTPException(503, "Committed model comparison results are unavailable")
    try:
        df = pd.read_csv(report)
        names = {"LR": "Logistic Regression", "RF": "Random Forest", "XGBOOST": "XGBoost"}
        required = {"Model", "ROC_AUC", "PR_AUC", "F1_Score", "Precision", "Recall", "Accuracy", "Balanced_Accuracy", "CV_ROC_AUC", "CV_PR_AUC"}
        if not required.issubset(df.columns):
            raise ValueError("Comparison CSV is missing required metric columns")
        rows = [
            {
                "model": names[str(row["Model"]).upper()],
                "roc_auc": float(row["ROC_AUC"]),
                "pr_auc": float(row["PR_AUC"]),
                "f1": float(row["F1_Score"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "accuracy": float(row["Accuracy"]),
                "balanced_accuracy": float(row["Balanced_Accuracy"]),
                "cv_roc_auc": float(row["CV_ROC_AUC"]),
                "cv_pr_auc": float(row["CV_PR_AUC"]),
            }
            for _, row in df.iterrows()
        ]
        champion = next(row for row in rows if row["model"] == "XGBoost")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Committed model comparison results are invalid") from exc
    return {
        "model": "XGBoost",
        "source": "Committed Module 5 model comparison (development study, not live API performance)",
        "report_path": "model_development/results/comparison/model_comparison_metrics.csv",
        "roc_auc": champion["roc_auc"],
        "pr_auc": champion["pr_auc"],
        "f1": champion["f1"],
        "precision": champion["precision"],
        "recall": champion["recall"],
        "models": rows,
        "deployment_evaluated": False,
        "serving_source": "module5_checkpoint",
        "study_reproduced": getattr(app.state.churn_model, "manifest", {}).get("study_reproduced", False),
        "checkpoint_sha256": getattr(app.state.churn_model, "manifest", {}).get("checkpoint_sha256"),
    }

def require_segmentation():
    bundle = app.state.segmentation
    if bundle is None:
        raise HTTPException(503, "Segmentation unavailable or official feature snapshot does not match the exported model. Run python -m api.segmentation_service.")
    return bundle


@app.get("/api/segments")
def segments(algorithm: Literal["kmeans", "hierarchical"] = "kmeans"):
    return segmentation_service.summary(require_segmentation(), algorithm)


@app.get("/api/segments/customers/{customer_id}")
def customer_segments(customer_id: int):
    result = segmentation_service.customer_membership(require_segmentation(), str(customer_id))
    if result is None:
        raise HTTPException(404, "Customer not found in the fitted segmentation snapshot")
    return result


@app.post("/api/segments/predict")
def predict_segments(payload: CustomerInput, algorithm: Literal["kmeans", "hierarchical"] = "kmeans"):
    try:
        return segmentation_service.predict(require_segmentation(), payload.model_dump(), algorithm)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, "Invalid segmentation features") from exc


# Optional synthetic customer data API; disabled unless a dataset path is configured.
if demo_dir := os.getenv("DEMO_DATA_DIR"):
    from demo_data.serve import demo_router

    app.include_router(demo_router(Path(demo_dir)))
