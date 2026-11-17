"""Bank customer churn inference and staff lookup API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
import os
import time
import numpy as np
import pandas as pd
from pydantic import ConfigDict, TypeAdapter, ValidationError
from api.scoring import score_probabilities, risk_level
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from api import module5, segmentation_service, artifacts_inspector

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

def compute_batch_risk(app: FastAPI, include_predictions: bool = False, force: bool = False) -> dict:
    model = getattr(app.state, "churn_model", None)
    customers = getattr(app.state, "customers", None)
    if model is None or customers is None or customers.empty:
        raise HTTPException(503, "Official customer feature table or churn model unavailable")

    batch_lock = getattr(app.state, "batch_lock", None)
    if batch_lock is None:
        batch_lock = Lock()
        app.state.batch_lock = batch_lock

    with batch_lock:
        if getattr(app.state, "risk_summary", None) is not None and not force:
            result = dict(app.state.risk_summary, cached=True)
            if include_predictions:
                result["predictions"] = app.state.batch_predictions
            return result

        try:
            probabilities = score_probabilities(model, customers)
        except Exception as exc:
            raise HTTPException(500, "Batch scoring failed: verify fitted model compatibility") from exc

        p_arr = np.asarray(probabilities, dtype=float)
        levels = np.where(p_arr < 0.3, "low", np.where(p_arr < 0.7, "medium", "high"))
        low_count = int(np.sum(levels == "low"))
        med_count = int(np.sum(levels == "medium"))
        high_count = int(np.sum(levels == "high"))
        counts = {"low": low_count, "medium": med_count, "high": high_count}

        scores = []
        if include_predictions:
            for identifier, p, level in zip(customers["customer_id"], p_arr, levels):
                scores.append({
                    "customer_id": str(int(identifier)),
                    "churn_probability": float(p),
                    "predicted_churn": bool(p >= 0.5),
                    "risk_level": str(level),
                })
        else:
            scores = getattr(app.state, "batch_predictions", None)

        summary = {
            "status": "scored",
            "source": "Committed Module 5 XGBoost checkpoint on the Module 5 customer feature table",
            "data_kind": "synthetic-extended study snapshot; not live bank activity",
            "model_name": "XGBoost (Module 5)",
            "model_source": "module5_checkpoint",
            "total_customers": len(customers),
            "scored_customers": len(customers),
            "counts": counts,
            "customers_requiring_attention": counts["high"],
            "risk_thresholds": {"medium_from": 0.3, "high_from": 0.7},
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        app.state.batch_predictions = scores
        app.state.risk_summary = summary
        app.state.inference_count = getattr(app.state, "inference_count", 0) + len(customers)
        result = dict(summary, cached=False)
        if include_predictions:
            result["predictions"] = scores
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = None
    app.state.customers = None
    app.state.inference_count = 0
    app.state.segmentation = None
    app.state.risk_summary = None
    app.state.batch_predictions = None
    app.state.batch_lock = Lock()

    try:
        app.state.churn_model = module5.load()
    except Exception as exc:
        print(f"[WARN] Failed to load Module 5 churn model: {exc}")

    try:
        if app.state.churn_model is not None:
            app.state.customers = app.state.churn_model.profiles(pd.read_csv(DATA_PATH))
    except Exception as exc:
        print(f"[WARN] Failed to materialize customer profiles: {exc}")

    try:
        app.state.segmentation = segmentation_service.load(SEGMENTATION_DATA_PATH)
    except Exception as exc:
        print(f"[WARN] Failed to load segmentation model: {exc}")

    yield

app = FastAPI(title="BaBank Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def telemetry_middleware(request, call_next):
    start_t = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start_t) * 1000
    path = request.url.path
    if not path.startswith("/docs") and not path.startswith("/openapi") and not path.startswith("/favicon"):
        artifacts_inspector.record_request_telemetry(
            method=request.method,
            path=path,
            status=response.status_code,
            latency_ms=elapsed_ms,
        )
    return response

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    model = getattr(app.state, "churn_model", None)
    metadata = getattr(model, "manifest", {}) if model is not None else {}
    return {
        "status": "ready" if model is not None else "model_unavailable",
        "model_loaded": model is not None,
        "customers_loaded": getattr(app.state, "customers", None) is not None,
        "segmentation_loaded": getattr(app.state, "segmentation", None) is not None,
        "model_source": "module5_checkpoint",
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "feature_count": len(metadata.get("features", [])),
        "study_reproduced": metadata.get("study_reproduced", False),
    }

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
    model = getattr(app.state, "churn_model", None)
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
    return compute_batch_risk(app, include_predictions=include_predictions, force=force)


@app.get("/api/dashboard/risk-summary")
def dashboard_risk_summary():
    require_churn_model()
    if getattr(app.state, "customers", None) is None:
        raise HTTPException(503, "Official customer feature table unavailable")
    if getattr(app.state, "risk_summary", None) is None:
        return compute_batch_risk(app, include_predictions=False)
    return app.state.risk_summary


@app.get("/api/dashboard/stats")
def dashboard_stats():
    """Return real dataset facts, actual monthly activity trend, and model risk counts."""
    facts = artifacts_inspector.get_dataset_facts()
    risk = getattr(app.state, "risk_summary", None)
    if risk is None and getattr(app.state, "churn_model", None) is not None:
        try:
            risk = compute_batch_risk(app, include_predictions=False)
        except Exception:
            pass
    return {
        **facts,
        "risk_summary": risk,
        "champion_model": "XGBoost (Module 5)",
    }


@app.post("/api/predict")
def predict(p: CustomerInput):
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
    return {"customer_id": str(p.customer_id or ""),"churn_probability":prob,
            "predicted_churn":prob>=.5,"risk_level":level,"model_name":"XGBoost (Module 5)",
            "model_source":"module5_checkpoint", "signals":signals(signal_input)}

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
    report = Path(__file__).resolve().parent.parent / "results/comparison/model_comparison_metrics.csv"
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
        champion = next((row for row in rows if row["model"] == "XGBoost"), rows[0])
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Committed model comparison results are invalid") from exc
    model = getattr(app.state, "churn_model", None)
    manifest = getattr(model, "manifest", {}) if model is not None else {}
    return {
        "model": "XGBoost",
        "source": "Committed Module 5 model comparison (development study, not live API performance)",
        "report_path": "results/comparison/model_comparison_metrics.csv",
        "roc_auc": champion["roc_auc"],
        "pr_auc": champion["pr_auc"],
        "f1": champion["f1"],
        "precision": champion["precision"],
        "recall": champion["recall"],
        "models": rows,
        "deployment_evaluated": False,
        "serving_source": "module5_checkpoint",
        "study_reproduced": manifest.get("study_reproduced", False),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
    }

@app.get("/api/models/checkpoints")
def model_checkpoints():
    """Return all detected model checkpoints, file paths, sizes, and hyperparameters."""
    return artifacts_inspector.get_model_checkpoints()


@app.get("/api/models/decile-lift")
def model_decile_lift():
    """Return the actual 10-decile lift table from the model development benchmark."""
    return artifacts_inspector.get_decile_lift_table()


@app.get("/api/models/feature-importance")
def model_feature_importance():
    """Return the top 15 features for XGBoost, Random Forest, and Logistic Regression."""
    return artifacts_inspector.get_top_features()


@app.get("/api/monitoring")
def monitoring():
    inf_count = getattr(app.state, "inference_count", 0)
    data = artifacts_inspector.get_monitoring_telemetry(inference_count=inf_count)
    data["model_loaded"] = getattr(app.state, "churn_model", None) is not None
    data["segmentation_loaded"] = getattr(app.state, "segmentation", None) is not None
    return data

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


# Serve static production frontend when built (for single-container cloud deploy or static serving)
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend/dist"
if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_frontend(full_path: str):
        if (
            full_path.startswith("api")
            or full_path.startswith("health")
            or full_path.startswith("docs")
            or full_path.startswith("openapi.json")
        ):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_file = FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not Found")
