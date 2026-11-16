"""Bank customer churn inference and staff lookup API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from typing import Literal
import os
import joblib
import pandas as pd
from pydantic import TypeAdapter, ValidationError
from api.scoring import score_probabilities, risk_level
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from api import segmentation_service

MODEL_PATH = Path(os.environ.get("MODEL_BUNDLE_PATH", "artifacts/churn_bundle.joblib"))
DATA_PATH = Path(os.environ.get("CUSTOMERS_CSV_PATH", "data/official/customer_features.csv"))

class CustomerInput(BaseModel):
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
    recency_days: int = Field(ge=1,le=180)
    monthly_transactions: int = Field(ge=0)
    trans_amount_3m: float = Field(ge=0)
    trans_trend_ratio: float = Field(ge=0)
    app_logins_monthly: float = Field(ge=0)
    customer_service_complaints: int = Field(ge=0)
    has_unresolved_complaint: int = Field(ge=0,le=1)
    preferred_channel: Literal["Mobile App","Online Web","ATM","Branch"]
    primary_device: str = Field(min_length=1,max_length=60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
    app.state.customers = pd.read_csv(DATA_PATH) if DATA_PATH.is_file() else None
    app.state.inference_count = 0
    app.state.segmentation = segmentation_service.load(DATA_PATH)
    app.state.risk_summary = None
    app.state.batch_predictions = None
    app.state.batch_lock = Lock()
    yield

app = FastAPI(title="BaBank Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("CORS_ORIGINS","http://localhost:5173").split(",") if x.strip()],
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health():
    return {"status":"ready" if app.state.bundle is not None else "model_unavailable",
            "model_loaded":app.state.bundle is not None,"customers_loaded":app.state.customers is not None,
            "segmentation_loaded":app.state.segmentation is not None}

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
    bundle = app.state.bundle
    if bundle is None:
        raise HTTPException(503, "Fitted churn model unavailable. Run python -m api.export_bundle")
    if not isinstance(bundle, dict) or not {"model", "pipeline", "features"} <= bundle.keys():
        raise HTTPException(503, "Invalid churn model bundle")
    return bundle


@app.post("/api/predict/batch")
def predict_batch(
    include_predictions: bool = Query(False, description="Include every customer's probability and risk level."),
    force: bool = Query(False, description="Recompute instead of returning the current process cache."),
    batch_size: int = Query(1000, ge=1, le=5000),
):
    """Score the entire official customer snapshot; cache only completed runs."""
    bundle = require_churn_model()
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

        required_columns = list(CustomerInput.model_fields)
        missing = set(required_columns) - set(customers.columns)
        if missing:
            raise HTTPException(503, "Official dataset is missing columns required by the prediction API")

        scores = []
        counts = {"low": 0, "medium": 0, "high": 0}
        for start in range(0, len(customers), batch_size):
            chunk = customers.iloc[start:start + batch_size].loc[:, required_columns]
            records = chunk.to_dict(orient="records")
            try:
                TypeAdapter(list[CustomerInput]).validate_python(records)
            except ValidationError as exc:
                raise HTTPException(503, "Official customer features failed input validation") from exc
            try:
                probabilities = score_probabilities(bundle, chunk)
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
        if sum(counts.values()) != len(customers):
            raise HTTPException(500, "Batch scoring was incomplete")
        summary = {
            "status": "scored",
            "source": "XGBoost inference on the official customer feature table",
            "data_kind": "synthetic-extended study snapshot; not live bank activity",
            "model_name": "XGBoost",
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
    bundle = require_churn_model()
    try:
        prob = float(score_probabilities(bundle, pd.DataFrame([p.model_dump()]))[0])
    except Exception as exc:
        raise HTTPException(500, "Model inference failed: verify training artifact compatibility") from exc
    app.state.inference_count += 1
    level = risk_level(prob)
    return {"customer_id": str(p.customer_id or ""),"churn_probability":prob,
            "predicted_churn":prob>=.5,"risk_level":level,"model_name":"XGBoost","signals":signals(p)}

@app.get("/api/customers")
def customers(search: str="",limit: int=Query(20,ge=1,le=100)):
    df=app.state.customers
    if df is None: raise HTTPException(503,"Customer dataset unavailable")
    matches=df[df.customer_id.astype(str).str.contains(search,regex=False)] if search else df
    records=matches.head(limit).drop(columns=["churn"],errors="ignore").fillna("").to_dict("records")
    for record in records:
        record["customer_id"]=str(record["customer_id"])
    return {"total":int(len(matches)),"items":records}

@app.get("/api/customers/{customer_id}")
def customer(customer_id:int):
    df=app.state.customers
    if df is None: raise HTTPException(503,"Customer dataset unavailable")
    found=df.loc[df.customer_id==customer_id]
    if found.empty: raise HTTPException(404,"Customer not found")
    record=found.drop(columns=["churn"],errors="ignore").iloc[0].to_dict()
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
    }

@app.get("/api/monitoring")
def monitoring():
    return {"model_loaded":app.state.bundle is not None,
            "predictions_this_process":app.state.inference_count,"drift_available":False,
            "performance_tracking_available":False}

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
