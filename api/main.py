"""Bank customer churn inference and staff lookup API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import os
import joblib
import pandas as pd
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

@app.post("/api/predict")
def predict(p: CustomerInput):
    bundle=app.state.bundle
    if bundle is None: raise HTTPException(503,"No fitted model bundle. Run python -m api.export_bundle")
    if not isinstance(bundle,dict) or not {"model","pipeline","features"} <= bundle.keys():
        raise HTTPException(503,"Invalid model bundle")
    row=pd.DataFrame([p.model_dump()])
    row["churn"]=0
    try:
        transformed=bundle["pipeline"].transform(row)
        features=bundle["features"]
        if any(k not in transformed.columns for k in features):
            raise ValueError("Missing feature columns")
        X=transformed.loc[:,features]
        model=bundle["model"]
        if hasattr(model,"is_fitted"):
            prob=float(model.predict_proba(X)[0])
        else:
            prob=float(model.predict_proba(X)[0,1])
    except Exception as exc:
        raise HTTPException(500,"Model inference failed: verify training artifact compatibility") from exc
    app.state.inference_count += 1
    level="high" if prob>=.7 else "medium" if prob>=.3 else "low"
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
