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
    yield

app = FastAPI(title="VioletBank Analytics API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("CORS_ORIGINS","http://localhost:5173").split(",") if x.strip()],
    allow_methods=["GET","POST"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health():
    return {"status":"ready" if app.state.bundle is not None else "model_unavailable",
            "model_loaded":app.state.bundle is not None,"customers_loaded":app.state.customers is not None}

def signals(p: CustomerInput) -> list[dict[str,str]]:
    result=[]
    for flag,title,description in [
        (p.trans_trend_ratio < .75,"Giao dịch giảm","Xu hướng giao dịch giảm hơn 25%."),
        (p.customer_service_complaints >= 2 or p.has_unresolved_complaint == 1,"Khiếu nại dịch vụ","Cần xử lý các khiếu nại."),
        (p.recency_days >= 45,"Ít tương tác","Không tương tác ít nhất 45 ngày."),
        (p.products_number >= 3,"Nhiều sản phẩm","Cần đánh giá trải nghiệm sử dụng nhiều sản phẩm."),
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
    return {"model":"XGBoost","source":"Repository ablation report, not deployment performance",
            "roc_auc":.9999,"pr_auc":.9997,"f1":.9914,"recall":.9877}

@app.get("/api/monitoring")
def monitoring():
    return {"model_loaded":app.state.bundle is not None,
            "predictions_this_process":app.state.inference_count,"drift_available":False,
            "performance_tracking_available":False}

@app.get("/api/segments")
def segments():
    raise HTTPException(503,"Segmentation model has not been provided")
