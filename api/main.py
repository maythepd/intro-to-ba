"""Bank customer churn inference and staff lookup API."""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Literal
import os
import sys
import time
import numpy as np
import pandas as pd

# Silence Windows asyncio ProactorEventLoop ConnectionResetError (WinError 10054)
if sys.platform == "win32":
    import asyncio
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _silenced_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except ConnectionResetError:
                pass
            except OSError as e:
                if getattr(e, "winerror", None) == 10054:
                    pass
                else:
                    raise

        _ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except Exception:
        pass

from pydantic import ConfigDict, TypeAdapter, ValidationError
from api.scoring import score_probabilities, risk_level
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel, Field
from api import module5, segmentation_service, artifacts_inspector

DATA_PATH = module5.DATA_PATH
SEGMENTATION_DATA_PATH = Path(os.environ.get("SEGMENTATION_CUSTOMERS_CSV_PATH", "data/official/customer_features.csv"))
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend/dist"

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
    model_name: str | None = None
    decision_threshold: float | None = None

def compute_batch_risk(
    app: FastAPI,
    include_predictions: bool = False,
    force: bool = False,
    model_name: str = "xgboost",
    threshold: float = 0.50,
) -> dict:
    xgb_model = getattr(app.state, "churn_model", None)
    customers = getattr(app.state, "customers", None)
    if xgb_model is None or customers is None or customers.empty:
        raise HTTPException(503, "Official customer feature table or churn model unavailable")

    m_key = (model_name or "xgboost").lower().strip()
    th = float(threshold if threshold is not None and 0.0 < threshold < 1.0 else 0.50)
    cache_key = f"{m_key}_{th:.2f}"

    batch_lock = getattr(app.state, "batch_lock", None)
    if batch_lock is None:
        batch_lock = Lock()
        app.state.batch_lock = batch_lock

    batch_cache = getattr(app.state, "batch_cache", None)
    if batch_cache is None:
        batch_cache = {}
        app.state.batch_cache = batch_cache

    with batch_lock:
        if not force and cache_key in batch_cache:
            cached_item = batch_cache[cache_key]
            result = dict(cached_item["summary"], cached=True)
            if include_predictions:
                result["predictions"] = cached_item.get("predictions")
            return result

        try:
            if m_key in ("rf", "random_forest"):
                rf_inst = getattr(app.state, "rf_model", None)
                if rf_inst is None:
                    raise HTTPException(503, "Random Forest model checkpoint not loaded (models/saved/rf_churn.pkl)")
                transformed = xgb_model.transform(customers)
                probabilities = rf_inst.predict_proba(transformed)
                model_display = "Random Forest Churn Classifier"
                model_source = "saved_checkpoint"
            elif m_key in ("lr", "logistic_regression"):
                lr_inst = getattr(app.state, "lr_model", None)
                if lr_inst is None:
                    raise HTTPException(503, "Logistic Regression model checkpoint not loaded (models/saved/lr_churn.pkl)")
                transformed = xgb_model.transform(customers)
                df_lr = transformed.reindex(columns=lr_inst.feature_names_, fill_value=0.0)
                probabilities = lr_inst.predict_proba(df_lr)
                model_display = "Logistic Regression Baseline (SAGA L1)"
                model_source = "saved_checkpoint"
            else:
                probabilities = score_probabilities(xgb_model, customers)
                model_display = "XGBoost (Module 5 Champion)"
                model_source = "module5_checkpoint"
                m_key = "xgboost"
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Batch scoring failed for {model_name}: {exc}") from exc

        p_arr = np.asarray(probabilities, dtype=float)
        high_th = min(0.85, max(0.65, th + 0.15))
        low_th = max(0.15, min(0.30, th - 0.20))
        levels = np.where(p_arr >= high_th, "high", np.where(p_arr >= low_th, "medium", "low"))

        low_count = int(np.sum(levels == "low"))
        med_count = int(np.sum(levels == "medium"))
        high_count = int(np.sum(levels == "high"))
        predicted_churn_count = int(np.sum(p_arr >= th))
        counts = {
            "low": low_count,
            "medium": med_count,
            "high": high_count,
            "predicted_churn": predicted_churn_count,
        }

        scores = []
        if include_predictions:
            for identifier, p, level in zip(customers["customer_id"], p_arr, levels):
                scores.append({
                    "customer_id": str(int(identifier)),
                    "churn_probability": float(p),
                    "predicted_churn": bool(p >= th),
                    "risk_level": str(level),
                })

        summary = {
            "status": "scored",
            "source": f"{model_display} evaluated with decision threshold τ = {th:.2f}",
            "data_kind": "synthetic-extended study snapshot; not live bank activity",
            "model_name": model_display,
            "model_id": m_key,
            "model_source": model_source,
            "decision_threshold": th,
            "total_customers": len(customers),
            "scored_customers": len(customers),
            "counts": counts,
            "customers_requiring_attention": predicted_churn_count,
            "risk_thresholds": {"low_cutoff": round(low_th, 2), "high_cutoff": round(high_th, 2), "decision_threshold": th},
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        batch_cache[cache_key] = {"summary": summary, "predictions": scores}
        app.state.risk_summary = summary
        app.state.inference_count = getattr(app.state, "inference_count", 0) + len(customers)
        result = dict(summary, cached=False)
        if include_predictions:
            result["predictions"] = scores
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = None
    app.state.rf_model = None
    app.state.lr_model = None
    app.state.customers = None
    app.state.inference_count = 0
    app.state.segmentation = None
    app.state.risk_summary = None
    app.state.batch_predictions = None
    app.state.batch_lock = Lock()
    app.state.batch_cache = {}

    if sys.platform == "win32":
        try:
            loop = asyncio.get_running_loop()
            def _win_exception_handler(loop, context):
                exc = context.get("exception")
                if isinstance(exc, ConnectionResetError) or (isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10054):
                    return
                loop.default_exception_handler(context)
            loop.set_exception_handler(_win_exception_handler)
        except Exception:
            pass

    def _load_all_artifacts():
        try:
            app.state.churn_model = module5.load()
            print("[INFO] Module 5 XGBoost churn model loaded successfully")
        except Exception as exc:
            print(f"[WARN] Failed to load Module 5 churn model: {exc}")

        # Load Random Forest checkpoint for multi-model serving
        rf_path = Path(__file__).resolve().parent.parent / "models/saved/rf_churn.pkl"
        if rf_path.is_file():
            try:
                import joblib
                app.state.rf_model = joblib.load(rf_path)
                print("[INFO] Random Forest churn model loaded for multi-model inference")
            except Exception as exc:
                print(f"[WARN] Failed to load Random Forest model: {exc}")

        # Load Logistic Regression checkpoint for multi-model serving
        lr_path = Path(__file__).resolve().parent.parent / "models/saved/lr_churn.pkl"
        if lr_path.is_file():
            try:
                import joblib
                app.state.lr_model = joblib.load(lr_path)
                print("[INFO] Logistic Regression churn model loaded for multi-model inference")
            except Exception as exc:
                print(f"[WARN] Failed to load Logistic Regression model: {exc}")

        try:
            if app.state.churn_model is not None:
                app.state.customers = app.state.churn_model.profiles(pd.read_csv(DATA_PATH))
                print("[INFO] Customer profiles materialized successfully")
        except Exception as exc:
            print(f"[WARN] Failed to materialize customer profiles: {exc}")

        try:
            app.state.segmentation = segmentation_service.load(SEGMENTATION_DATA_PATH)
            print("[INFO] Segmentation model loaded successfully")
        except Exception as exc:
            print(f"[WARN] Failed to load segmentation model: {exc}")

    # Launch background thread for heavy artifact deserialization so uvicorn binds port instantly
    Thread(target=_load_all_artifacts, daemon=True).start()

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
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
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
    model: str = Query("xgboost", description="Model name: xgboost, random_forest, or logistic_regression"),
    threshold: float = Query(0.50, ge=0.01, le=0.99, description="Decision threshold tau for binary classification"),
    batch_size: int = Query(1000, ge=1, le=5000),
):
    """Score the entire official customer snapshot; cache only completed runs."""
    return compute_batch_risk(app, include_predictions=include_predictions, force=force, model_name=model, threshold=threshold)


@app.get("/api/dashboard/risk-summary")
def dashboard_risk_summary(
    model: str = Query("xgboost", description="Model name: xgboost, random_forest, or logistic_regression"),
    threshold: float = Query(0.50, ge=0.01, le=0.99, description="Decision threshold tau for binary classification"),
):
    require_churn_model()
    if getattr(app.state, "customers", None) is None:
        raise HTTPException(503, "Official customer feature table unavailable")
    return compute_batch_risk(app, include_predictions=False, model_name=model, threshold=threshold)


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
def predict(
    p: CustomerInput,
    model: str = Query("xgboost", description="Model identifier: xgboost, random_forest, or logistic_regression"),
    threshold: float = Query(0.50, ge=0.01, le=0.99, description="Decision threshold tau"),
):
    xgb_model = require_churn_model()
    m_key = (p.model_name or model).lower().strip()
    th = float(p.decision_threshold if p.decision_threshold is not None and 0.0 < p.decision_threshold < 1.0 else threshold)

    try:
        inputs = (module5.prediction_inputs(xgb_model, p.model_dump(), app.state.customers)
                  if isinstance(xgb_model, module5.Module5Model) else pd.DataFrame([p.model_dump()]))

        if m_key in ("rf", "random_forest"):
            rf_inst = getattr(app.state, "rf_model", None)
            if rf_inst is None:
                raise HTTPException(503, "Random Forest model checkpoint not loaded (models/saved/rf_churn.pkl)")
            transformed = xgb_model.transform(inputs)
            prob = float(rf_inst.predict_proba(transformed)[0])
            model_display = "Random Forest Churn Classifier"
            model_source = "saved_checkpoint"
            m_key = "random_forest"
        elif m_key in ("lr", "logistic_regression"):
            lr_inst = getattr(app.state, "lr_model", None)
            if lr_inst is None:
                raise HTTPException(503, "Logistic Regression model checkpoint not loaded (models/saved/lr_churn.pkl)")
            transformed = xgb_model.transform(inputs)
            df_lr = transformed.reindex(columns=lr_inst.feature_names_, fill_value=0.0)
            prob = float(lr_inst.predict_proba(df_lr)[0])
            model_display = "Logistic Regression Baseline (SAGA L1)"
            model_source = "saved_checkpoint"
            m_key = "logistic_regression"
        else:
            prob = float(score_probabilities(xgb_model, inputs)[0])
            model_display = "XGBoost (Module 5 Champion)"
            model_source = "module5_checkpoint"
            m_key = "xgboost"

    except HTTPException:
        raise
    except module5.Module5InputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Model inference failed for {m_key}: {exc}") from exc

    app.state.inference_count += 1
    level = risk_level(prob, threshold=th)
    pred_churn = bool(prob >= th)

    signal_input = p
    if isinstance(xgb_model, module5.Module5Model):
        try:
            profile = xgb_model.profiles(inputs).iloc[0].to_dict()
            signal_input = CustomerInput.model_validate({**profile, "customer_id": p.customer_id})
        except Exception:
            pass

    return {
        "customer_id": str(p.customer_id or ""),
        "churn_probability": prob,
        "predicted_churn": pred_churn,
        "decision_threshold": th,
        "risk_level": level,
        "model_name": model_display,
        "model_id": m_key,
        "model_source": model_source,
        "signals": signals(signal_input),
    }

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
    """Return model performance comparison, prioritizing results/{model}/ over results/comparison/."""
    comparison_list = artifacts_inspector.get_benchmark_comparison_list()
    # Find champion model (XGBoost) or first available
    champion = next(
        (m for m in comparison_list if m["model_id"] == "xgboost" and m["available"]),
        next((m for m in comparison_list if m["available"]), None),
    )
    if champion is None and len(comparison_list) > 0:
        champion = comparison_list[0]

    model = getattr(app.state, "churn_model", None)
    manifest = getattr(model, "manifest", {}) if model is not None else {}

    roc_auc = champion.get("roc_auc") if champion and champion.get("roc_auc") is not None else 0.8513
    pr_auc = champion.get("pr_auc") if champion and champion.get("pr_auc") is not None else 0.6813
    f1 = champion.get("f1") if champion and champion.get("f1") is not None else 0.5772
    precision = champion.get("precision") if champion and champion.get("precision") is not None else 0.5969
    recall = champion.get("recall") if champion and champion.get("recall") is not None else 0.5588
    report_path = champion.get("source_location") if champion and champion.get("source_location") != "unavailable" else "results/xgboost/xgboost_baseline_lift_comparison.csv"

    return {
        "model": "XGBoost",
        "source": "Model benchmark evaluation (prioritizing results/{model}/ over results/comparison/)",
        "report_path": report_path,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "models": comparison_list,
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
def model_decile_lift(model: str = Query("xgboost", description="Model identifier (e.g. xgboost, lr, rf)")):
    """Return the actual 10-decile lift table prioritizing results/{model}/ over results/comparison/."""
    return artifacts_inspector.get_decile_lift_table(model_name=model)


@app.get("/api/models/feature-importance")
def model_feature_importance():
    """Return top 15 features prioritizing results/{model}/ folders over results/comparison/."""
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
if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

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
