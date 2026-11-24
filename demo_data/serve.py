"""Opt-in lookup and per-batch scoring for the generated demo snapshot."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from api.scoring import risk_level
from demo_data.generate import BATCHES, summarize
from demo_data.validate import read_outputs, request_for, score_batch


def demo_router(output: Path) -> APIRouter:
    router = APIRouter(prefix="/api/demo", tags=["Simulated demo snapshot"])

    def dataset():
        try:
            manifest, frame, _ = read_outputs(output)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(503, f"Demo dataset is unavailable or invalid: {exc}") from exc
        return manifest, frame

    @router.get("/customers")
    def customers(search: str = "", limit: int = Query(20, ge=1, le=50)):
        _, frame = dataset()
        found = frame.loc[frame.customer_id.astype(str).str.contains(search, regex=False)]
        items = found.head(limit)[["customer_id", "batch_id", "country", "age"]]
        return {"total": len(found), "items": items.to_dict("records")}

    @router.get("/customers/{customer_id}")
    def customer(customer_id: int):
        manifest, frame = dataset()
        match = frame.loc[frame.customer_id.eq(customer_id)]
        if match.empty:
            raise HTTPException(404, "Demo customer not found")
        row = match.to_dict("records")[0]
        return {
            "batch_id": row["batch_id"], "scenario": row["scenario"],
            "snapshot": row["feature_snapshot"],
            "prediction_request": request_for(row, manifest["input_columns"]),
            "segmentation_request": request_for(row, manifest["input_columns"], segmentation=True),
        }

    @router.post("/batches/{batch_id}/score")
    def score(batch_id: str, request: Request, include_predictions: bool = False,
              threshold: float = Query(.50, ge=.01, le=.99)):
        if batch_id not in BATCHES:
            raise HTTPException(404, "Unknown demo batch")
        manifest, frame = dataset()
        rows = frame.loc[frame.batch_id.eq(batch_id)]
        if rows.empty:
            raise HTTPException(404, "Demo batch is empty")
        classifier = getattr(request.app.state, "churn_model", None)
        cluster_bundle = getattr(request.app.state, "segmentation", None)
        if classifier is None or cluster_bundle is None:
            raise HTTPException(503, "The fitted churn and segmentation artifacts must be loaded")
        try:
            probabilities, clusters = score_batch(rows, manifest["input_columns"], classifier, cluster_bundle)
        except (KeyError, ValueError) as exc:
            raise HTTPException(503, f"Demo model inputs are incompatible: {exc}") from exc
        baseline = manifest["monitoring"]["by_batch"][batch_id]
        result = {
            "batch_id": batch_id, "row_count": len(rows), "snapshot": manifest["snapshot"],
            "decision_threshold": threshold, "mean_churn_probability": float(probabilities.mean()),
            "predicted_churn_count": int((probabilities >= threshold).sum()),
            "cluster_counts": {
                name: {str(cluster): int((values == cluster).sum()) for cluster in sorted(set(values))}
                for name, values in clusters.items()
            },
            "hierarchical_method": "nearest_centroid_surrogate_not_ward_prediction",
            "monitoring": {"batch": summarize(rows),
                           "source_reference": manifest["monitoring"]["source_reference"],
                           "matched_donor": baseline["matched_donor"],
                           "synthetic_baseline": baseline.get("synthetic_baseline", baseline["matched_donor"])},
            "performance_metrics": manifest["labels"]["performance_metrics"],
        }
        if include_predictions:
            result["predictions"] = [
                {"customer_id": int(customer_id), "churn_probability": probability,
                 "predicted_churn": probability >= threshold, "risk_level": risk_level(probability),
                 **{name: int(values[index]) for name, values in clusters.items()}}
                for index, (customer_id, probability) in enumerate(zip(rows.customer_id, probabilities.tolist()))
            ]
        return result

    return router
