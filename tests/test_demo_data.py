import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

from api import module5, segmentation_service
from api.main import SEGMENTATION_DATA_PATH
from demo_data.generate import (
    BATCHES,
    aggregate,
    batch_sizes,
    generate,
    load_sources,
    sample_count,
)
from demo_data.serve import demo_router
from demo_data.synthesize import synthesize
from demo_data.validate import (
    check_dataset,
    check_novelty,
    read_outputs,
    score_batch,
    validate,
)
from preprocessing.pipeline import FeatureEngineeringPipeline


@pytest.mark.parametrize(("n_base", "expected"), [(1, 10), (999, 9990), (1000, 10000), (10000, 10000)])
def test_sample_count_is_calculated_from_customers(n_base, expected):
    assert sample_count(n_base) == expected
    assert sum(batch_sizes(expected).values()) == expected


def test_remainder_goes_to_first_batches_in_ingestion_order():
    assert list(batch_sizes(12).items()) == list(zip(BATCHES, (3, 3, 2, 2, 2)))
    assert sample_count(10000, multiplier=100, max_samples=20000) == 10000
    with pytest.raises(ValueError, match="positive"):
        sample_count(0)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    files = {
        "kaggle_bank_churn": """customer_id,credit_score,country,gender,age,tenure,balance,products_number,credit_card,active_member,estimated_salary,churn
1,660,France,Female,32,4,400,2,1,1,15000,1
2,720,Spain,Male,25,0,300,1,1,1,12000,0
""",
        "customer_context": """customer_id,relationship_start_date,observation_start,snapshot_boundary
1,2022-09-01,2026-04-01,2026-10-01
2,2026-09-30T18:00:00,2026-09-30T18:00:00,2026-10-01
""",
        "accounts": """account_id,customer_id,account_type,created_at,closed_at,balance,renewal_type,payout_account_id
A1,1,everyday,2022-09-01,,300,,
A2,1,savings,2026-09-28,,100,auto,A1
A3,2,everyday,2026-09-30T18:00:00,,300,,
""",
        "transactions": """transaction_id,account_id,transaction_datetime,amount,direction,transaction_type,transaction_status,channel,transfer_id
T1,A1,2026-09-28T12:00:15,100,out,internal_transfer,success,mobile_app,PAIR
T2,A2,2026-09-28T12:00:15,100,in,internal_transfer,success,mobile_app,PAIR
T3,A3,2026-09-30T18:10:00,300,in,external_transfer,success,system,
T4,A1,2026-09-29T12:00:10,20,out,payment,success,branch,
T5,A1,2026-09-30T12:00:00,10,in,external_transfer,success,mobile_app,
""",
        "login": """login_id,customer_id,sign_in_datetime,login_status,device_type
L1,1,2026-09-28T12:00:00,success,android
L2,1,2026-09-29T12:00:00,success,desktop
""",
        "complaints": """complaint_id,customer_id,complaint_datetime,resolved_datetime
C1,1,2026-07-01,
""",
        "savings_cycles": """account_id,started_at,term_months,init_balance,interest_rate,early_withdrawal_rate
A2,2026-09-28,3,100,0.035,0.001
""",
    }
    for name, contents in files.items():
        (root / f"{name}.csv").write_text(contents, encoding="utf-8")
    return root


@pytest.mark.parametrize(("file", "contents", "match"), [
    ("accounts.csv", None, "Missing source table"),
    ("login.csv", "login_id,customer_id,sign_in_datetime,login_status,device_type\n", "Empty source table"),
    ("transactions.csv", "account_id\nA1\n", "missing columns"),
])
def test_invalid_sources_are_reported(source, file, contents, match):
    path = source / file
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises((OSError, ValueError), match=match):
        load_sources(source)


def test_etl_counts_pairs_once_and_excludes_passive_activity(source):
    model_manifest = json.loads(module5.MANIFEST_PATH.read_text(encoding="utf-8"))
    tables, reported = load_sources(source)
    rows = aggregate(tables, model_manifest["input_columns"])
    assert reported["transactions.csv"]["unique_customer_ids"] == 2
    assert rows.loc[1, "customer_initiated_tx_count"] == 2  # internal transfer and payment
    assert rows.loc[1, "total_transactions"] == 3  # passive receipt is included in posted volume
    assert rows.loc[1, "seg_recency_days"] == pytest.approx(1.0)
    assert rows.loc[2, "customer_initiated_tx_count"] == 0
    assert rows.loc[2, "recency_days"] == rows.loc[2, "exposure_days"] == .25
    assert rows.loc[2, "seg_recency_days"] == .25
    assert rows.loc[2, "trans_trend_ratio"] == 1
    assert pd.isna(rows.loc[2, "initiated_failure_rate"])


@pytest.fixture
def demo(source, tmp_path):
    out = tmp_path / "demo"
    with (
        patch.object(FeatureEngineeringPipeline, "fit", side_effect=AssertionError("No preprocessing fit")),
        patch.object(segmentation_service.BalancedFeatures, "fit", side_effect=AssertionError("No segmentation fit")),
        patch.object(XGBClassifier, "fit", side_effect=AssertionError("No classifier fit")),
        patch.object(KMeans, "fit", side_effect=AssertionError("No KMeans fit")),
    ):
        result = generate(source, out, seed=42)
    return out, result


def test_demo_is_reproducible_and_examples_are_exact(source, demo, tmp_path):
    out, manifest = demo
    repeat = generate(source, tmp_path / "repeat", seed=42)
    assert manifest == repeat
    _, frame, examples = read_outputs(out)
    assert len(frame) == len(set(frame.customer_id)) == 20
    assert set(frame.customer_id).isdisjoint({1, 2})
    assert len(examples) == 20
    assert any(entry["raw_features"]["avg_resolution_delay_days"] is None for entry in examples)
    assert "churn" not in frame and "source_customer_id" not in frame
    assert manifest["inference"]["status"] == "passed", manifest["inference"]
    broken = [dict(entry) for entry in examples]
    broken[0]["credit_score"] += 1
    with pytest.raises(ValueError, match="exact requests"):
        check_dataset(frame, broken, manifest, [1, 2])


def test_normal_profiles_are_new_even_without_customer_ids(source, demo):
    out, manifest = demo
    _, frame, _ = read_outputs(out)
    tables, _ = load_sources(source)
    reference = aggregate(tables, manifest["input_columns"])
    normal = frame[frame.scenario.eq("normal")]
    report = check_novelty(normal, reference, manifest["input_columns"])
    assert report["matches_source_profiles"] == report["matches_source_classifier_inputs"] == 0
    assert manifest["validation"]["novelty"] == report
    with pytest.raises(ValueError, match="Copied source feature vectors"):
        check_novelty(reference, reference, manifest["input_columns"])


def test_event_synthesis_changes_identity_and_time_but_keeps_transfer_pairs(source):
    original, _ = load_sources(source)
    tx = original["transactions"]
    external = pd.DataFrame([
        ["T6", "A1", pd.Timestamp("2026-09-30 20:00"), 18.0, "out", "external_transfer",
         "success", "mobile_app", "BANK-PAIR"],
        ["T7", "A3", pd.Timestamp("2026-09-30 20:00"), 18.0, "in", "external_transfer",
         "success", "mobile_app", "BANK-PAIR"],
    ], columns=tx.columns)
    original["transactions"] = pd.concat([tx, external], ignore_index=True)
    generated = synthesize(original, [1001, 1002, 1003], [1, 1, 2], [1, 1, 2], 1800, np.random.default_rng(42))
    event = generated["transactions"]
    assert event.transaction_id.is_unique and set(event.transaction_id).isdisjoint(tx.transaction_id)
    assert set(event.transaction_datetime).isdisjoint(original["transactions"].transaction_datetime)
    assert set(generated["login"].login_id).isdisjoint(original["login"].login_id)
    paired = event.loc[event.transfer_id.notna()]
    assert paired.transfer_id.nunique() == 3
    for _, pair in paired.groupby("transfer_id"):
        assert len(pair) == 2 and set(pair.direction) == {"in", "out"}
        assert pair.amount.nunique() == pair.transaction_datetime.nunique() == 1
    accounts = generated["accounts"].set_index("account_id")
    assert event.transaction_datetime.ge(event.account_id.map(accounts.created_at)).all()
    assert event.transaction_datetime.lt(pd.Timestamp("2026-10-01")).all()


def test_complete_raw_request_never_reads_official_rows(demo):
    class RejectOfficialRead:
        def __getattr__(self, name):
            raise AssertionError(f"Read official snapshot: {name}")

    out, _ = demo
    _, frame, requests = read_outputs(out)
    model = module5.load()
    request = requests[0]
    request = {**request, "customer_id": 15565701}  # An official ID must also be ignored for full raw input.
    actual = module5.prediction_inputs(model, request, RejectOfficialRead())
    expected = frame.iloc[[0]][model.input_columns]
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_real_api_and_uncached_demo_batch_match_local_inference(source, demo):
    out, manifest = demo
    result = validate(source, out)
    parity = result["inference"]["api_parity"]
    assert parity["classifier"] == parity["kmeans"] == parity["hierarchical"] == 20
    assert parity["max_probability_delta"] <= 1e-7

    _, frame, requests = read_outputs(out)
    app = FastAPI()
    app.include_router(demo_router(out))
    app.state.churn_model = module5.load()
    app.state.segmentation = segmentation_service.load(SEGMENTATION_DATA_PATH)
    with TestClient(app) as client:
        customer = requests[0]["customer_id"]
        assert client.get(f"/api/demo/customers/{customer}").json()["prediction_request"] == requests[0]
        assert client.get("/api/demo/customers/1").status_code == 404
        for batch in ("normal_01", "friction_shift"):
            rows = frame.loc[frame.batch_id.eq(batch)]
            probs, clusters = score_batch(rows, manifest["input_columns"], app.state.churn_model,
                                          app.state.segmentation)
            resp = client.post(f"/api/demo/batches/{batch}/score?include_predictions=true&threshold=0.6")
            assert resp.status_code == 200, resp.text
            report = resp.json()
            assert report["row_count"] == 4
            assert report["performance_metrics"] == "insufficient_labeled_data"
            assert [record["customer_id"] for record in report["predictions"]] == rows.customer_id.tolist()
            assert [record["churn_probability"] for record in report["predictions"]] == pytest.approx(probs)
            assert [record["kmeans"] for record in report["predictions"]] == clusters["kmeans"].tolist()
        (out / "customer_features.csv").write_text("corrupt artifact", encoding="utf-8")
        assert client.post("/api/demo/batches/normal_01/score").status_code == 503
