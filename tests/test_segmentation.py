"""Small fixtures prove protocol correctness, never experimental performance."""
import numpy as np
import pandas as pd
import pytest

from segmentation.features import ORIGINAL_FEATURES, SYNTHETIC_FEATURES, build_features, provenance
from segmentation.data import guard_tables


def raw_tables():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "churn": [0, 1, 0]})
    for col in ORIGINAL_FEATURES:
        customers[col] = [1, 2, 3] if col not in ("country", "gender") else ["A", "B", "A"]
    context = pd.DataFrame({"customer_id": [1, 2, 3],
        "relationship_start_date": ["2020-01-01", "2026-09-01", "2020-01-01"],
        "observation_start": ["2026-04-01", "2026-09-01", "2026-04-01"],
        "snapshot_boundary": ["2026-10-01"] * 3})
    accounts = pd.DataFrame({"account_id": ["a", "b", "c", "d"],
        "customer_id": [1, 1, 2, 3], "created_at": ["2020-01-01"] * 2 + ["2026-09-01", "2020-01-01"],
        "closed_at": [None] * 4, "account_type": ["everyday", "savings", "everyday", "everyday"],
        "balance": [1., 0., 2., 3.], "renewal_type": [None, "auto", None, None]})
    transactions = pd.DataFrame({"transaction_id": ["t1", "t2", "t3", "t4", "t5"],
        "account_id": ["a", "b", "a", "a", "a"], "transaction_datetime": ["2026-08-01"] * 4 + ["2026-10-01"],
        "amount": [10., 10., 3., 100., 999.], "direction": ["out", "in", "out", "in", "out"],
        "transaction_type": ["internal_transfer", "internal_transfer", "payment", "external_transfer", "payment"],
        "transaction_status": ["success"] * 5, "channel": ["mobile_app", "mobile_app", "mobile_app", "system", "mobile_app"],
        "transfer_id": ["pair", "pair", None, None, None]})
    login = pd.DataFrame({"login_id": ["l"], "customer_id": [1],
        "sign_in_datetime": ["2026-08-01"], "login_status": ["success"], "device_type": ["ios"]})
    complaints = pd.DataFrame({"complaint_id": ["x"], "customer_id": [1],
        "complaint_datetime": ["2026-08-01"], "resolved_datetime": ["2026-10-05"]})
    savings = pd.DataFrame({"account_id": ["b"], "started_at": ["2026-07-01"],
        "term_months": [6], "interest_rate": [.04], "early_withdrawal_rate": [.01], "init_balance": [10.]})
    return dict(kaggle_bank_churn=customers, customer_context=context, accounts=accounts,
                transactions=transactions, login=login, complaints=complaints, savings_cycles=savings)


def test_guard_excludes_snapshot_events_and_censors_future_resolution():
    tables, audit = guard_tables(raw_tables())
    assert len(tables["transactions"]) == 4
    assert tables["complaints"].resolved_datetime.isna().all()
    assert audit["transactions"]["excluded_outside_observation"] == 1


def test_guard_rejects_duplicate_customer_and_missing_context():
    tables = raw_tables()
    tables["kaggle_bank_churn"] = pd.concat([tables["kaggle_bank_churn"]] * 2)
    with pytest.raises(ValueError, match="unique"):
        guard_tables(tables)
    tables = raw_tables()
    tables["customer_context"] = tables["customer_context"].iloc[:2]
    with pytest.raises(ValueError, match="context"):
        guard_tables(tables)


def test_etl_reuse_deduplicates_transfers_and_excludes_passive_activity(tmp_path):
    from segmentation.data import SegmentationETL
    for name, table in raw_tables().items():
        table.to_csv(tmp_path / (name + ".csv"), index=False)
    frame = SegmentationETL(tmp_path).process(verbose=False)
    features = build_features(frame).set_index("customer_id")
    assert features.loc[1, "customer_initiated_tx_count"] == 2
    assert features.loc[1, "trans_amount_3m"] == 3
    assert features.loc[1, "monthly_transactions"] == pytest.approx(2 / (183 / 30.4375))
    assert features.loc[2, "recency_days"] == 30
    assert features.loc[2, "prior_history_observed"] == 0
    assert np.isnan(features.loc[2, "trans_trend_ratio"])
    assert features.loc[3, "monthly_transactions"] == 0
    assert features.customer_service_complaints.sum() == 1
    assert features.unresolved_complaints_count.sum() == 1
    assert len(features) == 3


def test_feature_allowlist_and_provenance():
    mapping = provenance()
    assert set(ORIGINAL_FEATURES).isdisjoint(SYNTHETIC_FEATURES)
    assert {mapping[c]["origin"] for c in ORIGINAL_FEATURES} == {"original"}
    assert {mapping[c]["origin"] for c in SYNTHETIC_FEATURES} == {"synthetic"}
    assert not {"churn", "customer_id", "savings_balance", "everyday_balance", "is_new_customer"} & set(SYNTHETIC_FEATURES)


def analytical_fixture(n=90):
    rng = np.random.default_rng(41)
    frame = pd.DataFrame({c: rng.gamma(2, 2, n) for c in ORIGINAL_FEATURES + SYNTHETIC_FEATURES})
    frame["country"] = np.resize(["France", "Germany", "Spain"], n)
    frame["gender"] = np.resize(["Female", "Male"], n)
    frame["customer_id"] = np.arange(n) + 100
    frame["churn"] = rng.integers(0, 2, n)
    return frame


def test_train_only_preprocessing_equal_group_energy_and_no_labels():
    from segmentation.preprocessing import GroupPreprocessor
    frame = analytical_fixture()
    train = frame.drop(columns=["customer_id", "churn"]).iloc[:60]
    prep = GroupPreprocessor().fit(train)
    transformed = prep.transform(train)
    for block in prep.slices_.values():
        assert transformed[:, block].var(axis=0).sum() == pytest.approx(.5)
    before = prep.transform(train)
    heldout = train.copy()
    heldout["age"] = 1e10
    heldout["country"] = "Unseen"
    heldout["recency_days"] = np.nan
    assert np.isfinite(prep.transform(heldout)).all()
    np.testing.assert_array_equal(before, prep.transform(train))
    for forbidden in ("customer_id", "churn", "wealth_deceleration_risk"):
        with pytest.raises(ValueError, match="Unapproved"):
            GroupPreprocessor().fit(train.assign(**{forbidden: 1}))


def test_whole_row_permutation_is_partition_isolated():
    from segmentation.kmeans import permute_synthetic
    frame = analytical_fixture()
    frame["monthly_logins"] = frame.recency_days * 2
    splits = pd.Series(np.resize(["train", "validation", "test"], len(frame)), index=frame.index)
    changed, donors = permute_synthetic(frame, splits, 7)
    np.testing.assert_array_equal(changed[ORIGINAL_FEATURES], frame[ORIGINAL_FEATURES])
    np.testing.assert_array_equal(changed.monthly_logins, changed.recency_days * 2)
    np.testing.assert_array_equal(splits.to_numpy(), splits.loc[donors].to_numpy())
    for part in splits.unique():
        assert sorted(donors[splits.eq(part)]) == sorted(frame.index[splits.eq(part)])
    repeat, _ = permute_synthetic(frame, splits, 7)
    pd.testing.assert_frame_equal(repeat, changed)


def test_model_determinism_and_reload(tmp_path):
    import joblib
    from segmentation.kmeans import select_model
    from segmentation.config import SegmentationConfig
    frame = analytical_fixture().drop(columns=["customer_id", "churn"])
    cfg = SegmentationConfig(k_values=[2, 3], seeds=[42, 43], n_init=3, silhouette_sample=70)
    model, scores = select_model(frame.iloc[:60], frame.iloc[60:], cfg)
    again, _ = select_model(frame.iloc[:60], frame.iloc[60:], cfg)
    np.testing.assert_array_equal(model.predict(frame), again.predict(frame))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(frame), loaded.predict(frame))
    assert set(scores.k) == {2, 3}
    assert model.named_steps["preprocess"].n_samples_fit_ == 60


def test_feature_population_identity_and_no_supervised_steps():
    from segmentation.features import feature_sets
    from segmentation.preprocessing import GroupPreprocessor
    frame = analytical_fixture()
    matrices = feature_sets(frame)
    for matrix in matrices.values():
        assert matrix.index.equals(frame.index)
        assert not {"churn", "customer_id"} & set(matrix)
        prep = GroupPreprocessor().fit(matrix)
        assert not {"TargetEncoder", "ClassBalancer", "SMOTE", "FeatureSelector"} & {
            type(component).__name__ for component in prep.groups_.values()}


def test_churn_wilson_and_label_invariant_agreement():
    from segmentation.evaluation import churn_summary, agreement, risk_separation
    labels = np.array([0, 0, 1, 1])
    churn = np.array([0, 0, 1, 1])
    summary = churn_summary(labels, churn)
    assert summary.customer_count.sum() == 4
    assert summary.churn_count.sum() == 2
    assert summary.iloc[0].ci_low == pytest.approx(0, abs=1e-10)
    assert summary.iloc[1].ci_high == pytest.approx(1)
    assert summary.iloc[1].lift == 2
    assert agreement(labels, 1 - labels)["ari"] == 1
    assert agreement(labels, 1 - labels)["nmi"] == 1
    assert risk_separation(labels, churn) == 1


def test_runner_outputs_and_heldout_population(tmp_path):
    from segmentation.runner import run_experiments
    from segmentation.config import SegmentationConfig
    cfg = SegmentationConfig(k_values=[2, 3], seeds=[42, 43], permutation_seeds=[1, 2],
        n_init=2, silhouette_sample=70, common_k=2, sensitivity_weights=[.25],
        output_dir=str(tmp_path / "results"), model_dir=str(tmp_path / "models"))
    frame = analytical_fixture(120)
    result = run_experiments(frame, cfg, {"test_size": .2, "val_size": .2, "random_state": 42},
                             audit={"fixture_only": True}, make_plots=False)
    assignments = pd.read_csv(tmp_path / "results" / "cluster_assignments.csv")
    assert len(assignments) == 120 and assignments.customer_id.is_unique
    assert assignments[["cluster_A", "cluster_B", "cluster_C"]].notna().all().all()
    assert len(assignments.loc[assignments.partition.eq("test")]) == 24
    assert set(result.experiment) >= {"A", "B", "C", "D_1", "D_2"}
    for name in ("comparison.csv", "churn_by_cluster.csv", "personas.csv", "common_k_comparison.csv",
                 "permutation_results.csv", "original_distribution_profiles.csv", "report.md", "run_manifest.json"):
        assert (tmp_path / "results" / name).exists()
    churn_summary = pd.read_csv(tmp_path / "results" / "churn_by_cluster.csv")
    assert (churn_summary.groupby("experiment").customer_count.sum() == 24).all()
    np.testing.assert_allclose(churn_summary.groupby("experiment").customer_share.sum(), 1)
    # Changing only held-out feature values and all labels must not affect any fit.
    import joblib
    from dataclasses import replace
    names = ["A", "B", "C", "D_1", "D_2", "C_weight_0.25", "B_reduced_dependency", "C_reduced_dependency"]
    before = {name: joblib.load(tmp_path / "models" / (name + ".joblib")) for name in names}
    heldout_ids = assignments.loc[assignments.partition.eq("test"), "customer_id"]
    frame.loc[frame.customer_id.isin(heldout_ids), "balance"] *= 1e8
    frame["churn"] = 0  # Also prove one-class evaluation reports unavailable statistics honestly.
    second = replace(cfg, output_dir=str(tmp_path / "second"), model_dir=str(tmp_path / "second_models"))
    run_experiments(frame, second, {"test_size": .2, "val_size": .2, "random_state": 42},
                    audit={"fixture_only": True}, make_plots=True)
    assert "unavailable" in (tmp_path / "second" / "report.md").read_text(encoding="utf-8")
    assert (tmp_path / "second" / "permutation_control.png").exists()
    for name in names:
        after = joblib.load(tmp_path / "second_models" / (name + ".joblib"))
        np.testing.assert_array_equal(before[name].named_steps["kmeans"].cluster_centers_, after.named_steps["kmeans"].cluster_centers_)


def test_label_free_split_and_cli_defaults():
    from preprocessing.splitting import DataSplitter
    from main import build_arg_parser
    frame = analytical_fixture()
    split = DataSplitter(stratify=False, val_size=.1).train_val_test_split(frame.drop(columns="churn"))
    assert set(split.train_df.customer_id).isdisjoint(split.test_df.customer_id)
    assert len(split.train_df) + len(split.val_df) + len(split.test_df) == len(frame)
    assert build_arg_parser().parse_args([]).mode == "compare"
    assert build_arg_parser().parse_args(["--mode", "segmentation"]).mode == "segmentation"


def test_observation_start_and_savings_overlap():
    tables = raw_tables()
    early = tables["transactions"].iloc[[2]].copy()
    early["transaction_id"] = "before_window"
    early["transaction_datetime"] = "2026-03-31"
    tables["transactions"] = pd.concat([tables["transactions"], early], ignore_index=True)
    early_login = tables["login"].iloc[[0]].copy()
    early_login["login_id"] = "before_join"
    early_login["customer_id"] = 2
    early_login["sign_in_datetime"] = "2026-08-01"
    tables["login"] = pd.concat([tables["login"], early_login], ignore_index=True)
    cycle = tables["savings_cycles"].iloc[[0]].copy()
    cycle["started_at"] = "2026-03-01"
    tables["savings_cycles"] = pd.concat([tables["savings_cycles"], cycle], ignore_index=True)
    clean, audit = guard_tables(tables)
    assert "before_window" not in set(clean["transactions"].transaction_id)
    assert "before_join" not in set(clean["login"].login_id)
    assert len(clean["savings_cycles"]) == 2  # March six-month term overlaps April.
    assert audit["transactions"]["excluded_outside_observation"] == 2


def test_invalid_internal_pair_and_foreign_key_fail_closed():
    tables = raw_tables()
    tables["transactions"].loc[1, "amount"] = 20
    with pytest.raises(ValueError, match="paired internal"):
        guard_tables(tables)
    tables = raw_tables()
    tables["transactions"].loc[1, "account_id"] = "unknown"
    with pytest.raises(ValueError, match="foreign key"):
        guard_tables(tables)
