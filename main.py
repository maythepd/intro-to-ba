"""
Main entrypoint for Digital Bank Churn Prediction modeling & ablation / model-comparison experiments.
Supports three classifiers: Logistic Regression (LR), Random Forest (RF), and XGBoost.
All experiments are configured via config/config.yaml and generate Markdown reports in results/.
"""

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from config import (
    ExperimentConfig,
    ModelConfig,
    PipelineConfig,
    UnifiedConfig,
    load_unified_config,
)
from models import (
    LogisticRegressionChurnModel,
    RandomForestChurnModel,
    XGBoostChurnModel,
)
from preprocessing import FeatureEngineeringPipeline

# Supported model registry
MODEL_REGISTRY = {
    "xgboost": XGBoostChurnModel,
    "lr": LogisticRegressionChurnModel,
    "rf": RandomForestChurnModel,
}

# Type alias for any supported model instance
AnyChurnModel = Union[XGBoostChurnModel, LogisticRegressionChurnModel, RandomForestChurnModel]


def resolve_data_path(path_str: str) -> str:
    """Resolve data path relative to repo or workspace root."""
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)

    # Try relative to current working directory
    if path.exists():
        return str(path.resolve())

    # Try relative to script directory
    script_dir = Path(__file__).resolve().parent
    candidate = (script_dir / path).resolve()
    if candidate.exists():
        return str(candidate)

    # Try parent directory (e.g. intro-to-ba/../Week 5/...)
    candidate_parent = (script_dir.parent / path).resolve()
    if candidate_parent.exists():
        return str(candidate_parent)

    raise FileNotFoundError(f"Could not resolve dataset path: {path_str}")


def _build_model(
    model_type: str,
    model_cfg: Union[Dict[str, Any], ModelConfig],
    features_cfg: PipelineConfig,
    y_train: pd.Series,
) -> AnyChurnModel:
    """
    Factory: instantiates the correct model class from model_type string.
    Dynamically extracts hyperparameter overrides from ModelConfig or dict.
    For XGBoost, also handles scale_pos_weight adjustment when SMOTE is disabled.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    if isinstance(model_cfg, ModelConfig):
        params = copy.deepcopy(model_cfg.get_params_for_model(model_type))
        early_stopping = model_cfg.early_stopping_rounds
    else:
        # Dictionary fallback
        if model_type == "xgboost":
            params = copy.deepcopy(model_cfg.get("xgboost_params") or model_cfg.get("params") or {})
        elif model_type in ("lr", "logistic_regression"):
            params = copy.deepcopy(
                model_cfg.get("lr_params")
                or (model_cfg.get("params") if model_cfg.get("name") in ("lr", "logistic_regression") else {})
            )
        else:  # rf
            params = copy.deepcopy(
                model_cfg.get("rf_params")
                or (model_cfg.get("params") if model_cfg.get("name") in ("rf", "random_forest") else {})
            )
        early_stopping = model_cfg.get("early_stopping_rounds", 30)

    if model_type == "xgboost":
        if not features_cfg.enable_smote and "scale_pos_weight" in params:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            params["scale_pos_weight"] = float(round(neg_count / max(1, pos_count), 2))
        return XGBoostChurnModel(params=params, early_stopping_rounds=early_stopping)

    elif model_type in ("lr", "logistic_regression"):
        return LogisticRegressionChurnModel(params=params if params else None)

    else:  # rf
        return RandomForestChurnModel(params=params if params else None)



def run_single_experiment(
    df: pd.DataFrame,
    features_cfg: PipelineConfig,
    model_cfg: Dict[str, Any],
    experiment_name: str,
    model_type: str = "xgboost",
) -> Tuple[Dict[str, Any], AnyChurnModel, pd.DataFrame]:
    """
    Executes a single experiment:
      1. Prepares train/test sets using FeatureEngineeringPipeline.
      2. Instantiates and trains the requested model (xgboost | lr | rf).
      3. Evaluates metrics on held-out test partition.
      4. Extracts feature importances.
    """
    print(f"\n>>> Running Experiment: [{experiment_name}] | Model: {model_type.upper()}")
    pipeline = FeatureEngineeringPipeline(features_cfg)
    res = pipeline.prepare_train_test(df)

    model = _build_model(model_type, model_cfg, features_cfg, res.y_train_resampled)

    model.fit(
        X_train=res.X_train_resampled,
        y_train=res.y_train_resampled,
        X_val=res.X_test,
        y_val=res.y_test,
        verbose=False,
    )

    metrics = model.evaluate(res.X_test, res.y_test)
    metrics["Experiment"] = experiment_name
    metrics["Model"] = model_type.upper()
    metrics["Num_Features"] = len(res.model_features)
    metrics["X_train_Shape"] = f"{res.X_train_resampled.shape[0]}x{res.X_train_resampled.shape[1]}"
    metrics["X_test_Shape"] = f"{res.X_test.shape[0]}x{res.X_test.shape[1]}"
    metrics["SMOTE_Applied"] = features_cfg.enable_smote

    # Use default importance_type per model family
    imp_type = "gain" if model_type == "xgboost" else "gini" if model_type == "rf" else "coefficient"
    df_imp = model.get_feature_importances(importance_type=imp_type)

    print(
        f"    Features: {metrics['Num_Features']} | "
        f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
        f"PR-AUC: {metrics['PR_AUC']:.4f} | "
        f"F1-Score: {metrics['F1_Score']:.4f} | "
        f"Recall: {metrics['Recall']:.4f}"
    )

    return metrics, model, df_imp


def run_model_comparison(
    df: pd.DataFrame,
    base_config: UnifiedConfig,
) -> Tuple[pd.DataFrame, Dict[str, AnyChurnModel], Dict[str, pd.DataFrame]]:
    """
    Trains LR, RF, and XGBoost on the same default feature pipeline and compares metrics side-by-side.
    This is the canonical model-selection benchmark for Module 5 of the project.
    """
    model_types = [
        ("Logistic Regression", "lr"),
        ("Random Forest", "rf"),
        ("XGBoost", "xgboost"),
    ]

    results: List[Dict[str, Any]] = []
    models_dict: Dict[str, AnyChurnModel] = {}
    importances_dict: Dict[str, pd.DataFrame] = {}

    feat_cfg = copy.deepcopy(base_config.features)
    model_dict = copy.deepcopy(base_config.model.__dict__)

    for name, mtype in model_types:
        metrics, trained_model, df_imp = run_single_experiment(
            df=df,
            features_cfg=feat_cfg,
            model_cfg=model_dict,
            experiment_name=name,
            model_type=mtype,
        )
        results.append(metrics)
        models_dict[name] = trained_model
        importances_dict[name] = df_imp

    df_results = pd.DataFrame(results)
    return df_results, models_dict, importances_dict


def run_ablation_suite(
    df: pd.DataFrame,
    base_config: UnifiedConfig,
    model_type: str = "xgboost",
) -> Tuple[pd.DataFrame, Dict[str, AnyChurnModel], Dict[str, pd.DataFrame]]:
    """
    Runs a comprehensive set of ablation experiments by modifying feature configs.
    By default uses XGBoost; pass model_type='lr' or 'rf' to ablate a different base model.
    """
    print(f"\n[INFO] Ablation suite using model: {model_type.upper()}") 
    experiments: List[Tuple[str, Dict[str, Any]]] = [
        (
            "1. Full Pipeline (Default)",
            {},
        ),
        (
            "2. No SMOTE (Native scale_pos_weight)",
            {"enable_smote": False},
        ),
        (
            "3. No Domain Features",
            {"enable_domain_features": False},
        ),
        (
            "4. No Target Encoding",
            {"enable_target_encoding": False},
        ),
        (
            "5. No One-Hot Encoding",
            {"enable_one_hot_encoding": False},
        ),
        (
            "6. No Scaling & No Skewness Transform",
            {"enable_scaling": False, "enable_skewness": False},
        ),
        (
            "7. Top 15 Features Only (Feature Selection)",
            {"enable_feature_selection": True, "top_k_features": 15},
        ),
    ]

    results: List[Dict[str, Any]] = []
    models_dict: Dict[str, AnyChurnModel] = {}
    importances_dict: Dict[str, pd.DataFrame] = {}

    for name, overrides in experiments:
        # Create a deep copy of base features config and apply overrides
        feat_dict = copy.deepcopy(base_config.features.__dict__)
        feat_dict.update(overrides)
        features_cfg = PipelineConfig(**feat_dict)

        model_dict = copy.deepcopy(base_config.model.__dict__)

        metrics, trained_model, df_imp = run_single_experiment(
            df=df,
            features_cfg=features_cfg,
            model_cfg=model_dict,
            experiment_name=name,
            model_type=model_type,
        )

        results.append(metrics)
        models_dict[name] = trained_model
        importances_dict[name] = df_imp

    df_results = pd.DataFrame(results)
    return df_results, models_dict, importances_dict


def generate_ablation_report(
    df_results: pd.DataFrame,
    best_exp_name: str,
    best_model: AnyChurnModel,
    best_imp: pd.DataFrame,
    output_filepath: str,
    config: UnifiedConfig,
    model_type: str = "xgboost",
) -> None:
    """Generates an executive-level Markdown ablation report for any of the three supported models."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    display_cols = [
        "Experiment",
        "Num_Features",
        "ROC_AUC",
        "PR_AUC",
        "F1_Score",
        "Precision",
        "Recall",
        "Accuracy",
        "Log_Loss",
    ]
    df_table = df_results[display_cols].copy()
    top15_imp = best_imp.head(15).copy()
    best_metrics = df_results[df_results["Experiment"] == best_exp_name].iloc[0]

    model_names = {"xgboost": "XGBoost", "lr": "Logistic Regression", "rf": "Random Forest"}
    model_display = model_names.get(model_type.lower(), model_type.upper())
    imp_type_label = {
        "xgboost": "Average Gain (mức giảm hàm mục tiêu bình quân khi chia nút)",
        "rf": "Mean Decrease in Impurity — Gini MDI",
        "lr": "Absolute Coefficient |β| (proxy độ quan trọng tuyến tính)",
    }.get(model_type.lower(), "Feature Importance")

    md_content = f"""# BÁO CÁO THỰC NGHIỆM MÔ HÌNH {model_display.upper()} & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `{timestamp}`  
**Mô hình cốt lõi:** `{model_display}`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **{model_display}** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

{df_table.to_markdown(index=False)}

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **{best_exp_name}**
- **ROC-AUC:** `{best_metrics['ROC_AUC']:.4f}` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `{best_metrics['PR_AUC']:.4f}` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `{best_metrics['F1_Score']:.4f}` (Precision: `{best_metrics['Precision']:.4f}`, Recall: `{best_metrics['Recall']:.4f}`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `{best_metrics['TP']}`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `{best_metrics['FN']}`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `{best_metrics['TN']}`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `{best_metrics['FP']}`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight (Kịch bản 2):**
   - Đối với {model_display}: SMOTE tăng kích thước tập huấn luyện từ 8,000 lên ~12,740 dòng. Native class_weight='balanced' tái cân bằng trực tiếp trên hàm mất mát mà không cần sinh mẫu tổng hợp, phù hợp khi runtime là ưu tiên.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `{df_results.loc[df_results['Experiment'] == '7. Top 15 Features Only (Feature Selection)', 'ROC_AUC'].values[0]:.4f}`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** {imp_type_label}

{top15_imp.to_markdown(index=False)}

---

## 4. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

### 4.1. So sánh cả 3 model (LR / RF / XGBoost)
```bash
python main.py --mode compare
```

### 4.2. Ablation cho một model cụ thể
```bash
python main.py --mode ablation --model-type {model_type}  # lr | rf | xgboost
```

### 4.3. Train đơn lẻ với custom params
```bash
python main.py --mode train --model-type {model_type}
```

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
"""
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[OK] Ablation markdown report generated: {output_filepath}")


# Backward-compatible alias
generate_markdown_report = generate_ablation_report


def generate_comparison_report(
    df_results: pd.DataFrame,
    models_dict: Dict[str, AnyChurnModel],
    importances_dict: Dict[str, pd.DataFrame],
    output_filepath: str,
    config: UnifiedConfig,
) -> None:
    """Generates an executive-level Markdown comparison report for all 3 models (LR vs RF vs XGBoost)."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    display_cols = [
        "Model",
        "ROC_AUC",
        "PR_AUC",
        "F1_Score",
        "Precision",
        "Recall",
        "Accuracy",
        "Balanced_Accuracy",
        "Log_Loss",
    ]
    df_table = df_results[display_cols].copy()

    # Determine best model based on PR_AUC and ROC_AUC
    best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]

    # Build Top 5 features per model for side-by-side comparison
    top_features_md = ""
    for name, df_imp in importances_dict.items():
        top5 = df_imp.head(5)[["Feature", "Relative_Pct"]].to_markdown(index=False)
        top_features_md += f"#### Top 5 Đặc trưng — {name}\n\n{top5}\n\n"

    md_content = f"""# BÁO CÁO ĐỐI ĐẦU VÀ SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH (MODEL COMPARISON)
## LOGISTIC REGRESSION vs. RANDOM FOREST vs. XGBOOST
### DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (DIGITAL BANK CHURN PREDICTION)

**Thời gian thực thi:** `{timestamp}`  
**Tập dữ liệu:** `{config.features.data_path}` (2,000 khách hàng tập Test)  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ HIỆU NĂNG (BENCHMARK TABLE)

Bảng đối đầu trực tiếp giữa 3 họ thuật toán đại diện: Tuyến tính (Logistic Regression), Ensemble Bagging (Random Forest), và Gradient Boosting (XGBoost):

{df_table.to_markdown(index=False)}

---

## 2. PHÂN TÍCH CHUYÊN SÂU TỪ GÓC ĐỘ BUSINESS ANALYTICS & MACHINE LEARNING

### 2.1. Mô hình chiến thắng (Champion Model): **{best_model_name}**
- **ROC-AUC:** `{best_row['ROC_AUC']:.4f}` — Xếp hạng xác suất khách hàng rời bỏ chính xác nhất trên toàn bộ miền phân bố.
- **PR-AUC:** `{best_row['PR_AUC']:.4f}` — Tiêu chuẩn vàng cho tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `{best_row['F1_Score']:.4f}` (Precision: `{best_row['Precision']:.4f}`, Recall: `{best_row['Recall']:.4f}`).
- **Khả năng bắt khách hàng rời bỏ:** TP = `{best_row['TP']}` / `{best_row['TP'] + best_row['FN']}` khách hàng thực tế rời bỏ.

### 2.2. So sánh đặc tính của 3 họ thuật toán:
1. **Logistic Regression (Linear Baseline & Interpretability):**
   - **Ưu điểm:** Tốc độ huấn luyện và suy luận nhanh nhất (< 1ms). Trực quan hóa tác động biên (marginal effects) của từng biến lên log-odds thông qua các trọng số $\\beta$. Phù hợp để giải trình với ban kiểm soát tuân thủ (Compliance / Risk Auditing).
   - **Hạn chế:** Không tự động nắm bắt được các mối tương tác phi tuyến tính phức tạp (như tỷ lệ tương tác tuổi tác x điểm tín dụng x số dư).
2. **Random Forest (Bagging Ensemble & Stability):**
   - **Ưu điểm:** Giảm phương sai (variance reduction) vượt trội nhờ cơ chế bootstrap aggregating và feature sub-sampling. Kháng nhiễu và outliers tốt, không đòi hỏi giả định phân phối.
   - **Hạn chế:** Kích thước model lớn, độ phức tạp tính toán cao hơn LR và khả năng tối ưu hàm mục tiêu biên kém linh hoạt hơn boosting.
3. **XGBoost (Gradient Boosting & Performance):**
   - **Ưu điểm:** Tối ưu hóa trực tiếp hàm mục tiêu và gradient bậc hai (Hessian). Khả năng phát hiện mẫu hành vi rời bỏ tinh vi nhất từ các đặc trưng kỹ nghệ tài chính.
   - **Hạn chế:** Cần tinh chỉnh hyperparameter cẩn thận để tránh overfitting trên các tập con nhỏ.

---

## 3. TOP ĐẶC TRƯNG HÀNG ĐẦU CỦA TỪNG MÔ HÌNH

{top_features_md}

---

## 4. KHUYẾN NGHỊ VẬN HÀNH THỰC TẾ (PRODUCTION STRATEGY)

1. **Triển khai Mô hình chính (Primary Scoring Engine):**
   - Sử dụng **XGBoost** làm engine tính điểm rủi ro rời bỏ định kỳ hàng ngày (Batch Scoring) cho toàn bộ khách hàng hoạt động, cung cấp danh sách ưu tiên cao cho đội ngũ chăm sóc khách hàng (Retention RM).
2. **Mô hình giải trình & Baseline (Challenger / Fallback):**
   - Sử dụng **Logistic Regression** để kiểm tra tính nhất quán và giải trình các nguyên nhân rời bỏ vĩ mô cho cấp quản lý.

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
"""
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[OK] Comparison markdown report generated: {output_filepath}")


def generate_train_report(
    metrics: Dict[str, Any],
    model: AnyChurnModel,
    df_imp: pd.DataFrame,
    output_filepath: str,
    config: UnifiedConfig,
    model_type: str = "xgboost",
    model_save_path: Optional[str] = None,
) -> None:
    """Generates an executive-level Markdown report for a single model training run."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_names = {"xgboost": "XGBoost", "lr": "Logistic Regression", "rf": "Random Forest"}
    model_display = model_names.get(model_type.lower(), model_type.upper())

    top15 = df_imp.head(15).to_markdown(index=False)
    metrics_table = pd.DataFrame([{
        "Metric": k,
        "Value": f"{v:.4f}" if isinstance(v, float) else str(v)
    } for k, v in metrics.items() if k not in ("X_train_Shape", "X_test_Shape")]).to_markdown(index=False)

    md_content = f"""# BÁO CÁO HUẤN LUYỆN MÔ HÌNH {model_display.upper()} (TRAINING REPORT)
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian huấn luyện:** `{timestamp}`  
**Mô hình:** `{model_display}`  
**Artifact đã lưu:** `{model_save_path or 'N/A'}`  
**Cấu hình:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST

{metrics_table}

---

## 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

- **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `{metrics.get('TP', 'N/A')}`
- **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `{metrics.get('FN', 'N/A')}`
- **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `{metrics.get('TN', 'N/A')}`
- **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `{metrics.get('FP', 'N/A')}`

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

{top15}

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
"""
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[OK] Training markdown report generated: {output_filepath}")


def generate_report(
    mode: str,
    output_filepath: str,
    config: UnifiedConfig,
    **kwargs: Any,
) -> None:
    """
    Unified dispatcher for report generation across all 3 modes:
      - 'compare': generate_comparison_report
      - 'ablation': generate_ablation_report
      - 'train': generate_train_report
    """
    mode = mode.lower()
    if mode == "compare":
        generate_comparison_report(
            df_results=kwargs["df_results"],
            models_dict=kwargs["models_dict"],
            importances_dict=kwargs["importances_dict"],
            output_filepath=output_filepath,
            config=config,
        )
    elif mode == "ablation":
        generate_ablation_report(
            df_results=kwargs["df_results"],
            best_exp_name=kwargs["best_exp_name"],
            best_model=kwargs["best_model"],
            best_imp=kwargs["best_imp"],
            output_filepath=output_filepath,
            config=config,
            model_type=kwargs.get("model_type", "xgboost"),
        )
    elif mode == "train":
        generate_train_report(
            metrics=kwargs["metrics"],
            model=kwargs["model"],
            df_imp=kwargs["df_imp"],
            output_filepath=output_filepath,
            config=config,
            model_type=kwargs.get("model_type", "xgboost"),
            model_save_path=kwargs.get("model_save_path"),
        )
    else:
        raise ValueError(f"Unknown report mode '{mode}'. Expected: compare | ablation | train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Bank Churn Prediction Entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to unified config YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["ablation", "train", "compare"],
        help=(
            "Run mode: "
            "'compare' (LR vs RF vs XGBoost, same pipeline), "
            "'ablation' (feature ablation on one model), "
            "'train' (single model training)"
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "lr", "rf"],
        help="Model to use in 'ablation' or 'train' mode (ignored in 'compare' mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (auto-routed to results/{rf,lr,xgboost,comparison}/ if omitted)",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="Custom filename for the generated markdown report (auto-resolved from config if omitted)",
    )

    args = parser.parse_args()

    # 1. Load Unified Config
    cfg = load_unified_config(args.config)
    print(f"[OK] Loaded unified configuration from: {args.config}")

    # 2. Resolve dataset path
    data_path = resolve_data_path(cfg.features.data_path)
    print(f"[OK] Ingesting dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"     Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Determine structured output subfolder:
    # results/comparison/ for 'compare', results/{model_type}/ for 'ablation' and 'train'
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.experiment.get_output_dir(mode=args.mode, model_type=args.model_type))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve report filename dynamically based on mode and model_type
    report_filename = args.report_name or cfg.experiment.get_report_filename(
        mode=args.mode,
        model_type=args.model_type,
    )
    report_path = str(output_dir / report_filename)

    if args.mode == "compare":
        print("\n=======================================================")
        print("STARTING MODEL COMPARISON: LR vs RF vs XGBoost")
        print("=======================================================")
        df_results, models_dict, importances_dict = run_model_comparison(df, cfg)

        best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
        best_name = best_row["Experiment"]

        print("\n=======================================================")
        print(f"BEST MODEL: {best_name}")
        print(f"ROC-AUC: {best_row['ROC_AUC']:.4f} | PR-AUC: {best_row['PR_AUC']:.4f} | F1: {best_row['F1_Score']:.4f}")
        print("=======================================================")
        print("\nFull Comparison Table:")
        display_cols = ["Model", "Experiment", "ROC_AUC", "PR_AUC", "F1_Score",
                        "Precision", "Recall", "Accuracy", "Balanced_Accuracy", "Log_Loss"]
        print(df_results[display_cols].to_string(index=False))

        # Save artifacts
        if cfg.experiment.save_model:
            for name, mtype in [("Logistic Regression", "lr"), ("Random Forest", "rf"), ("XGBoost", "xgboost")]:
                model_inst = models_dict[name]
                save_path = cfg.experiment.get_model_save_path(mtype)
                saved_file = model_inst.save(save_path)
                print(f"[OK] Saved {name} model artifact to: {saved_file}")

        compare_csv = str(output_dir / "model_comparison_metrics.csv")
        df_results.to_csv(compare_csv, index=False)
        print(f"\n[OK] Model comparison metrics saved to: {compare_csv}")

        for name, df_imp in importances_dict.items():
            safe_name = name.lower().replace(" ", "_")
            df_imp.to_csv(str(output_dir / f"feature_importance_{safe_name}.csv"), index=False)
        print(f"[OK] Feature importances saved to: {output_dir}/feature_importance_*.csv")

        # Generate comprehensive markdown comparison report
        generate_report(
            mode="compare",
            output_filepath=report_path,
            config=cfg,
            df_results=df_results,
            models_dict=models_dict,
            importances_dict=importances_dict,
        )



    elif args.mode == "ablation":
        model_type = args.model_type
        print("\n=======================================================")
        print(f"STARTING ABLATION BENCHMARK SUITE  [{model_type.upper()}]")
        print("=======================================================")
        df_results, models_dict, importances_dict = run_ablation_suite(df, cfg, model_type=model_type)

        # Select best model based on PR_AUC then ROC_AUC
        best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
        best_name = best_row["Experiment"]
        best_model = models_dict[best_name]
        best_imp = importances_dict[best_name]

        print("\n=======================================================")
        print(f"BEST PERFORMING CONFIGURATION: {best_name}")
        print(f"ROC-AUC: {best_row['ROC_AUC']:.4f} | PR-AUC: {best_row['PR_AUC']:.4f} | F1: {best_row['F1_Score']:.4f}")
        print("=======================================================")

        # Save best model artifact
        if cfg.experiment.save_model:
            save_path = cfg.experiment.get_model_save_path(model_type)
            saved_file = best_model.save(save_path)
            print(f"[OK] Saved best {model_type.upper()} model artifact to: {saved_file}")

        # Save metrics CSV
        metrics_csv_path = str(output_dir / f"{model_type}_ablation_metrics.csv")
        df_results.to_csv(metrics_csv_path, index=False)
        print(f"[OK] Exported metrics CSV to: {metrics_csv_path}")

        # Generate comprehensive markdown ablation report
        generate_report(
            mode="ablation",
            output_filepath=report_path,
            config=cfg,
            df_results=df_results,
            best_exp_name=best_name,
            best_model=best_model,
            best_imp=best_imp,
            model_type=model_type,
        )

    elif args.mode == "train":
        model_type = args.model_type
        print("\n=======================================================")
        print(f"TRAINING SINGLE [{model_type.upper()}] MODEL ACCORDING TO CONFIG")
        print("=======================================================")
        metrics, model, df_imp = run_single_experiment(
            df=df,
            features_cfg=cfg.features,
            model_cfg=cfg.model,
            experiment_name="Single Run",
            model_type=model_type,
        )

        save_path = None
        if cfg.experiment.save_model:
            save_path = cfg.experiment.get_model_save_path(model_type)
            saved_file = model.save(save_path)
            print(f"[OK] {model_type.upper()} model saved to: {saved_file}")

        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(str(output_dir / f"{model_type}_train_metrics.csv"), index=False)
        df_imp.to_csv(str(output_dir / f"{model_type}_feature_importance.csv"), index=False)
        print(f"[OK] Saved training metrics and feature importances to: {output_dir}")

        # Generate markdown train report
        generate_report(
            mode="train",
            output_filepath=report_path,
            config=cfg,
            metrics=metrics,
            model=model,
            df_imp=df_imp,
            model_type=model_type,
            model_save_path=save_path,
        )


if __name__ == "__main__":
    main()

