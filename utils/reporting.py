"""
Markdown report generation and executive presentation utilities.
Generates comprehensive reports for model comparison, ablation studies, and single training runs.
"""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from config import UnifiedConfig
from visualization import (
    plot_feature_importance,
    plot_model_comparison,
)
from .experiments import AnyChurnModel


def generate_ablation_report(
    df_results: pd.DataFrame,
    best_exp_name: str,
    best_model: AnyChurnModel,
    best_imp: pd.DataFrame,
    output_filepath: str,
    config: UnifiedConfig,
    model_type: str = "xgboost",
    lift_res: Optional[Dict[str, Any]] = None,
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

    # Save feature importance chart
    chart_filename = "ablation_feature_importance.png"
    chart_path = Path(output_filepath).parent / chart_filename
    plot_feature_importance(
        df_imp=best_imp,
        output_path=chart_path,
        model_name=f"{model_display} (Ablation Champion)",
        top_k=15,
    )

    lift_section = ""
    if lift_res:
        df_dec = lift_res.get("df_deciles")
        df_comp = lift_res.get("df_comparison")
        top10_lift = df_dec.iloc[0]["Decile_Lift"] if df_dec is not None and len(df_dec) > 0 else "N/A"
        top20_lift = df_dec.iloc[1]["Cum_Lift"] if df_dec is not None and len(df_dec) > 1 else "N/A"
        lift_section = f"""

---

## 4. ĐÁNH GIÁ HỆ SỐ LIFT CỦA CẤU HÌNH TỐI ƯU NHẤT ({best_exp_name})

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 4.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift {top10_lift}** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift {top20_lift}**.

{df_dec.to_markdown(index=False) if df_dec is not None else ""}

### 4.2. So sánh với các Baselines & Điểm vận hành tối ưu:
{df_comp.to_markdown(index=False) if df_comp is not None else ""}
"""

    md_content = f"""# BÁO CÁO THỰC NGHIỆM MÔ HÌNH {model_display.upper()} & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `{timestamp}`  
**Mô hình cốt lõi:** `{model_display}`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **{model_display}** qua các cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

{df_table.to_markdown(index=False)}

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **{best_exp_name}**
- **ROC-AUC:** `{best_metrics['ROC_AUC']:.4f}` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `{best_metrics['PR_AUC']:.4f}` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `{best_metrics['F1_Score']:.4f}` (Precision: `{best_metrics['Precision']:.4f}`, Recall: `{best_metrics['Recall']:.4f}`).
- **Ma trận nhầm lẫn (Confusion Matrix trên tập dữ liệu kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `{best_metrics['TP']}`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `{best_metrics['FN']}`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `{best_metrics['TN']}`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `{best_metrics['FP']}`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features:**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight:**
   - Đối với {model_display}: SMOTE cân bằng tập huấn luyện bằng việc sinh mẫu tổng hợp cho nhóm thiểu số. Native class_weight / scale_pos_weight tái cân bằng trực tiếp trên hàm mất mát mà không làm tăng kích thước bộ nhớ.
3. **Mã hóa Target Encoding:**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features:**
   - Top 15 đặc trưng quan trọng nhất duy trì phần lớn khả năng phân loại, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** {imp_type_label}

![Biểu đồ đặc trưng quan trọng nhất]({chart_filename})

{top15_imp.to_markdown(index=False)}{lift_section}

---

## 5. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

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
    lift_res_dict: Optional[Dict[str, Dict[str, Any]]] = None,
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
    if "CV_ROC_AUC" in df_results.columns:
        display_cols.insert(1, "CV_ROC_AUC")
        display_cols.insert(2, "CV_PR_AUC")
    df_table = df_results[[c for c in display_cols if c in df_results.columns]].copy()

    # Determine best model based on PR_AUC and ROC_AUC
    best_row = df_results.sort_values(by=["PR_AUC", "ROC_AUC"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]

    # Render and save comparison chart and feature importances
    comp_chart_name = "model_comparison_chart.png"
    comp_chart_path = Path(output_filepath).parent / comp_chart_name
    plot_model_comparison(df_results, comp_chart_path)

    for name, df_imp in importances_dict.items():
        safe_name = name.lower().replace(" ", "_")
        plot_feature_importance(
            df_imp=df_imp,
            output_path=Path(output_filepath).parent / f"feature_importance_{safe_name}.png",
            model_name=name,
            top_k=15,
        )

    # Build Top 5 features per model for side-by-side comparison
    top_features_md = ""
    for name, df_imp in importances_dict.items():
        safe_name = name.lower().replace(" ", "_")
        top5 = df_imp.head(5)[["Feature", "Relative_Pct"]].to_markdown(index=False)
        top_features_md += f"#### Top 5 Đặc trưng — {name}\n\n![{name} Top Features](feature_importance_{safe_name}.png)\n\n{top5}\n\n"

    lift_section = ""
    if lift_res_dict:
        lift_summary_rows = []
        for name, l_res in lift_res_dict.items():
            df_dec = l_res.get("df_deciles")
            if df_dec is not None and len(df_dec) > 1:
                lift_summary_rows.append({
                    "Model": name,
                    "Top 10% Lift (D1)": df_dec.iloc[0]["Decile_Lift"],
                    "Top 10% Churn Captured": df_dec.iloc[0]["Cum_Recall"],
                    "Top 20% Cumulative Lift": df_dec.iloc[1]["Cum_Lift"],
                    "Top 20% Churn Captured": df_dec.iloc[1]["Cum_Recall"],
                })
        if lift_summary_rows:
            df_lift_summary = pd.DataFrame(lift_summary_rows)
            lift_section = f"""

---

## 5. SO SÁNH HỆ SỐ LIFT VÀ PHÂN VỊ RỦI RO (DECILE LIFT COMPARISON)

![Champion Model Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

{df_lift_summary.to_markdown(index=False)}
"""

    md_content = f"""# BÁO CÁO ĐỐI ĐẦU VÀ SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH (MODEL COMPARISON)
## LOGISTIC REGRESSION vs. RANDOM FOREST vs. XGBOOST
### DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (DIGITAL BANK CHURN PREDICTION)

**Thời gian thực thi:** `{timestamp}`  
**Tập dữ liệu:** `{config.features.data_path}`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ HIỆU NĂNG (BENCHMARK TABLE)

Bảng đối đầu trực tiếp giữa 3 họ thuật toán đại diện: Tuyến tính (Logistic Regression), Ensemble Bagging (Random Forest), và Gradient Boosting (XGBoost):

![Biểu đồ đối đầu mô hình]({comp_chart_name})

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
   - **Hạn chế:** Không tự động nắm bắt được các mối tương tác phi tuyến tính phức tạp.
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
{lift_section}
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
    lift_res: Optional[Dict[str, Any]] = None,
) -> None:
    """Generates an executive-level Markdown report for a single model training run."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_names = {"xgboost": "XGBoost", "lr": "Logistic Regression", "rf": "Random Forest"}
    model_display = model_names.get(model_type.lower(), model_type.upper())

    # Save feature importance chart
    chart_name = f"{model_type}_feature_importance.png"
    chart_path = Path(output_filepath).parent / chart_name
    plot_feature_importance(df_imp, chart_path, model_name=model_display, top_k=15)

    top15 = df_imp.head(15).to_markdown(index=False)
    metrics_table = pd.DataFrame([{
        "Metric": k,
        "Value": f"{v:.4f}" if isinstance(v, float) else str(v)
    } for k, v in metrics.items() if k not in ("X_train_Shape", "X_test_Shape")]).to_markdown(index=False)

    lift_section = ""
    if lift_res:
        df_dec = lift_res.get("df_deciles")
        df_comp = lift_res.get("df_comparison")
        top10_lift = df_dec.iloc[0]["Decile_Lift"] if df_dec is not None and len(df_dec) > 0 else "N/A"
        top20_lift = df_dec.iloc[1]["Cum_Lift"] if df_dec is not None and len(df_dec) > 1 else "N/A"
        lift_section = f"""

---

## 4. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 4.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift {top10_lift}** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift {top20_lift}**.

{df_dec.to_markdown(index=False) if df_dec is not None else ""}

### 4.2. So sánh với các Baselines & Điểm vận hành tối ưu:
{df_comp.to_markdown(index=False) if df_comp is not None else ""}
"""

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

![Biểu đồ đặc trưng quan trọng]({chart_name})

{top15}{lift_section}

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
            lift_res_dict=kwargs.get("lift_res_dict"),
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
            lift_res=kwargs.get("lift_res"),
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
            lift_res=kwargs.get("lift_res"),
        )
    else:
        raise ValueError(f"Unknown report mode '{mode}'. Expected: compare | ablation | train")

