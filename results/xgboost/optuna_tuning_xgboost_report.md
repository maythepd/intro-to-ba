# BÁO CÁO TỐI ƯU HÓA SIÊU THAM SỐ (OPTUNA TUNING REPORT)
## MÔ HÌNH: XGBOOST CHURN PREDICTION

**Thời gian thực hiện:** `2026-11-06 15:36:28`  
**Mục tiêu tối ưu:** `Recall` (Cross-Validation Folds: `5`)  
**Tổng số trials:** `30` (Hoàn thành: `19`, Cắt tỉa sớm - Pruned: `11`)  
**CV Score cao nhất (Recall):** `0.8712`  

---

## 1. TIẾN TRÌNH TỐI ƯU HÓA (OPTIMIZATION PROGRESS)

![Tiến trình tối ưu hóa](xgboost_tuning_history.png)

---

## 2. BỘ SIÊU THAM SỐ TỐI ƯU (BEST HYPERPARAMETERS)

| Hyperparameter   |   Optimal Value |
|:-----------------|----------------:|
| scale_pos_weight |        4.5      |
| n_estimators     |      150        |
| max_depth        |        3        |
| learning_rate    |        0.023368 |
| subsample        |        0.7      |
| colsample_bytree |        0.8      |
| gamma            |        1.4      |
| min_child_weight |        6        |
| reg_alpha        |        0.651525 |
| reg_lambda       |        0.147708 |

---

## 3. HIỆU NĂNG ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (GENERALIZATION TEST SET)

Được đánh giá trên tập kiểm thử độc lập (20% dữ liệu) hoàn toàn không rò rỉ trong quá trình tuning:

| Metric            |   Score |
|:------------------|--------:|
| ROC_AUC           |  0.8534 |
| PR_AUC            |  0.6276 |
| F1_Score          |  0.5178 |
| Precision         |  0.3647 |
| Recall            |  0.8922 |
| Accuracy          |  0.661  |
| Balanced_Accuracy |  0.747  |
| Log_Loss          |  0.6351 |

### Ma trận nhầm lẫn (Confusion Matrix):
- **True Positives (Bắt đúng churn):** `182`
- **False Negatives (Bỏ sót churn):** `22`
- **True Negatives (Khách hàng ở lại):** `479`
- **False Positives (Báo nhầm churn):** `317`

---

## 4. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (TUNED MODEL)

![Đặc trưng quan trọng](xgboost_feature_importance_tuned.png)

| Feature                   |   Importance |   Relative_Pct |
|:--------------------------|-------------:|---------------:|
| is_middle_aged_vulnerable |      710.094 |           8.34 |
| gender_Male               |      557.433 |           6.54 |
| is_product_danger_zone    |      435.315 |           5.11 |
| gender_Female             |      404.827 |           4.75 |
| age                       |      329.621 |           3.87 |
| gender_target_enc         |      322.618 |           3.79 |
| has_resolved_complaint    |      291.914 |           3.43 |
| is_zero_balance           |      271.615 |           3.19 |
| std_age                   |      263.328 |           3.09 |
| products_number           |      254.19  |           2.98 |
| country_France            |      252.5   |           2.96 |
| country_Germany           |      237.323 |           2.79 |
| is_high_friction          |      218.266 |           2.56 |
| is_spending_collapsing    |      216.092 |           2.54 |
| germany_high_balance      |      214.451 |           2.52 |

---

## 5. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 5.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift 3.38x** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift 2.87x**.

| Decile   |   Customers |   Churners | Churn_Rate   | Decile_Lift   | Cum_Recall   | Cum_Lift   |
|:---------|------------:|-----------:|:-------------|:--------------|:-------------|:-----------|
| D1       |         100 |         69 | 69.00%       | 3.38x         | 33.82%       | 3.38x      |
| D2       |         100 |         48 | 48.00%       | 2.35x         | 57.35%       | 2.87x      |
| D3       |         100 |         30 | 30.00%       | 1.47x         | 72.06%       | 2.40x      |
| D4       |         100 |         23 | 23.00%       | 1.13x         | 83.33%       | 2.08x      |
| D5       |         100 |         12 | 12.00%       | 0.59x         | 89.22%       | 1.78x      |
| D6       |         100 |         11 | 11.00%       | 0.54x         | 94.61%       | 1.58x      |
| D7       |         100 |          5 | 5.00%        | 0.25x         | 97.06%       | 1.39x      |
| D8       |         100 |          4 | 4.00%        | 0.20x         | 99.02%       | 1.24x      |
| D9       |         100 |          1 | 1.00%        | 0.05x         | 99.51%       | 1.11x      |
| D10      |         100 |          1 | 1.00%        | 0.05x         | 100.00%      | 1.00x      |

### 5.2. So sánh với các Baselines & Điểm vận hành tối ưu:
| Configuration                              | Threshold   | Accuracy   | Precision   | Recall   |   F1_Score |   ROC_AUC |   PR_AUC | Precision_Lift   | Recall_Lift   | PR_AUC_Lift   |
|:-------------------------------------------|:------------|:-----------|:------------|:---------|-----------:|----------:|---------:|:-----------------|:--------------|:--------------|
| 1. Random Guess Baseline                   | -           | 67.52%     | 20.40%      | 20.40%   |     0.204  |    0.5    |   0.204  | 1.00x            | 1.00x         | 1.00x         |
| 2. Majority Class Baseline (All Non-Churn) | 1.00        | 79.60%     | 0.00%       | 0.00%    |     0      |    0.5    |   0.204  | 0.00x            | 0.00x         | 1.00x         |
| Model @ Default (0.50)                     | 0.50        | 66.10%     | 36.47%      | 89.22%   |     0.5178 |    0.8534 |   0.6276 | 1.79x            | 4.37x         | 3.08x         |
| Model @ Optimal F1                         | 0.60        | 76.10%     | 45.07%      | 78.43%   |     0.5725 |    0.8534 |   0.6276 | 2.21x            | 3.84x         | 3.08x         |
| Model @ Optimal F2 (High Recall)           | 0.56        | 72.30%     | 41.16%      | 83.33%   |     0.5511 |    0.8534 |   0.6276 | 2.02x            | 4.08x         | 3.08x         |
| Model @ Target Recall (>=75%)              | 0.60        | 76.10%     | 45.07%      | 78.43%   |     0.5725 |    0.8534 |   0.6276 | 2.21x            | 3.84x         | 3.08x         |


---
*Báo cáo được sinh tự động bởi Optuna Tuner Engine - Digital Bank Churn Prediction.*
