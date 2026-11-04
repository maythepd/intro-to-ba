# BÁO CÁO TỐI ƯU HÓA SIÊU THAM SỐ (OPTUNA TUNING REPORT)
## MÔ HÌNH: XGBOOST CHURN PREDICTION

**Thời gian thực hiện:** `2026-11-04 22:50:31`  
**Mục tiêu tối ưu:** `F2_Score` (Cross-Validation Folds: `5`)  
**Tổng số trials:** `30` (Hoàn thành: `17`, Cắt tỉa sớm - Pruned: `13`)  
**CV Score cao nhất (F2_Score):** `0.6920`  

---

## 1. TIẾN TRÌNH TỐI ƯU HÓA (OPTIMIZATION PROGRESS)

![Tiến trình tối ưu hóa](xgboost_tuning_history.png)

---

## 2. BỘ SIÊU THAM SỐ TỐI ƯU (BEST HYPERPARAMETERS)

| Hyperparameter   |   Optimal Value |
|:-----------------|----------------:|
| scale_pos_weight |       4         |
| n_estimators     |     150         |
| max_depth        |       5         |
| learning_rate    |       0.0608571 |
| subsample        |       0.6       |
| colsample_bytree |       0.7       |
| gamma            |       0.4       |
| min_child_weight |       6         |
| reg_alpha        |       1.57665   |
| reg_lambda       |       0.557272  |

---

## 3. HIỆU NĂNG ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (GENERALIZATION TEST SET)

Được đánh giá trên tập kiểm thử độc lập (20% dữ liệu) hoàn toàn không rò rỉ trong quá trình tuning:

| Metric            |   Score |
|:------------------|--------:|
| ROC_AUC           |  0.8667 |
| PR_AUC            |  0.6654 |
| F1_Score          |  0.5868 |
| Precision         |  0.462  |
| Recall            |  0.8039 |
| Accuracy          |  0.769  |
| Balanced_Accuracy |  0.782  |
| Log_Loss          |  0.5114 |

### Ma trận nhầm lẫn (Confusion Matrix):
- **True Positives (Bắt đúng churn):** `164`
- **False Negatives (Bỏ sót churn):** `40`
- **True Negatives (Khách hàng ở lại):** `605`
- **False Positives (Báo nhầm churn):** `191`

---

## 4. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (TUNED MODEL)

![Đặc trưng quan trọng](xgboost_feature_importance_tuned.png)

| Feature                   |   Importance |   Relative_Pct |
|:--------------------------|-------------:|---------------:|
| is_middle_aged_vulnerable |     422.257  |           9.19 |
| is_product_danger_zone    |     240.056  |           5.23 |
| gender_Male               |     200.567  |           4.37 |
| active_member             |     166.312  |           3.62 |
| std_age                   |     127.407  |           2.77 |
| country_Germany           |     119.576  |           2.6  |
| gender_target_enc         |     117.264  |           2.55 |
| primary_device_target_enc |     116.898  |           2.54 |
| products_number           |     104.474  |           2.27 |
| age                       |     101.26   |           2.2  |
| gender_Female             |      94.799  |           2.06 |
| has_resolved_complaint    |      93.5529 |           2.04 |
| is_login_collapsing       |      84.437  |           1.84 |
| country_target_enc        |      76.7655 |           1.67 |
| multi_account_holder      |      76.37   |           1.66 |

---

## 5. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 5.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift 3.68x** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift 3.06x**.

| Decile   |   Customers |   Churners | Churn_Rate   | Decile_Lift   | Cum_Recall   | Cum_Lift   |
|:---------|------------:|-----------:|:-------------|:--------------|:-------------|:-----------|
| D1       |         100 |         75 | 75.00%       | 3.68x         | 36.76%       | 3.68x      |
| D2       |         100 |         50 | 50.00%       | 2.45x         | 61.27%       | 3.06x      |
| D3       |         100 |         25 | 25.00%       | 1.23x         | 73.53%       | 2.45x      |
| D4       |         100 |         23 | 23.00%       | 1.13x         | 84.80%       | 2.12x      |
| D5       |         100 |         12 | 12.00%       | 0.59x         | 90.69%       | 1.81x      |
| D6       |         100 |          9 | 9.00%        | 0.44x         | 95.10%       | 1.58x      |
| D7       |         100 |          7 | 7.00%        | 0.34x         | 98.53%       | 1.41x      |
| D8       |         100 |          1 | 1.00%        | 0.05x         | 99.02%       | 1.24x      |
| D9       |         100 |          1 | 1.00%        | 0.05x         | 99.51%       | 1.11x      |
| D10      |         100 |          1 | 1.00%        | 0.05x         | 100.00%      | 1.00x      |

### 5.2. So sánh với các Baselines & Điểm vận hành tối ưu:
| Configuration                              | Threshold   | Accuracy   | Precision   | Recall   |   F1_Score |   ROC_AUC |   PR_AUC | Precision_Lift   | Recall_Lift   | PR_AUC_Lift   |
|:-------------------------------------------|:------------|:-----------|:------------|:---------|-----------:|----------:|---------:|:-----------------|:--------------|:--------------|
| 1. Random Guess Baseline                   | -           | 67.52%     | 20.40%      | 20.40%   |     0.204  |    0.5    |   0.204  | 1.00x            | 1.00x         | 1.00x         |
| 2. Majority Class Baseline (All Non-Churn) | 1.00        | 79.60%     | 0.00%       | 0.00%    |     0      |    0.5    |   0.204  | 0.00x            | 0.00x         | 1.00x         |
| Model @ Default (0.50)                     | 0.50        | 76.90%     | 46.20%      | 80.39%   |     0.5868 |    0.8667 |   0.6654 | 2.26x            | 3.94x         | 3.26x         |
| Model @ Optimal F1                         | 0.72        | 85.00%     | 64.21%      | 59.80%   |     0.6193 |    0.8667 |   0.6654 | 3.15x            | 2.93x         | 3.26x         |
| Model @ Optimal F2 (High Recall)           | 0.45        | 73.50%     | 42.58%      | 85.78%   |     0.5691 |    0.8667 |   0.6654 | 2.09x            | 4.21x         | 3.26x         |
| Model @ Target Recall (>=75%)              | 0.55        | 78.50%     | 48.26%      | 75.00%   |     0.5873 |    0.8667 |   0.6654 | 2.37x            | 3.68x         | 3.26x         |


---
*Báo cáo được sinh tự động bởi Optuna Tuner Engine - Digital Bank Churn Prediction.*
