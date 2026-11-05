# BÁO CÁO TỐI ƯU HÓA SIÊU THAM SỐ (OPTUNA TUNING REPORT)
## MÔ HÌNH: XGBOOST CHURN PREDICTION

**Thời gian thực hiện:** `2026-11-05 22:09:34`  
**Mục tiêu tối ưu:** `PR_AUC` (Cross-Validation Folds: `5`)  
**Tổng số trials:** `2` (Hoàn thành: `2`, Cắt tỉa sớm - Pruned: `0`)  
**CV Score cao nhất (PR_AUC):** `0.7553`  

---

## 1. TIẾN TRÌNH TỐI ƯU HÓA (OPTIMIZATION PROGRESS)

![Tiến trình tối ưu hóa](xgboost_tuning_history.png)

---

## 2. BỘ SIÊU THAM SỐ TỐI ƯU (BEST HYPERPARAMETERS)

| Hyperparameter   |   Optimal Value |
|:-----------------|----------------:|
| n_estimators     |      450        |
| max_depth        |        7        |
| learning_rate    |        0.109203 |
| subsample        |        0.7      |
| colsample_bytree |        1        |
| gamma            |        1        |
| min_child_weight |        7        |
| reg_alpha        |        1.2745   |
| reg_lambda       |        3.98879  |

---

## 3. HIỆU NĂNG ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (GENERALIZATION TEST SET)

Được đánh giá trên tập kiểm thử độc lập (20% dữ liệu) hoàn toàn không rò rỉ trong quá trình tuning:

| Metric            |   Score |
|:------------------|--------:|
| ROC_AUC           |  0.84   |
| PR_AUC            |  0.6435 |
| F1_Score          |  0.5543 |
| Precision         |  0.622  |
| Recall            |  0.5    |
| Accuracy          |  0.836  |
| Balanced_Accuracy |  0.7111 |
| Log_Loss          |  0.4279 |

### Ma trận nhầm lẫn (Confusion Matrix):
- **True Positives (Bắt đúng churn):** `102`
- **False Negatives (Bỏ sót churn):** `102`
- **True Negatives (Khách hàng ở lại):** `734`
- **False Positives (Báo nhầm churn):** `62`

---

## 4. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (TUNED MODEL)

![Đặc trưng quan trọng](xgboost_feature_importance_tuned.png)

| Feature                      |   Importance |   Relative_Pct |
|:-----------------------------|-------------:|---------------:|
| nl_supervised_z5             |     203.726  |          18.86 |
| nl_supervised_z10            |      61.5289 |           5.7  |
| is_login_collapsing          |      34.7644 |           3.22 |
| is_middle_aged_vulnerable    |      32.0012 |           2.96 |
| primary_device_target_enc    |      29.0799 |           2.69 |
| has_resolved_complaint       |      25.94   |           2.4  |
| gender_target_enc            |      23.8356 |           2.21 |
| multi_account_holder         |      21.703  |           2.01 |
| nl_supervised_z7             |      20.1762 |           1.87 |
| is_product_danger_zone       |      18.8448 |           1.74 |
| has_fee_penalty              |      18.0479 |           1.67 |
| preferred_channel_Mobile App |      17.5664 |           1.63 |
| max_savings_term_months      |      16.6287 |           1.54 |
| gender_Male                  |      15.2261 |           1.41 |
| country_Germany              |      15.0616 |           1.39 |

---

## 5. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 5.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift 3.73x** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift 2.99x**.

| Decile   |   Customers |   Churners | Churn_Rate   | Decile_Lift   | Cum_Recall   | Cum_Lift   |
|:---------|------------:|-----------:|:-------------|:--------------|:-------------|:-----------|
| D1       |         100 |         76 | 76.00%       | 3.73x         | 37.25%       | 3.73x      |
| D2       |         100 |         46 | 46.00%       | 2.25x         | 59.80%       | 2.99x      |
| D3       |         100 |         21 | 21.00%       | 1.03x         | 70.10%       | 2.34x      |
| D4       |         100 |         17 | 17.00%       | 0.83x         | 78.43%       | 1.96x      |
| D5       |         100 |         16 | 16.00%       | 0.78x         | 86.27%       | 1.73x      |
| D6       |         100 |         12 | 12.00%       | 0.59x         | 92.16%       | 1.54x      |
| D7       |         100 |          9 | 9.00%        | 0.44x         | 96.57%       | 1.38x      |
| D8       |         100 |          4 | 4.00%        | 0.20x         | 98.53%       | 1.23x      |
| D9       |         100 |          2 | 2.00%        | 0.10x         | 99.51%       | 1.11x      |
| D10      |         100 |          1 | 1.00%        | 0.05x         | 100.00%      | 1.00x      |

### 5.2. So sánh với các Baselines & Điểm vận hành tối ưu:
| Configuration                              | Threshold   | Accuracy   | Precision   | Recall   |   F1_Score |   ROC_AUC |   PR_AUC | Precision_Lift   | Recall_Lift   | PR_AUC_Lift   |
|:-------------------------------------------|:------------|:-----------|:------------|:---------|-----------:|----------:|---------:|:-----------------|:--------------|:--------------|
| 1. Random Guess Baseline                   | -           | 67.52%     | 20.40%      | 20.40%   |     0.204  |      0.5  |   0.204  | 1.00x            | 1.00x         | 1.00x         |
| 2. Majority Class Baseline (All Non-Churn) | 1.00        | 79.60%     | 0.00%       | 0.00%    |     0      |      0.5  |   0.204  | 0.00x            | 0.00x         | 1.00x         |
| Model @ Default (0.50)                     | 0.50        | 83.60%     | 62.20%      | 50.00%   |     0.5543 |      0.84 |   0.6435 | 3.05x            | 2.45x         | 3.15x         |
| Model @ Optimal F1                         | 0.42        | 84.20%     | 62.78%      | 55.39%   |     0.5885 |      0.84 |   0.6435 | 3.08x            | 2.72x         | 3.15x         |
| Model @ Optimal F2 (High Recall)           | 0.42        | 84.20%     | 62.78%      | 55.39%   |     0.5885 |      0.84 |   0.6435 | 3.08x            | 2.72x         | 3.15x         |
| Model @ Target Recall (>=75%)              | 0.73        | 84.60%     | 71.93%      | 40.20%   |     0.5157 |      0.84 |   0.6435 | 3.53x            | 1.97x         | 3.15x         |


---
*Báo cáo được sinh tự động bởi Optuna Tuner Engine - Digital Bank Churn Prediction.*
