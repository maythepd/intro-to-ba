# BÁO CÁO HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION (TRAINING REPORT)
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian huấn luyện:** `2026-11-19 23:31:19`  
**Mô hình:** `Logistic Regression`  
**Artifact đã lưu:** `models/saved/lr_churn.pkl`  
**Cấu hình:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST

### 1.1. Chỉ số hiệu năng phân loại (Performance Metrics)

| Chỉ số (Metric)   |   Giá trị | Ý nghĩa Nghiệp vụ                                                    |
|:------------------|----------:|:---------------------------------------------------------------------|
| ROC_AUC           |    0.8384 | Khả năng phân biệt tổng thể giữa khách hàng rời bỏ và ở lại          |
| PR_AUC            |    0.6402 | Độ chính xác trung bình trên dữ liệu mất cân bằng (Baseline: ~20.4%) |
| F1_Score          |    0.5193 | Trung bình điều hòa giữa Precision và Recall                         |
| Precision         |    0.3792 | Tỷ lệ khách hàng cảnh báo rời bỏ thực sự rời bỏ                      |
| Recall            |    0.8235 | Tỷ lệ nhận diện thành công khách hàng rời bỏ thực tế                 |
| Accuracy          |    0.689  | Tỷ lệ dự báo đúng trên toàn bộ tập kiểm thử                          |
| Balanced_Accuracy |    0.739  | Trung bình cộng độ nhạy của 2 lớp (giảm thiên lệch đa số)            |
| Log_Loss          |    0.4912 | Độ mất mát entropy chéo và chất lượng xác suất hiệu chuẩn            |

### 1.2. Thông tin mẫu và cấu hình huấn luyện

| Tham số                                         | Giá trị   |
|:------------------------------------------------|:----------|
| Mô hình thuật toán                              | LR        |
| Ngưỡng quyết định phân lớp (Decision Threshold) | 0.35      |
| Số lượng đặc trưng đầu vào                      | 374       |
| Áp dụng kỹ thuật cân bằng mẫu SMOTE             | False     |
| Kích thước tập huấn luyện ban đầu               | 7999x374  |
| Kích thước tập validation                       | 1001x374  |
| Kích thước tập kiểm thử độc lập                 | 1000x374  |

---

## 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

- **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `168` (82.35% tổng khách hàng rời bỏ)
- **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `36` (17.65% khách hàng rời bỏ bị bỏ sót)
- **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `521` (65.45% khách hàng ở lại nhận diện đúng)
- **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `275` (34.55% khách hàng ở lại bị cảnh báo nhầm)

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

![Biểu đồ đặc trưng quan trọng](lr_feature_importance.png)

| Feature                                    |   Importance |   Coefficient | Relative_Pct   |
|:-------------------------------------------|-------------:|--------------:|:---------------|
| lda_log_odds                               |       0.8919 |     0.89186   | 19.12%         |
| lda_projection_churn                       |       0.8919 |     0.89186   | 19.12%         |
| credit_to_age_ratio                        |       0.258  |    -0.258042  | 5.53%          |
| quant_inter_germany_x_danger_single        |       0.1951 |    -0.195085  | 4.18%          |
| products_number                            |       0.1562 |    -0.156182  | 3.35%          |
| credit_card                                |       0.1095 |    -0.10954   | 2.35%          |
| quant_multiscale_ratio_tx_amount_7d_to_30d |       0.0994 |     0.0993504 | 2.13%          |
| quant_cyclic_tx_day_cos                    |       0.0973 |    -0.0972916 | 2.09%          |
| quant_multiscale_outflow_share_7d_of_30d   |       0.0799 |    -0.0798516 | 1.71%          |
| std_age                                    |       0.0796 |    -0.079641  | 1.71%          |
| age                                        |       0.0796 |    -0.079641  | 1.71%          |
| quant_cyclic_inflow_day_dispersion         |       0.0716 |     0.0716027 | 1.54%          |
| quant_inter_germany_x_age                  |       0.066  |     0.0659699 | 1.41%          |
| quant_inter_age_x_danger_products          |       0.0626 |     0.0625974 | 1.34%          |
| num_closed_accounts                        |       0.0626 |     0.0625577 | 1.34%          |


---

## 4. ĐÁNH GIÁ HỆ SỐ LIFT & 10 PHÂN VỊ RỦI RO (DECILE LIFT & GAINS)

![Lift and Cumulative Gains Curve](lift_and_gains_curve.png)

### 4.1. Bảng 10 phân vị rủi ro (Decile Lift Analysis):
* **Top 10% rủi ro cao nhất (Decile D1):** Đạt hệ số **Lift 3.68x** so với chọn ngẫu nhiên.
* **Top 20% rủi ro cao nhất (Decile D1 + D2):** Đạt hệ số **Cumulative Lift 2.87x**.

| Decile   |   Customers |   Churners | Churn_Rate   | Decile_Lift   | Cum_Recall   | Cum_Lift   |
|:---------|------------:|-----------:|:-------------|:--------------|:-------------|:-----------|
| D1       |         100 |         75 | 75.00%       | 3.68x         | 36.76%       | 3.68x      |
| D2       |         100 |         42 | 42.00%       | 2.06x         | 57.35%       | 2.87x      |
| D3       |         100 |         29 | 29.00%       | 1.42x         | 71.57%       | 2.39x      |
| D4       |         100 |         18 | 18.00%       | 0.88x         | 80.39%       | 2.01x      |
| D5       |         100 |         13 | 13.00%       | 0.64x         | 86.76%       | 1.74x      |
| D6       |         100 |         10 | 10.00%       | 0.49x         | 91.67%       | 1.53x      |
| D7       |         100 |          9 | 9.00%        | 0.44x         | 96.08%       | 1.37x      |
| D8       |         100 |          4 | 4.00%        | 0.20x         | 98.04%       | 1.23x      |
| D9       |         100 |          3 | 3.00%        | 0.15x         | 99.51%       | 1.11x      |
| D10      |         100 |          1 | 1.00%        | 0.05x         | 100.00%      | 1.00x      |

### 4.2. So sánh với các Baselines & Điểm vận hành tối ưu:

| Configuration                              | Threshold   | Accuracy   | Precision   | Recall   |   F1_Score |   ROC_AUC |   PR_AUC | Precision_Lift   | Recall_Lift   | PR_AUC_Lift   |
|:-------------------------------------------|:------------|:-----------|:------------|:---------|-----------:|----------:|---------:|:-----------------|:--------------|:--------------|
| 1. Random Guess Baseline                   | -           | 67.52%     | 20.40%      | 20.40%   |     0.204  |    0.5    |   0.204  | 1.00x            | 1.00x         | 1.00x         |
| 2. Majority Class Baseline (All Non-Churn) | 1.00        | 79.60%     | 0.00%       | 0.00%    |     0      |    0.5    |   0.204  | 0.00x            | 0.00x         | 1.00x         |
| Model @ Default (0.50)                     | 0.50        | 79.10%     | 49.15%      | 70.59%   |     0.5795 |    0.8384 |   0.6402 | 2.41x            | 3.46x         | 3.14x         |
| Model @ Optimal F1                         | 0.62        | 82.20%     | 55.75%      | 61.76%   |     0.586  |    0.8384 |   0.6402 | 2.73x            | 3.03x         | 3.14x         |
| Model @ Optimal F2 (High Recall)           | 0.30        | 65.20%     | 35.43%      | 85.78%   |     0.5014 |    0.8384 |   0.6402 | 1.74x            | 4.21x         | 3.14x         |
| Model @ Target Recall (>=75%)              | 0.36        | 69.40%     | 38.19%      | 80.88%   |     0.5189 |    0.8384 |   0.6402 | 1.87x            | 3.96x         | 3.14x         |
| Model @ Manual (0.35)                      | 0.35        | 68.90%     | 37.92%      | 82.35%   |     0.5193 |    0.8384 |   0.6402 | 1.86x            | 4.04x         | 3.14x         |


---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
