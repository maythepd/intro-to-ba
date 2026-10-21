# BÁO CÁO HUẤN LUYỆN MÔ HÌNH XGBOOST (TRAINING REPORT)
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian huấn luyện:** `2026-10-21 20:22:07`  
**Mô hình:** `XGBoost`  
**Artifact đã lưu:** `models/saved/xgboost_churn.json`  
**Cấu hình:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST

| Metric            | Value      |
|:------------------|:-----------|
| ROC_AUC           | 0.9999     |
| PR_AUC            | 0.9996     |
| F1_Score          | 0.9876     |
| Precision         | 0.9950     |
| Recall            | 0.9803     |
| Accuracy          | 0.9950     |
| Balanced_Accuracy | 0.9895     |
| Log_Loss          | 0.0132     |
| TP                | 399        |
| FP                | 2          |
| TN                | 1591       |
| FN                | 8          |
| Experiment        | Single Run |
| Model             | XGBOOST    |
| Num_Features      | 55         |
| SMOTE_Applied     | True       |

---

## 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

- **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `399`
- **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `8`
- **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1591`
- **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `2`

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

| Feature                     |   Importance |   Relative_Pct |
|:----------------------------|-------------:|---------------:|
| robust_trans_trend_ratio    |    263.359   |          40.94 |
| trans_trend_ratio           |    200.102   |          31.11 |
| is_spending_collapsing      |     48.8732  |           7.6  |
| robust_recency_days         |     24.5888  |           3.82 |
| interaction_velocity        |     16.3569  |           2.54 |
| recency_days                |     15.9587  |           2.48 |
| log_recency_days            |     12.9788  |           2.02 |
| monthly_transactions        |      8.13486 |           1.26 |
| robust_monthly_transactions |      8.08547 |           1.26 |
| wealth_deceleration_risk    |      4.17976 |           0.65 |
| robust_app_logins_monthly   |      3.04966 |           0.47 |
| app_logins_monthly          |      2.86277 |           0.45 |
| friction_index              |      2.57606 |           0.4  |
| is_product_sweetspot        |      2.23    |           0.35 |
| avg_ticket_size             |      1.73936 |           0.27 |

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
