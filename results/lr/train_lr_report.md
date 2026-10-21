# BÁO CÁO HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION (TRAINING REPORT)
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian huấn luyện:** `2026-10-21 20:21:49`  
**Mô hình:** `Logistic Regression`  
**Artifact đã lưu:** `models/saved/lr_churn.pkl`  
**Cấu hình:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST

| Metric            | Value      |
|:------------------|:-----------|
| ROC_AUC           | 0.9020     |
| PR_AUC            | 0.7246     |
| F1_Score          | 0.6390     |
| Precision         | 0.5050     |
| Recall            | 0.8698     |
| Accuracy          | 0.8000     |
| Balanced_Accuracy | 0.8260     |
| Log_Loss          | 0.4202     |
| TP                | 354        |
| FP                | 347        |
| TN                | 1246       |
| FN                | 53         |
| Experiment        | Single Run |
| Model             | LR         |
| Num_Features      | 55         |
| SMOTE_Applied     | True       |

---

## 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

- **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `354`
- **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `53`
- **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1246`
- **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `347`

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

| Feature                  |   Importance |   Coefficient |   Relative_Pct |
|:-------------------------|-------------:|--------------:|---------------:|
| recency_days             |  0.00803619  |   0.00803619  |          36.7  |
| avg_ticket_size          |  0.00549905  |   0.00549905  |          25.11 |
| app_logins_monthly       |  0.00154533  |  -0.00154533  |           7.06 |
| monthly_transactions     |  0.00136602  |  -0.00136602  |           6.24 |
| age                      |  0.00121012  |   0.00121012  |           5.53 |
| wealth_deceleration_risk |  0.000764501 |   0.000764501 |           3.49 |
| trans_amount_3m          |  0.000510585 |  -0.000510585 |           2.33 |
| friction_index           |  0.00039961  |   0.00039961  |           1.82 |
| robust_trans_trend_ratio |  0.000303532 |  -0.000303532 |           1.39 |
| robust_recency_days      |  0.000286491 |   0.000286491 |           1.31 |
| germany_high_balance     |  0.000250222 |   0.000250222 |           1.14 |
| log_recency_days         |  0.000226465 |   0.000226465 |           1.03 |
| credit_score             |  0.000168858 |  -0.000168858 |           0.77 |
| std_age                  |  0.000113666 |   0.000113666 |           0.52 |
| is_spending_collapsing   |  0.00011127  |   0.00011127  |           0.51 |

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
