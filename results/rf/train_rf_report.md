# BÁO CÁO HUẤN LUYỆN MÔ HÌNH RANDOM FOREST (TRAINING REPORT)
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian huấn luyện:** `2026-10-21 20:21:59`  
**Mô hình:** `Random Forest`  
**Artifact đã lưu:** `models/saved/rf_churn.pkl`  
**Cấu hình:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST

| Metric            | Value      |
|:------------------|:-----------|
| ROC_AUC           | 0.9998     |
| PR_AUC            | 0.9991     |
| F1_Score          | 0.9853     |
| Precision         | 0.9853     |
| Recall            | 0.9853     |
| Accuracy          | 0.9940     |
| Balanced_Accuracy | 0.9907     |
| Log_Loss          | 0.0227     |
| TP                | 401        |
| FP                | 6          |
| TN                | 1587       |
| FN                | 6          |
| Experiment        | Single Run |
| Model             | RF         |
| Num_Features      | 55         |
| SMOTE_Applied     | True       |

---

## 2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

- **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `401`
- **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `6`
- **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1587`
- **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `6`

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

| Feature                     |   Importance |   Relative_Pct |
|:----------------------------|-------------:|---------------:|
| trans_trend_ratio           |   0.189903   |          18.99 |
| robust_trans_trend_ratio    |   0.176149   |          17.61 |
| recency_days                |   0.0918991  |           9.19 |
| robust_recency_days         |   0.0909535  |           9.1  |
| log_recency_days            |   0.0802551  |           8.03 |
| interaction_velocity        |   0.053949   |           5.39 |
| monthly_transactions        |   0.0523009  |           5.23 |
| is_spending_collapsing      |   0.0487025  |           4.87 |
| robust_monthly_transactions |   0.0481217  |           4.81 |
| wealth_deceleration_risk    |   0.0453095  |           4.53 |
| robust_app_logins_monthly   |   0.0325766  |           3.26 |
| app_logins_monthly          |   0.028108   |           2.81 |
| friction_index              |   0.017496   |           1.75 |
| is_product_sweetspot        |   0.00890028 |           0.89 |
| trans_amount_3m             |   0.00518108 |           0.52 |

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
