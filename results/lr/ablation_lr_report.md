# BÁO CÁO THỰC NGHIỆM MÔ HÌNH LOGISTIC REGRESSION & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-21 20:20:38`  
**Mô hình cốt lõi:** `Logistic Regression`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **Logistic Regression** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

| Experiment                                  |   Num_Features |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Log_Loss |
|:--------------------------------------------|---------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|-----------:|
| 1. Full Pipeline (Default)                  |             55 |    0.902  |   0.7246 |     0.639  |      0.505  |   0.8698 |     0.8    |     0.4202 |
| 2. No SMOTE (Native scale_pos_weight)       |             55 |    0.8803 |   0.6667 |     0.605  |      0.4667 |   0.86   |     0.7715 |     0.4578 |
| 3. No Domain Features                       |             44 |    0.8762 |   0.6555 |     0.5972 |      0.4662 |   0.8305 |     0.772  |     0.4508 |
| 4. No Target Encoding                       |             51 |    0.902  |   0.7246 |     0.639  |      0.505  |   0.8698 |     0.8    |     0.4202 |
| 5. No One-Hot Encoding                      |             43 |    0.902  |   0.7246 |     0.639  |      0.505  |   0.8698 |     0.8    |     0.4203 |
| 6. No Scaling & No Skewness Transform       |             42 |    0.9018 |   0.7242 |     0.639  |      0.505  |   0.8698 |     0.8    |     0.4205 |
| 7. Top 15 Features Only (Feature Selection) |             15 |    0.9998 |   0.9994 |     0.9853 |      0.9853 |   0.9853 |     0.994  |     0.0141 |

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **7. Top 15 Features Only (Feature Selection)**
- **ROC-AUC:** `0.9998` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `0.9994` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.9853` (Precision: `0.9853`, Recall: `0.9853`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `401`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `6`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1587`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `6`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight (Kịch bản 2):**
   - Đối với Logistic Regression: SMOTE tăng kích thước tập huấn luyện từ 8,000 lên ~12,740 dòng. Native class_weight='balanced' tái cân bằng trực tiếp trên hàm mất mát mà không cần sinh mẫu tổng hợp, phù hợp khi runtime là ưu tiên.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `0.9998`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** Absolute Coefficient |β| (proxy độ quan trọng tuyến tính)

| Feature                     |   Importance |   Coefficient |   Relative_Pct |
|:----------------------------|-------------:|--------------:|---------------:|
| robust_trans_trend_ratio    |    3.46309   |    -3.46309   |          74.15 |
| friction_index              |    0.36369   |     0.36369   |           7.79 |
| monthly_transactions        |    0.358297  |    -0.358297  |           7.67 |
| wealth_deceleration_risk    |    0.20911   |     0.20911   |           4.48 |
| app_logins_monthly          |    0.184863  |    -0.184863  |           3.96 |
| recency_days                |    0.0912738 |     0.0912738 |           1.95 |
| trans_trend_ratio           |    0         |     0         |           0    |
| interaction_velocity        |    0         |     0         |           0    |
| robust_recency_days         |    0         |     0         |           0    |
| log_recency_days            |    0         |     0         |           0    |
| is_spending_collapsing      |    0         |     0         |           0    |
| robust_monthly_transactions |    0         |     0         |           0    |
| robust_app_logins_monthly   |    0         |     0         |           0    |
| has_unresolved_complaint    |    0         |     0         |           0    |
| customer_service_complaints |    0         |     0         |           0    |

---

## 4. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

### 4.1. So sánh cả 3 model (LR / RF / XGBoost)
```bash
python main.py --mode compare
```

### 4.2. Ablation cho một model cụ thể
```bash
python main.py --mode ablation --model-type lr  # lr | rf | xgboost
```

### 4.3. Train đơn lẻ với custom params
```bash
python main.py --mode train --model-type lr
```

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
