# BÁO CÁO THỰC NGHIỆM MÔ HÌNH XGBOOST & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-21 20:20:55`  
**Mô hình cốt lõi:** `XGBoost`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **XGBoost** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

| Experiment                                  |   Num_Features |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Log_Loss |
|:--------------------------------------------|---------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|-----------:|
| 1. Full Pipeline (Default)                  |             55 |    0.9999 |   0.9996 |     0.9876 |      0.995  |   0.9803 |     0.995  |     0.0132 |
| 2. No SMOTE (Native scale_pos_weight)       |             55 |    0.9999 |   0.9997 |     0.9889 |      0.9926 |   0.9853 |     0.9955 |     0.011  |
| 3. No Domain Features                       |             44 |    0.9999 |   0.9996 |     0.9864 |      0.9925 |   0.9803 |     0.9945 |     0.0123 |
| 4. No Target Encoding                       |             51 |    0.9999 |   0.9996 |     0.9876 |      0.995  |   0.9803 |     0.995  |     0.0127 |
| 5. No One-Hot Encoding                      |             43 |    0.9999 |   0.9996 |     0.9876 |      0.995  |   0.9803 |     0.995  |     0.0125 |
| 6. No Scaling & No Skewness Transform       |             42 |    0.9995 |   0.9981 |     0.9754 |      0.9754 |   0.9754 |     0.99   |     0.4978 |
| 7. Top 15 Features Only (Feature Selection) |             15 |    0.9998 |   0.9993 |     0.9839 |      0.9925 |   0.9754 |     0.9935 |     0.0185 |

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **2. No SMOTE (Native scale_pos_weight)**
- **ROC-AUC:** `0.9999` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `0.9997` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.9889` (Precision: `0.9926`, Recall: `0.9853`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `401`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `6`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1590`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `3`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight (Kịch bản 2):**
   - Đối với XGBoost: SMOTE tăng kích thước tập huấn luyện từ 8,000 lên ~12,740 dòng. Native class_weight='balanced' tái cân bằng trực tiếp trên hàm mất mát mà không cần sinh mẫu tổng hợp, phù hợp khi runtime là ưu tiên.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `0.9998`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** Average Gain (mức giảm hàm mục tiêu bình quân khi chia nút)

| Feature                     |   Importance |   Relative_Pct |
|:----------------------------|-------------:|---------------:|
| is_spending_collapsing      |    469.671   |          42.9  |
| robust_trans_trend_ratio    |    256.305   |          23.41 |
| trans_trend_ratio           |    196.397   |          17.94 |
| interaction_velocity        |     24.4265  |           2.23 |
| log_recency_days            |     23.9148  |           2.18 |
| recency_days                |     19.4984  |           1.78 |
| robust_monthly_transactions |     14.7661  |           1.35 |
| monthly_transactions        |     11.366   |           1.04 |
| robust_recency_days         |      6.45925 |           0.59 |
| has_unresolved_complaint    |      5.89662 |           0.54 |
| app_logins_monthly          |      4.70191 |           0.43 |
| friction_index              |      4.30723 |           0.39 |
| wealth_deceleration_risk    |      3.76241 |           0.34 |
| tenure                      |      3.24413 |           0.3  |
| products_number             |      2.82392 |           0.26 |

---

## 4. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

### 4.1. So sánh cả 3 model (LR / RF / XGBoost)
```bash
python main.py --mode compare
```

### 4.2. Ablation cho một model cụ thể
```bash
python main.py --mode ablation --model-type xgboost  # lr | rf | xgboost
```

### 4.3. Train đơn lẻ với custom params
```bash
python main.py --mode train --model-type xgboost
```

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
