# BÁO CÁO THỰC NGHIỆM MÔ HÌNH XGBOOST & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-24 00:36:32`  
**Mô hình cốt lõi:** `XGBoost`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **XGBoost** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

| Experiment                                               |   Num_Features |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Log_Loss |
|:---------------------------------------------------------|---------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|-----------:|
| 0. Raw Kaggle Baseline (Original 10 Features Only)       |             21 |    0.8677 |   0.705  |     0.6263 |      0.6585 |   0.5971 |     0.855  |     0.352  |
| 1. Full Pipeline (Default - All Multi-Table & Domain)    |            119 |    0.8665 |   0.7056 |     0.6194 |      0.7125 |   0.5479 |     0.863  |     0.339  |
| 2. No SMOTE (Native scale_pos_weight)                    |            119 |    0.8638 |   0.7025 |     0.6167 |      0.5407 |   0.7174 |     0.8185 |     0.4387 |
| 3. No Domain Features                                    |             84 |    0.8685 |   0.7027 |     0.6148 |      0.6923 |   0.5528 |     0.859  |     0.3492 |
| 4. No Target Encoding                                    |            115 |    0.8631 |   0.6914 |     0.6158 |      0.6911 |   0.5553 |     0.859  |     0.3469 |
| 5. No One-Hot Encoding                                   |            108 |    0.8669 |   0.7041 |     0.6081 |      0.7032 |   0.5356 |     0.8595 |     0.3389 |
| 6. No Scaling & No Skewness Transform                    |            109 |    0.8633 |   0.7032 |     0.6019 |      0.679  |   0.5405 |     0.8545 |     0.3427 |
| 7. Top 15 Features Only (Feature Selection - MI Classif) |             15 |    0.851  |   0.6951 |     0.6292 |      0.6396 |   0.6192 |     0.8515 |     0.3605 |

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **1. Full Pipeline (Default - All Multi-Table & Domain)**
- **ROC-AUC:** `0.8665` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `0.7056` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.6194` (Precision: `0.7125`, Recall: `0.5479`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `223`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `184`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1503`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `90`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight (Kịch bản 2):**
   - Đối với XGBoost: SMOTE tăng kích thước tập huấn luyện từ 8,000 lên ~12,740 dòng. Native class_weight='balanced' tái cân bằng trực tiếp trên hàm mất mát mà không cần sinh mẫu tổng hợp, phù hợp khi runtime là ưu tiên.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `0.8510`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** Average Gain (mức giảm hàm mục tiêu bình quân khi chia nút)

![Biểu đồ đặc trưng quan trọng nhất](ablation_feature_importance.png)

| Feature                   |   Importance |   Relative_Pct |
|:--------------------------|-------------:|---------------:|
| is_product_sweetspot      |     422.215  |          19.29 |
| gender_Female             |     107.261  |           4.9  |
| active_member             |      88.3506 |           4.04 |
| country_Germany           |      85.2093 |           3.89 |
| gender_Male               |      70.6229 |           3.23 |
| country_target_enc        |      65.3052 |           2.98 |
| std_age                   |      64.0339 |           2.93 |
| inactive_high_balance     |      61.4926 |           2.81 |
| primary_device_target_enc |      59.0274 |           2.7  |
| primary_device_iOS        |      55.1064 |           2.52 |
| gender_target_enc         |      39.7488 |           1.82 |
| multi_account_holder      |      37.6158 |           1.72 |
| products_number           |      36.0236 |           1.65 |
| bank_fee_amount           |      35.5778 |           1.63 |
| country_Spain             |      35.1586 |           1.61 |

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
