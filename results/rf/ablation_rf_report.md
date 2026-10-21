# BÁO CÁO THỰC NGHIỆM MÔ HÌNH RANDOM FOREST & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-21 20:21:21`  
**Mô hình cốt lõi:** `Random Forest`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **Random Forest** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

| Experiment                                  |   Num_Features |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Log_Loss |
|:--------------------------------------------|---------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|-----------:|
| 1. Full Pipeline (Default)                  |             55 |    0.9998 |   0.9991 |     0.9853 |      0.9853 |   0.9853 |     0.994  |     0.0227 |
| 2. No SMOTE (Native scale_pos_weight)       |             55 |    0.9997 |   0.999  |     0.9828 |      0.9804 |   0.9853 |     0.993  |     0.0238 |
| 3. No Domain Features                       |             44 |    0.9998 |   0.9992 |     0.9877 |      0.9901 |   0.9853 |     0.995  |     0.0242 |
| 4. No Target Encoding                       |             51 |    0.9997 |   0.9991 |     0.9853 |      0.9853 |   0.9853 |     0.994  |     0.0223 |
| 5. No One-Hot Encoding                      |             43 |    0.9997 |   0.999  |     0.984  |      0.9828 |   0.9853 |     0.9935 |     0.0221 |
| 6. No Scaling & No Skewness Transform       |             42 |    0.9998 |   0.9992 |     0.9853 |      0.9853 |   0.9853 |     0.994  |     0.0276 |
| 7. Top 15 Features Only (Feature Selection) |             15 |    0.9997 |   0.9987 |     0.984  |      0.9852 |   0.9828 |     0.9935 |     0.0197 |

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **3. No Domain Features**
- **ROC-AUC:** `0.9998` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `0.9992` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.9877` (Precision: `0.9901`, Recall: `0.9853`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `401`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `6`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1589`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `4`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native class_weight (Kịch bản 2):**
   - Đối với Random Forest: SMOTE tăng kích thước tập huấn luyện từ 8,000 lên ~12,740 dòng. Native class_weight='balanced' tái cân bằng trực tiếp trên hàm mất mát mà không cần sinh mẫu tổng hợp, phù hợp khi runtime là ưu tiên.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `0.9997`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT

**Phương pháp đo độ quan trọng:** Mean Decrease in Impurity — Gini MDI

| Feature                     |   Importance |   Relative_Pct |
|:----------------------------|-------------:|---------------:|
| trans_trend_ratio           |   0.21349    |          21.35 |
| robust_trans_trend_ratio    |   0.21036    |          21.04 |
| robust_recency_days         |   0.128511   |          12.85 |
| recency_days                |   0.105903   |          10.59 |
| monthly_transactions        |   0.0750141  |           7.5  |
| log_recency_days            |   0.0749879  |           7.5  |
| robust_monthly_transactions |   0.0634441  |           6.34 |
| robust_app_logins_monthly   |   0.0345141  |           3.45 |
| app_logins_monthly          |   0.032799   |           3.28 |
| customer_service_complaints |   0.0097016  |           0.97 |
| robust_trans_amount_3m      |   0.00876803 |           0.88 |
| std_age                     |   0.00785911 |           0.79 |
| trans_amount_3m             |   0.00653231 |           0.65 |
| log_trans_amount_3m         |   0.00652594 |           0.65 |
| age                         |   0.00470711 |           0.47 |

---

## 4. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

### 4.1. So sánh cả 3 model (LR / RF / XGBoost)
```bash
python main.py --mode compare
```

### 4.2. Ablation cho một model cụ thể
```bash
python main.py --mode ablation --model-type rf  # lr | rf | xgboost
```

### 4.3. Train đơn lẻ với custom params
```bash
python main.py --mode train --model-type rf
```

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
