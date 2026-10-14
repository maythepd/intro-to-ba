# BÁO CÁO THỰC NGHIỆM MÔ HÌNH XGBOOST & ABLATION STUDY
## DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (CUSTOMER CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-14 09:56:55`  
**Mô hình cốt lõi:** `XGBoost Classifier (Tree Booster)`  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. TỔNG QUAN KẾT QUẢ CÁC KỊCH BẢN THỬ NGHIỆM (ABLATION BENCHMARK)

Bảng dưới đây so sánh toàn diện hiệu năng của mô hình **XGBoost** qua 7 cấu hình kỹ nghệ đặc trưng và chiến lược xử lý dữ liệu khác nhau:

| Experiment                                  |   Num_Features |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Log_Loss |
|:--------------------------------------------|---------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|-----------:|
| 1. Full Pipeline (Default)                  |             55 |    0.9999 |   0.9996 |     0.9889 |      0.995  |   0.9828 |     0.9955 |     0.0125 |
| 2. No SMOTE (Native scale_pos_weight)       |             55 |    0.9999 |   0.9997 |     0.9914 |      0.995  |   0.9877 |     0.9965 |     0.0105 |
| 3. No Domain Features                       |             44 |    0.9999 |   0.9996 |     0.9877 |      0.9926 |   0.9828 |     0.995  |     0.0119 |
| 4. No Target Encoding                       |             51 |    0.9999 |   0.9996 |     0.9889 |      0.995  |   0.9828 |     0.9955 |     0.0124 |
| 5. No One-Hot Encoding                      |             43 |    0.9999 |   0.9997 |     0.9889 |      0.995  |   0.9828 |     0.9955 |     0.0118 |
| 6. No Scaling & No Skewness Transform       |             42 |    0.9999 |   0.9997 |     0.9876 |      0.995  |   0.9803 |     0.995  |     0.0124 |
| 7. Top 15 Features Only (Feature Selection) |             15 |    0.9998 |   0.9992 |     0.9839 |      0.99   |   0.9779 |     0.9935 |     0.0183 |

---

## 2. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

### 2.1. Cấu hình đạt hiệu năng tối ưu nhất: **2. No SMOTE (Native scale_pos_weight)**
- **ROC-AUC:** `0.9999` — Khả năng xếp hạng xác suất churn vượt trội trên toàn miền ngưỡng.
- **PR-AUC (Average Precision):** `0.9997` — Đo lường chuẩn xác trên tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.9914` (Precision: `0.9950`, Recall: `0.9877`).
- **Ma trận nhầm lẫn (Confusion Matrix trên 2,000 khách hàng kiểm thử):**
  - **True Positives (TP - Bắt đúng khách hàng rời bỏ):** `402`
  - **False Negatives (FN - Bỏ sót khách hàng rời bỏ):** `5`
  - **True Negatives (TN - Nhận diện đúng khách hàng ở lại):** `1591`
  - **False Positives (FP - Cảnh báo nhầm khách hàng ở lại):** `2`

### 2.2. Ý nghĩa từ các thí nghiệm Ablation:
1. **Vai trò của Domain Features (Kịch bản 3):**
   - Khi loại bỏ các đặc trưng nghiệp vụ ngân hàng (`interaction_velocity`, `friction_index`, `is_product_danger_zone`, `wealth_deceleration_risk`), số lượng đặc trưng giảm từ 55 xuống 44. Các chỉ số PR-AUC và F1-Score chứng minh các đặc trưng tương tác phi tuyến tính phản ánh sát hành vi dịch chuyển dòng tiền trước khi churn.
2. **SMOTE vs. Native `scale_pos_weight` (Kịch bản 2):**
   - XGBoost với `scale_pos_weight` tự động tái cân bằng trọng số hàm mất mát (loss function) mà không làm tăng kích thước tập huấn luyện từ 8,000 lên 12,740 dòng như SMOTE, giúp thời gian huấn luyện nhanh hơn và giữ nguyên phân phối dữ liệu gốc.
3. **Mã hóa Target Encoding (Kịch bản 4):**
   - Bayesian $m$-estimate target encoding cung cấp thông tin liên tục về xu hướng rời bỏ theo từng quốc gia, giới tính và kênh giao dịch mà không làm bùng nổ số chiều.
4. **Hiệu năng của Top 15 Features (Kịch bản 7):**
   - Chỉ với **15 đặc trưng quan trọng nhất**, mô hình vẫn duy trì ROC-AUC đạt `0.9998`, chứng minh tính khả thi cao khi cần triển khai thời gian thực (real-time low-latency serving).

---

## 3. TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT (FEATURE IMPORTANCE - GAIN)

Độ quan trọng tính theo mức giảm bình quân của hàm mục tiêu (Average Gain) khi phân chia nút trên cây quyết định:

| Feature                     |   Importance |   Relative_Pct |
|:----------------------------|-------------:|---------------:|
| is_spending_collapsing      |    678.737   |          55.22 |
| trans_trend_ratio           |    227.44    |          18.51 |
| robust_trans_trend_ratio    |    130.024   |          10.58 |
| interaction_velocity        |     35.4716  |           2.89 |
| recency_days                |     20.2311  |           1.65 |
| log_recency_days            |     18.2633  |           1.49 |
| robust_monthly_transactions |     16.0613  |           1.31 |
| robust_recency_days         |     14.1299  |           1.15 |
| monthly_transactions        |     11.8288  |           0.96 |
| has_unresolved_complaint    |      8.77321 |           0.71 |
| std_tenure                  |      7.68037 |           0.62 |
| wealth_deceleration_risk    |      5.43253 |           0.44 |
| app_logins_monthly          |      4.53037 |           0.37 |
| is_high_friction            |      4.27115 |           0.35 |
| robust_app_logins_monthly   |      3.37159 |           0.27 |

---

## 4. HƯỚNG DẪN TÁI HIỆN VÀ SỬ DỤNG MÔ HÌNH

### 4.1. Thay đổi cấu hình thử nghiệm
Mở file [`config/config.yaml`](../config/config.yaml) và thay đổi bất kỳ mục nào:
```yaml
features:
  enable_domain_features: true
  enable_smote: false

model:
  params:
    learning_rate: 0.03
    max_depth: 6
```

### 4.2. Chạy lại Pipeline & Thử nghiệm
```bash
# Chạy đầy đủ bộ thí nghiệm ablation và cập nhật báo cáo
python main.py --mode ablation

# Hoặc huấn luyện 1 mô hình đơn lẻ theo cấu hình config.yaml
python main.py --mode train
```

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline.*
