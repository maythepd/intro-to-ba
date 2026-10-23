# BÁO CÁO ĐỐI ĐẦU VÀ SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH (MODEL COMPARISON)
## LOGISTIC REGRESSION vs. RANDOM FOREST vs. XGBOOST
### DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (DIGITAL BANK CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-23 23:56:56`  
**Tập dữ liệu:** `data/full_bank_data.csv` (2,000 khách hàng tập Test)  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ HIỆU NĂNG (BENCHMARK TABLE)

Bảng đối đầu trực tiếp giữa 3 họ thuật toán đại diện: Tuyến tính (Logistic Regression), Ensemble Bagging (Random Forest), và Gradient Boosting (XGBoost):

| Model   |   CV_ROC_AUC |   CV_PR_AUC |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Balanced_Accuracy |   Log_Loss |
|:--------|-------------:|------------:|----------:|---------:|-----------:|------------:|---------:|-----------:|--------------------:|-----------:|
| LR      |       0.6073 |      0.2674 |    0.601  |   0.2622 |     0.3608 |      0.2576 |   0.602  |     0.566  |              0.5794 |     0.6811 |
| RF      |       0.8539 |      0.644  |    0.856  |   0.6565 |     0.6035 |      0.6717 |   0.5479 |     0.8535 |              0.7397 |     0.3823 |
| XGBOOST |       0.8609 |      0.6893 |    0.8665 |   0.7056 |     0.6194 |      0.7125 |   0.5479 |     0.863  |              0.7457 |     0.339  |

---

## 2. PHÂN TÍCH CHUYÊN SÂU TỪ GÓC ĐỘ BUSINESS ANALYTICS & MACHINE LEARNING

### 2.1. Mô hình chiến thắng (Champion Model): **XGBOOST**
- **ROC-AUC:** `0.8665` — Xếp hạng xác suất khách hàng rời bỏ chính xác nhất trên toàn bộ miền phân bố.
- **PR-AUC:** `0.7056` — Tiêu chuẩn vàng cho tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.6194` (Precision: `0.7125`, Recall: `0.5479`).
- **Khả năng bắt khách hàng rời bỏ:** TP = `223` / `407` khách hàng thực tế rời bỏ.

### 2.2. So sánh đặc tính của 3 họ thuật toán:
1. **Logistic Regression (Linear Baseline & Interpretability):**
   - **Ưu điểm:** Tốc độ huấn luyện và suy luận nhanh nhất (< 1ms). Trực quan hóa tác động biên (marginal effects) của từng biến lên log-odds thông qua các trọng số $\beta$. Phù hợp để giải trình với ban kiểm soát tuân thủ (Compliance / Risk Auditing).
   - **Hạn chế:** Không tự động nắm bắt được các mối tương tác phi tuyến tính phức tạp (như tỷ lệ tương tác tuổi tác x điểm tín dụng x số dư).
2. **Random Forest (Bagging Ensemble & Stability):**
   - **Ưu điểm:** Giảm phương sai (variance reduction) vượt trội nhờ cơ chế bootstrap aggregating và feature sub-sampling. Kháng nhiễu và outliers tốt, không đòi hỏi giả định phân phối.
   - **Hạn chế:** Kích thước model lớn, độ phức tạp tính toán cao hơn LR và khả năng tối ưu hàm mục tiêu biên kém linh hoạt hơn boosting.
3. **XGBoost (Gradient Boosting & Performance):**
   - **Ưu điểm:** Tối ưu hóa trực tiếp hàm mục tiêu và gradient bậc hai (Hessian). Khả năng phát hiện mẫu hành vi rời bỏ tinh vi nhất từ các đặc trưng kỹ nghệ tài chính.
   - **Hạn chế:** Cần tinh chỉnh hyperparameter cẩn thận để tránh overfitting trên các tập con nhỏ.

---

## 3. TOP ĐẶC TRƯNG HÀNG ĐẦU CỦA TỪNG MÔ HÌNH

#### Top 5 Đặc trưng — Logistic Regression

| Feature                 |   Relative_Pct |
|:------------------------|---------------:|
| avg_ticket_size         |          13.37 |
| credit_score            |          11.58 |
| salary_to_balance_ratio |           7.93 |
| age                     |           7.27 |
| complaint_recency_days  |           4.7  |

#### Top 5 Đặc trưng — Random Forest

| Feature              |   Relative_Pct |
|:---------------------|---------------:|
| is_product_sweetspot |          13.53 |
| std_age              |           8.3  |
| age                  |           6.53 |
| gender_target_enc    |           4.78 |
| active_member        |           4.19 |

#### Top 5 Đặc trưng — XGBoost

| Feature              |   Relative_Pct |
|:---------------------|---------------:|
| is_product_sweetspot |          19.29 |
| gender_Female        |           4.9  |
| active_member        |           4.04 |
| country_Germany      |           3.89 |
| gender_Male          |           3.23 |



---

## 4. KHUYẾN NGHỊ VẬN HÀNH THỰC TẾ (PRODUCTION STRATEGY)

1. **Triển khai Mô hình chính (Primary Scoring Engine):**
   - Sử dụng **XGBoost** làm engine tính điểm rủi ro rời bỏ định kỳ hàng ngày (Batch Scoring) cho toàn bộ khách hàng hoạt động, cung cấp danh sách ưu tiên cao cho đội ngũ chăm sóc khách hàng (Retention RM).
2. **Mô hình giải trình & Baseline (Challenger / Fallback):**
   - Sử dụng **Logistic Regression** để kiểm tra tính nhất quán và giải trình các nguyên nhân rời bỏ vĩ mô cho cấp quản lý.

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
