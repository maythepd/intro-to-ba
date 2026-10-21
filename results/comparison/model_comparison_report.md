# BÁO CÁO ĐỐI ĐẦU VÀ SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH (MODEL COMPARISON)
## LOGISTIC REGRESSION vs. RANDOM FOREST vs. XGBOOST
### DỰ BÁO RỦI RO RỜI BỎ KHÁCH HÀNG (DIGITAL BANK CHURN PREDICTION)

**Thời gian thực thi:** `2026-10-21 20:19:12`  
**Tập dữ liệu:** `data/cleaned_bank_data.csv` (2,000 khách hàng tập Test)  
**Cấu hình hợp nhất:** [`config/config.yaml`](../config/config.yaml)  

---

## 1. BẢNG TỔNG HỢP SO SÁNH CHỈ SỐ HIỆU NĂNG (BENCHMARK TABLE)

Bảng đối đầu trực tiếp giữa 3 họ thuật toán đại diện: Tuyến tính (Logistic Regression), Ensemble Bagging (Random Forest), và Gradient Boosting (XGBoost):

| Model   |   ROC_AUC |   PR_AUC |   F1_Score |   Precision |   Recall |   Accuracy |   Balanced_Accuracy |   Log_Loss |
|:--------|----------:|---------:|-----------:|------------:|---------:|-----------:|--------------------:|-----------:|
| LR      |    0.902  |   0.7246 |     0.639  |      0.505  |   0.8698 |      0.8   |              0.826  |     0.4202 |
| RF      |    0.9998 |   0.9991 |     0.9853 |      0.9853 |   0.9853 |      0.994 |              0.9907 |     0.0227 |
| XGBOOST |    0.9999 |   0.9996 |     0.9876 |      0.995  |   0.9803 |      0.995 |              0.9895 |     0.0132 |

---

## 2. PHÂN TÍCH CHUYÊN SÂU TỪ GÓC ĐỘ BUSINESS ANALYTICS & MACHINE LEARNING

### 2.1. Mô hình chiến thắng (Champion Model): **XGBOOST**
- **ROC-AUC:** `0.9999` — Xếp hạng xác suất khách hàng rời bỏ chính xác nhất trên toàn bộ miền phân bố.
- **PR-AUC:** `0.9996` — Tiêu chuẩn vàng cho tập dữ liệu mất cân bằng (tỷ lệ churn ~20.37%).
- **F1-Score:** `0.9876` (Precision: `0.9950`, Recall: `0.9803`).
- **Khả năng bắt khách hàng rời bỏ:** TP = `399` / `407` khách hàng thực tế rời bỏ.

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

| Feature              |   Relative_Pct |
|:---------------------|---------------:|
| recency_days         |          36.7  |
| avg_ticket_size      |          25.11 |
| app_logins_monthly   |           7.06 |
| monthly_transactions |           6.24 |
| age                  |           5.53 |

#### Top 5 Đặc trưng — Random Forest

| Feature                  |   Relative_Pct |
|:-------------------------|---------------:|
| trans_trend_ratio        |          18.99 |
| robust_trans_trend_ratio |          17.61 |
| recency_days             |           9.19 |
| robust_recency_days      |           9.1  |
| log_recency_days         |           8.03 |

#### Top 5 Đặc trưng — XGBoost

| Feature                  |   Relative_Pct |
|:-------------------------|---------------:|
| robust_trans_trend_ratio |          40.94 |
| trans_trend_ratio        |          31.11 |
| is_spending_collapsing   |           7.6  |
| robust_recency_days      |           3.82 |
| interaction_velocity     |           2.54 |



---

## 4. KHUYẾN NGHỊ VẬN HÀNH THỰC TẾ (PRODUCTION STRATEGY)

1. **Triển khai Mô hình chính (Primary Scoring Engine):**
   - Sử dụng **XGBoost** làm engine tính điểm rủi ro rời bỏ định kỳ hàng ngày (Batch Scoring) cho toàn bộ khách hàng hoạt động, cung cấp danh sách ưu tiên cao cho đội ngũ chăm sóc khách hàng (Retention RM).
2. **Mô hình giải trình & Baseline (Challenger / Fallback):**
   - Sử dụng **Logistic Regression** để kiểm tra tính nhất quán và giải trình các nguyên nhân rời bỏ vĩ mô cho cấp quản lý.

---
*Báo cáo được sinh tự động bởi hệ thống Digital Bank Churn Prediction Pipeline — Module 5: Model Development.*
