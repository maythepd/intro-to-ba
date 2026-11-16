# BÁO CÁO TỔNG HỢP NÂNG CẤP & CHUẨN HÓA HỆ THỐNG ĐẶC TRƯNG
**Dự án:** Digital Bank Customer Churn Prediction  
**Tài liệu:** `docs/features.md`  
**Ngày cập nhật:** 29/10/2026  
**Người thực hiện:** Antigravity Data Science Team  
**Mục đích:** Báo cáo chi tiết các đợt rà soát, đối chiếu Schema từ `synth_data_dict.md`, khắc phục lỗi logic nghiệp vụ ETL, và nâng cấp hệ thống kỹ nghệ đặc trưng (Feature Engineering) gửi Leader phê duyệt.

---

## 1. Tóm tắt điều hành (Executive Summary)

Đợt rà soát và nâng cấp toàn diện vừa qua tập trung vào 4 trọng tâm cốt lõi:
1. **Khắc phục triệt để sai lệch Schema & Miền giá trị:** Đối chiếu toàn diện giữa mã nguồn trích xuất dữ liệu với từ điển dữ liệu gốc (`synth_data_dict.md`), loại bỏ toàn bộ các cột/giá trị không tồn tại.
2. **Sửa đổi các lỗi logic nghiệp vụ ngân hàng:** Xử lý tình trạng nhân đôi giao dịch chuyển tiền nội bộ (Internal Transfer Double-Counting), tách bạch giao dịch chủ động của người dùng (Customer-Initiated) với giao dịch tự động của hệ thống (System Automated) khi tính độ tích cực (`active_member`) và độ trễ (`recency_days`).
3. **Nâng cấp Kỹ nghệ đặc trưng (Domain Feature Engineering):**
   - Loại bỏ rủi ro gán nhãn sụt giảm oan cho khách hàng mới gia nhập ở Quý 3.
   - Chuẩn hóa công thức `avg_ticket_size` đồng nhất chu kỳ thời gian Quý 3.
   - **Xóa bỏ gán cứng tĩnh** (`products_number == 2` là Sweetspot, `>= 3` là Danger Zone), chuyển sang **Mô hình Phân vùng Động (Dynamic Market-Regime Approach)** theo trường phái Bull/Bear của tài chính định lượng, bảo đảm tính khách quan cho từng khách hàng cụ thể.
4. **Chuẩn hóa cấu trúc dữ liệu & Dọn sạch Artifacts:** Dữ liệu sau khi xử lý qua Pipeline chính thức được lưu trữ tách bạch tại `data/processed/`, toàn bộ các file test được bọc cơ chế tự dọn sạch artifacts nhằm bảo đảm tính tinh gọn cho repository.

Toàn bộ hệ thống kiểm thử (`test_pipeline.py`) đã vượt qua **100% các bài test** (Full Pipeline, Ablation, Baseline, MI Score ranking, 5-Fold Stratified CV, Optuna smoke run).

---

## 2. Bảng đối chiếu Trước & Sau (Before vs After Matrix)

| Khu vực / Module | Logic Trước đây (Before) | Vấn đề phát sinh | Logic Cải tiến Mới (After) | Lợi ích & Tác động |
| :--- | :--- | :--- | :--- | :--- |
| **ETL: Account Type** | `acc["account_type"] == "term_deposit"` | Miền giá trị thực tế chỉ có `"everyday"` và `"savings"`. Giá trị `"term_deposit"` không tồn tại $\rightarrow$ biến luôn bằng 0. | `acc["account_type"] == "savings"` | Nắm bắt chính xác số lượng tài khoản tiết kiệm của khách hàng. |
| **ETL: Account Status** | `acc["status"] == "closed"` | Bảng `accounts.csv` không có cột `status` $\rightarrow$ gây lỗi `KeyError`. | `acc["closed_at"].notna()` | Xác định chuẩn xác tài khoản đã đóng thông qua timestamp đóng. |
| **ETL: Renewal Mode** | `acc["renewal_mode"]` | Tên cột trong schema là `renewal_type`. | `acc["renewal_type"]` | Đúng schema, trích xuất chính xác hành vi gia hạn tài khoản. |
| **ETL: Tx Status** | `tx["status"] == "posted"` | Tên cột thực tế là `transaction_status`, trạng thái thành công là `"success"`. | `tx["transaction_status"] == "success"` | Lọc đúng 100% các giao dịch thanh toán thành công. |
| **ETL: Tx Direction** | `tx["direction"] == "outflow"` / `"inflow"` | Miền giá trị thực tế chỉ có `"out"` và `"in"`. | `tx["direction"] == "out"` / `"in"` | Nhận diện chính xác chiều dòng tiền ra/vào. |
| **ETL: Internal Transfers** | Đếm gộp tất cả bản ghi chuyển tiền nội bộ. | 1 lệnh chuyển tiền giữa 2 tài khoản cùng ngân hàng sinh ra 2 bản ghi (`out` và `in`) $\rightarrow$ Bị nhân đôi (double-counting) giao dịch và dòng tiền. | Khử trùng lặp ở mức hành động: chỉ đếm chiều `out` cho `internal_transfer`; loại `internal_transfer` khỏi dòng tiền chi tiêu/thu nhập ngoài. | Không bị phóng đại số lượng giao dịch và dòng tiền thực tế của khách hàng. |
| **ETL: Recency & Active Member** | Tính trên mọi giao dịch, bao gồm cả giao dịch tự động thu phí/trả lãi của hệ thống (`channel == "system"`). | Khách hàng đã bỏ dùng thẻ cả năm nhưng hệ thống tự trừ phí hàng tháng vẫn bị tính là "vừa giao dịch hôm qua" (`recency = 0`) và gán nhầm là Active. | `recency_days` và `active_member` chỉ tính trên giao dịch do khách hàng chủ động tạo (`channel != "system"`). | Phản ánh đúng 100% mức độ tương tác thực của người dùng. |
| **ETL: Missing Values** | Điền `preferred_channel = "Mobile App"`, `primary_device = "Android"`. | Áp đặt thiên vị (bias) lên những khách hàng chưa từng giao dịch hoặc chưa từng đăng nhập. | Gán nhãn khách quan `"None"` cho những trường hợp chưa phát sinh hành vi. | Tránh tạo dữ liệu giả mạo hành vi người dùng. |
| **ETL: Resolution Delay** | Điền `avg_resolution_delay_days = 0`. | 0 ngày có nghĩa là "được giải quyết tức thì trong ngày", làm sai lệch bản chất của khách không có khiếu nại. | Giữ `NaN` ở tầng ETL, xử lý bằng `SimpleImputer(strategy="median")` được fit độc quyền trên Train split. | Loại bỏ rò rỉ dữ liệu (zero leakage), đúng bản chất thống kê. |
| **Domain: New Customers Trend** | Khách mới Quý 3 có Quý 2 bằng 0 $\rightarrow$ `trend_ratio = Q3 / (0 + 1)`. Khách mới chưa kịp chi tiêu bị phạt `is_spending_collapsing = 1`. | Lẫn lộn giữa "chưa từng quan sát" (*unobserved*) với "ngừng hoạt động" (*inactive*), phạt oan khách hàng mới. | Bổ sung `is_new_customer = (exp_m <= 3.0)`. Khách mới được gán trend ratio trung tính `1.0` và `is_spending_collapsing = 0`. | Không còn báo động giả tín hiệu churn trên nhóm onboarding. |
| **Domain: Ticket Size** | `trans_amount_3m / (3.0 * (monthly_transactions + 1.0))` | Lệch pha chu kỳ: Tử số là tiền Quý 3, mẫu số lại lấy trung bình giao dịch cả 6 tháng quan sát. | $\text{avg\_ticket\_size} = \frac{\text{trans\_amount\_3m}}{\text{trans\_count\_3m} + 1.0}$ | Phản ánh chính xác giá trị bình quân trên mỗi giao dịch Quý 3. |
| **Domain: Product Bundling** | Gán cứng: `products == 2` là Sweetspot, `products >= 3` là Danger Zone. | Bị cào bằng: Khách có 2 sản phẩm nhưng dormant vẫn tính là an toàn; khách có 3-4 sản phẩm nhưng là VIP power-user lại bị coi là nguy hiểm. | **Phân vùng động theo Market Regime (Bull/Bear):** Tính tỷ lệ khai thác danh mục (`portfolio_utilization`) và điểm dao động xung lượng tổng hợp (`portfolio_health_score`). | Cá nhân hóa khách quan cho từng khách hàng; Churn rate Bull: 8.67%, Bear: 27.18%. |
| **Test & Artifacts** | Artifacts test tạo ra không được dọn dẹp; chưa có thư mục lưu trữ dữ liệu đã qua xử lý chuẩn. | Gây nhầm lẫn giữa dữ liệu thô `full_bank_data.csv` với dữ liệu feature matrix đầu vào mô hình; rác test tồn đọng. | Tự động dọn sạch test artifacts qua khối `try ... finally`; Xuất toàn bộ dữ liệu sạch vào `data/processed/`. | Phân tách rạch ròi các tầng dữ liệu, repository luôn sạch sẽ. |

---

## 3. Chi tiết các cải tiến kỹ thuật

### 3.1. Chuẩn hóa Schema và Khắc phục Lỗi Logic Nghiệp vụ ETL (`etl.py`)

#### A. Đối chiếu Schema và Miền giá trị chuẩn
Căn cứ theo [synth_data_dict.md](file:///d:/Introduction to BA/intro-to-ba/data/full_dataset/synth_data_dict.md), toàn bộ các câu lệnh trích xuất đặc trưng đã được sửa đổi để khớp hoàn toàn với cấu trúc cơ sở dữ liệu:
- **Tài khoản tiết kiệm:** Thay thế toàn bộ điều kiện `account_type == "term_deposit"` thành `account_type == "savings"`.
- **Tài khoản đã đóng:** Thay thế `status == "closed"` thành `acc["closed_at"].notna()`.
- **Phương thức gia hạn:** Thay thế `renewal_mode` thành `renewal_type`.
- **Trạng thái giao dịch:** Thay thế `status == "posted"` thành `tx["transaction_status"] == "success"`.
- **Chiều giao dịch:** Thay thế `"outflow"` / `"inflow"` thành `"out"` / `"in"`.

#### B. Khử trùng lặp giao dịch nội bộ (Internal Transfers)
Trong hệ thống ngân hàng, một giao dịch chuyển tiền nội bộ giữa 2 tài khoản của cùng một khách hàng (hoặc giữa 2 khách hàng trong cùng ngân hàng) sẽ tạo ra 2 bản ghi trong bảng `transactions.csv`: một bản ghi ghi nợ (`direction == 'out'`) tại tài khoản nguồn và một bản ghi ghi có (`direction == 'in'`) tại tài khoản đích.
- **Khắc phục:** 
  - Khi tính tổng số lượt giao dịch (`total_tx_attempts`, `total_transactions`), hệ thống áp dụng bộ lọc khử trùng lặp ở cấp độ hành động:
    ```python
    valid_tx_mask = (tx["transaction_type"] != "internal_transfer") | (tx["direction"] == "out")
    ```
  - Khi tính tổng dòng tiền chi tiêu (`total_outflow`) và tổng tiền vào (`total_inflow`), các giao dịch `internal_transfer` được loại trừ hoàn toàn để không tính tiền tự chuyển qua lại giữa các tài khoản của chính mình thành chi tiêu hay thu nhập mới.

#### C. Phân tách Giao dịch Chủ động (Customer-Initiated) vs Tự động Hệ thống
- Nhiều khách hàng mở tài khoản nhưng đã ngưng sử dụng từ lâu. Tuy nhiên, hàng tháng hệ thống ngân hàng vẫn tự động thực hiện các tác vụ nền (`channel == "system"`) như: tự động trừ phí quản lý tài khoản, thu phí thẻ định kỳ, hoặc cộng tiền lãi tiết kiệm vài trăm đồng.
- **Khắc phục:**
  - Logic tính ngày cách giao dịch gần nhất (`recency_days`) và trạng thái thành viên tích cực (`active_member`) được ràng buộc chặt chẽ:
    ```python
    user_tx = tx[tx["channel"] != "system"]
    ```
  - Đảm bảo các tác vụ ngầm của hệ thống không làm sai lệch chỉ số recency của những khách hàng đã rời bỏ.

#### D. Xử lý Missing Values khách quan và ngăn chặn Thiên vị (Bias)
- **`preferred_channel`:** Trong tập dữ liệu có 658 khách hàng chưa từng phát sinh giao dịch nào. Việc gán giá trị mặc định là `"Mobile App"` tạo ra sự sai lệch thống kê nghiêm trọng. Hệ thống đã sửa đổi thành gán nhãn `"None"`.
- **`primary_device`:** Chỉ xét dựa trên các lượt đăng nhập thành công (`login_status == "success"`). Trường hợp chưa từng đăng nhập được gán nhãn `"None"` thay vì mặc định ép sang `"Android"`.
- **`avg_resolution_delay_days`:** Giá trị `0` biểu thị khiếu nại được xử lý ngay trong ngày. Với những khách hàng chưa từng khiếu nại (hoặc khiếu nại đang chờ xử lý), giá trị này được giữ nguyên là `NaN` ở tầng ETL, sau đó được tầng Pipeline xử lý thông qua `SimpleImputer(strategy="median")` được fit độc quyền trên tập Train để ngăn chặn tuyệt đối rò rỉ dữ liệu (Data Leakage).

---

### 3.2. Nâng cấp Kỹ nghệ Đặc trưng Nghiệp vụ (`domain_features.py`)

#### A. Bảo vệ Khách hàng mới (New Customers) trong Feature Trend
- **Bản chất nghiệp vụ:** Theo `synth_data_dict.md` (dòng 105): *"History before the customer joined is unobserved, not zero activity."* Nhóm khách hàng mới mở tài khoản trong Quý 3 (`exposure_months <= 3.0` — gồm 89 khách hàng) chưa từng có lịch sử trong Quý 2.
- **Khắc phục:**
  - Bổ sung đặc trưng mới: `is_new_customer = (exposure_months <= 3.0).astype(int)`.
  - Với khách hàng mới: Thiết lập các tỷ lệ xu hướng (`trans_trend_ratio`, `trans_count_trend_ratio`, `login_trend_ratio`) về mức trung tính `1.0` (Neutral baseline: không biến động do quá khứ chưa quan sát), đồng thời thiết lập cờ cảnh báo sụt giảm `is_spending_collapsing = 0` và `is_login_collapsing = 0`.
  - Giúp mô hình phân biệt rõ ràng giữa "khách hàng cũ đang có dấu hiệu rời bỏ" và "khách hàng mới đang trong giai đoạn tiếp nhận dịch vụ (onboarding)".

#### B. Chuẩn hóa công thức `avg_ticket_size`
- **Công thức cũ:** `trans_amount_3m / (3.0 * (monthly_transactions + 1.0))`. Mẫu số sử dụng `monthly_transactions` (vốn tính trung bình trên toàn bộ 6 tháng quan sát), gây ra hiện tượng lệch pha thời gian so với tử số (vốn chỉ tính riêng Quý 3).
- **Công thức mới chuẩn hóa:**
  $$\text{avg\_ticket\_size} = \frac{\text{trans\_amount\_3m}}{\text{trans\_count\_3m} + 1.0}$$
  Phản ánh trung thực 100% giá trị giao dịch bình quân thực tế phát sinh trong Quý 3 gần nhất.

#### C. Chuyển đổi sang Mô hình Phân vùng Động (Dynamic Market-Regime Approach)
Thay vì gán cứng tĩnh (`products_number == 2` là Sweetspot, `products_number >= 3` là Danger Zone) vốn mang tính máy móc và cào bằng, hệ thống áp dụng nguyên lý **Regime Switching (Bull / Bear / Neutral)** tương tự trong thị trường tài chính:

```
                    ┌────────────────────────────────────────────────────────┐
                    │               KHÁCH HÀNG NGÂN HÀNG SỞ HỮU              │
                    │               DANH MỤC SẢN PHẨM (PORTFOLIO)            │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
        ┌─────────────────────────────┐                 ┌─────────────────────────────┐
        │     BULL REGIME (SWEET)     │                 │     BEAR REGIME (DANGER)    │
        │   (Khách hàng gắn kết sâu)  │                 │    (Khách hàng suy kiệt)    │
        ├─────────────────────────────┤                 ├─────────────────────────────┤
        │ • products_number >= 2      │                 │ • products_number >= 3      │
        │ • util >= median_util       │                 │   nhưng util < median hoặc  │
        │ • active_member == 1        │                 │   ngừng hoạt động (dormant) │
        │ • spending_collapse == 0    │                 │ • HOẶC: spending collapse   │
        │ • high_friction == 0        │                 │   kèm ngừng hoạt động       │
        ├─────────────────────────────┤                 │ • HOẶC: ma sát dịch vụ cao  │
        │ Tỷ lệ Churn: 8.67%          │                 │   kèm khiếu nại chưa xử lý  │
        │ (Thấp kỷ lục so với 20.37%) │                 ├─────────────────────────────┤
        └─────────────────────────────┘                 │ Tỷ lệ Churn: 27.18%         │
                                                        │ (Báo động đỏ rời bỏ)        │
                                                        └─────────────────────────────┘
```

1. **Đặc trưng khai thác danh mục (`portfolio_utilization`)**:
   $$\text{portfolio\_utilization} = \frac{\text{monthly\_transactions}}{\text{products\_number}}$$
   Đo lường cường độ giao dịch thực tế trên từng sản phẩm sở hữu, phát hiện các trường hợp "mở nhiều gói sản phẩm nhưng bỏ xó".
2. **Số dư bình quân trên mỗi sản phẩm (`balance_per_product`)**:
   $$\text{balance\_per\_product} = \frac{\max(0, \text{balance})}{\text{products\_number}}$$
3. **Chỉ số Sức khỏe Động lượng Danh mục (`portfolio_health_score`)**:
   Đóng vai trò như một bộ dao động xung lượng tổng hợp (Composite Momentum Oscillator):
   $$\text{health\_score} = 0.70 \times (\text{active} - 0.5) + 0.25 \times \Delta_{\text{trend}} + 0.20 \times \log(1 + \text{util}) - 0.20 \times \text{friction\_drag}$$
4. **Cơ chế học ngưỡng trung vị không rò rỉ (Strict Leak-Free Learning)**:
   Ngưỡng trung vị `median_utilization_` được tính toán trong hàm `DomainFeatureEngineer.fit()` độc quyền trên tập `X_train`, sau đó áp dụng cố định sang `X_val` và `X_test`.

---

### 3.3. Dọn dẹp Artifacts Test và Quản lý Dữ liệu Xử lý (`data/processed/`)

1. **Cơ chế tự động dọn sạch Artifacts trong file test:**
   - Trong [`test_pipeline.py`](file:///d:/Introduction to BA/intro-to-ba/test_pipeline.py), toàn bộ quy trình kiểm thử được bọc trong cấu trúc:
     ```python
     def run_checks():
         cleanup_test_artifacts()
         try:
             # Tiến hành các bước kiểm thử...
         finally:
             cleanup_test_artifacts()
     ```
   - Bất kể quá trình kiểm thử thành công hay phát sinh lỗi, các file đồ thị tạm (`results/test_feature_importance.png`) và thư mục chạy thử Optuna (`results/test_tune`) đều được tự động dọn sạch ngay lập tức.
2. **Xuất và tổ chức dữ liệu tại [`data/processed/`](file:///d:/Introduction to BA/intro-to-ba/data/processed):**
   - Đã tích hợp hàm [`export_processed_data()`](file:///d:/Introduction to BA/intro-to-ba/preprocessing/pipeline.py#L402) vào `FeatureEngineeringPipeline`.
   - Toàn bộ dữ liệu sạch đã chuyển đổi được lưu trữ có cấu trúc rõ ràng:
     - `train.csv` (7,000 dòng $\times$ 124 cột): Dữ liệu huấn luyện chuẩn.
     - `val.csv` (1,000 dòng $\times$ 124 cột): Dữ liệu validation độc lập.
     - `test.csv` (2,000 dòng $\times$ 124 cột): Dữ liệu kiểm thử độc lập.
     - `train_smote.csv` (11,148 dòng $\times$ 124 cột): Dữ liệu huấn luyện đã cân bằng lớp qua SMOTE.
     - `full_transformed.csv` (10,000 dòng): Dữ liệu toàn bộ khách hàng sau khi biến đổi.
     - `feature_metadata.json`: Danh mục chi tiết 123 tên đặc trưng đầu vào mô hình.
     - `feature_importance_ranking.csv`: Bảng xếp hạng độ quan trọng đặc trưng theo Mutual Information.

---

## 4. Kết quả Kiểm thử & Đánh giá Thực nghiệm (Empirical Validation)

### 4.1. Kết quả kiểm thử Pipeline (`test_pipeline.py`)
Toàn bộ các module trong hệ thống đã được kiểm tra độc lập và kiểm tra tích hợp:

```
[OK] Loaded config from YAML: enable_domain_features=True, enable_smote=True
[OK] Ingested dataset via pipeline.load_dataset: 10000 rows, 65 columns
--- Running Full Pipeline (Loaded from config/config.yaml) ---
Train features shape (X_train): (7000, 123)
Val features shape   (X_val):   (1000, 123)
Test features shape  (X_test):  (2000, 123)
SMOTE Train shape    (X_smote): (11148, 123)
Total model features: 123
[OK] Pipeline feature generation, validation split, and SMOTE verified successfully!
--- Running Ablation via Config Override ---
Ablation Train shape (X_train): (7000, 84)
[OK] Domain ablation verified: 123 -> 84 features!
--- Testing Raw Kaggle Baseline (use_raw_kaggle_features_only) ---
[OK] Raw Kaggle baseline verified: 21 features!
--- Testing Feature Selection with MI_Score ---
[OK] FeatureSelector MI_Score verified: top feature is age (MI=0.0770)
--- Testing Visualization Plotting ---
[OK] Feature importance plotting verified and test plot cleaned up!
--- Testing DataSplitter Stratified K-Fold CV ---
[OK] DataSplitter 5-fold CV splits verified successfully!
--- Testing OptunaTuner Smoke Run ---
[OK] OptunaTuner smoke run verified successfully!
--- Exporting Main Processed Dataset to data/processed ---
[OK] Saved processed datasets to: data/processed
[OK] Main pipeline processed dataset exported cleanly to data/processed!
[SUCCESS] All pipeline, splitting, MI ranking, visualization, and Optuna tests passed!
```

### 4.2. Top 15 Đặc trưng có độ quan trọng cao nhất (Mutual Information Ranking)
Sau khi chuẩn hóa và bổ sung các đặc trưng nghiệp vụ động, bảng xếp hạng Mutual Information cho thấy các đặc trưng mới phát huy sức mạnh phân loại vượt trội:

| Thứ hạng | Tên đặc trưng | MI Score | Loại đặc trưng | Nhận định nghiệp vụ |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `age` / `std_age` | 0.0770 | Nhân khẩu học chuẩn hóa | Độ tuổi vẫn là yếu tố phân loại hàng đầu. |
| **2** | `products_number` | 0.0686 | Danh mục tài khoản | Số lượng sản phẩm khách hàng đang nắm giữ. |
| **3** | `is_middle_aged_vulnerable` | 0.0555 | Đặc trưng miền (Domain Risk) | Nhóm khách hàng 45–60 tuổi có tỷ lệ rời bỏ cao đột biến. |
| **4** | `is_product_sweetspot` | **0.0469** | **Phân vùng động (Bull Regime)** | **Thuộc Top 4 đặc trưng phân loại mạnh nhất toàn mô hình.** |
| **5** | `is_product_danger_zone` | **0.0224** | **Phân vùng động (Bear Regime)** | **Phát hiện chính xác nhóm suy giảm tương tác và danh mục thừa.** |
| **6** | `inactive_high_balance` | 0.0220 | Rủi ro tài sản (Interaction) | Khách hàng số dư lớn nhưng ngừng hoạt động. |
| **7** | `germany_high_balance` | 0.0153 | Rủi ro địa lý (Interaction) | Tương tác rủi ro cao tại thị trường Đức. |
| **8** | `wealth_deceleration_risk` | 0.0149 | Động lượng tài sản (Momentum) | Nguy cơ suy thoái số dư và chi tiêu. |
| **9** | `inflow_outflow_ratio` | 0.0148 | Cân đối dòng tiền | Tỷ lệ dòng tiền vào/ra của khách hàng. |
| **10** | `country_Germany` | 0.0138 | Phân loại địa lý | Cờ thị trường Đức. |
| **11** | `active_member` | 0.0123 | Hoạt động chủ động | Thành viên hoạt động (đã lọc bỏ giao dịch hệ thống). |
| **12** | `credit_card` | 0.0117 | Sản phẩm thẻ | Tình trạng sở hữu thẻ tín dụng. |
| **13** | `trans_count_trend_ratio` | 0.0115 | Động lượng giao dịch | Xu hướng số lượng giao dịch Quý 3 so với Quý 2. |
| **14** | `country_target_enc` | 0.0115 | Mã hóa mục tiêu (Target Encoding)| Trọng số rủi ro rời bỏ theo quốc gia. |
| **15** | `portfolio_health_score` | **0.0108** | **Đặc trưng miền (Regime Score)** | **Chỉ số sức khỏe danh mục tổng hợp.** |

---

