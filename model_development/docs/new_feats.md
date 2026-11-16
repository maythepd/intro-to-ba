# DANH MỤC ĐẶC TRƯNG MỚI ĐỀ XUẤT KHAI THÁC TỪ TÀI LIỆU WEEK 6 & WEEK 7
**Dự án:** Digital Bank Customer Churn Prediction & Value-Based Retention  
**Tài liệu:** `docs/new_feats.md`  
**Ngày lập:** 30/10/2026  
**Tác giả:** Antigravity Data Science & Business Analytics Team  
**Căn cứ tài liệu:** 
- `Week 6`: Banking Customer RFM Analytics & Segmentation (`Banking_RFM_Report.tex`, `rfm_pipeline.py`, `Banking_RFM_Analysis.ipynb`)
- `Week 7`: Banking Customer Lifetime Value (CLV) & Retention Decisioning (`clv_pipeline.py`, `build_pdf_report.py`, `Banking_CLV_Data_20000/DATA_DICTIONARY.md`)
- `intro-to-ba`: Báo cáo hiện trạng chuẩn hóa đặc trưng (`intro-to-ba/docs/features.md`, `synth_data_dict.md`, `etl.py`, `domain_features.py`)

---

## 1. Tóm tắt điều hành & Mục đích nghiên cứu (Executive Summary)

### 1.1. Bối cảnh bài toán
Dự án dự đoán khách hàng ngân hàng số rời bỏ (Customer Churn Prediction) hiện tại đang vận hành với 65 cột dữ liệu trích xuất từ 7 bảng sự kiện (`kaggle_bank_churn.csv`, `customer_context.csv`, `accounts.csv`, `savings_cycles.csv`, `complaints.csv`, `login.csv`, `transactions.csv`) và 33 đặc trưng nghiệp vụ động tại `domain_features.py`.

Tuy nhiên, mô hình hiện tại chủ yếu tập trung vào **các tỷ lệ sụt giảm vận hành ngắn hạn** (trans trend ratio, login collapse, friction index, products danger zone) mà **chưa tích hợp hai trụ cột phân tích kinh doanh cốt lõi trong ngân hàng bán lẻ**:
1. **Mô hình RFM chuẩn hóa (Recency, Frequency, Monetary):** Đã được nghiên cứu và kiểm chứng tại **Week 6**, cho phép phân nhóm hành vi khách hàng qua phân vị điểm số (Quintile scoring) và nhận diện sớm phân khúc **"At Risk"** (chiếm 23.91% tổng giá trị giao dịch của ngân hàng nhưng đang có dấu hiệu rời bỏ).
2. **Kinh tế học đơn vị & Giá trị vòng đời khách hàng (Unit Economics & CLV):** Đã được xây dựng bài bản tại **Week 7**, lượng hóa chính xác dòng đóng góp ròng ($m = \text{Fee} + \text{NII} - \text{Cost} - \text{ECL}$), mô hình định giá chiết khấu dòng tiền 3 năm ($CLV_3$), nhận diện **21.93% khách hàng thâm hụt giá trị (Negative-Value)**, và đo lường **độ co giãn giữ chân ($\Delta CLV$)** tránh bẫy trần giữ chân ($r \le 1.0$).

### 1.2. Mục đích tài liệu
Tài liệu này tổng hợp toàn bộ các phát hiện lý thuyết và thực nghiệm từ Week 6 và Week 7, xác định khoảng trống đặc trưng hiện tại, và thiết kế chi tiết **30 đặc trưng mới (chia thành 4 nhóm chiến lược)** có thể khai thác vào hệ thống Feature Engineering trước khi tiến hành lập trình triển khai (`implementation`).

```
                    ┌────────────────────────────────────────────────────────┐
                    │       HỆ THỐNG ĐẶC TRƯNG TÍCH HỢP TOÀN DIỆN MỚI        │
                    │         (CHURN PREDICTION + VALUE-BASED CRM)           │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
         ┌───────────────────┬─────────────────┴─────────────────┬───────────────────┐
         ▼                   ▼                                   ▼                   ▼
┌──────────────────┐┌──────────────────┐               ┌──────────────────┐┌──────────────────┐
│     PILLAR 1     ││     PILLAR 2     │               │     PILLAR 3     ││     PILLAR 4     │
│   RFM SCORING &  ││  CLV PROXIES &   │               │   RFM × CLV      ││   LIFECYCLE &    │
│    SEGMENTATION  ││  UNIT ECONOMICS  │               │    SYNERGY       ││  COHORT DYNAMICS │
│   (Từ Week 6)    ││   (Từ Week 7)    │               │  (Giao thoa mới) ││  (Chiều thời gian│
├──────────────────┤├──────────────────┤               ├──────────────────┤├──────────────────┤
│• R, F, M Quintile││• Proxy Fee Rev   │               │• Volume vs Profit││• Precise Tenure  │
│• RFM Composite   ││• Proxy CASA/NII  │               │  Divergence      │  (Months)         │
│• is_rfm_at_risk  ││• Servicing Cost  │               │• CASA Stability  │• Account Velocity │
│• is_rfm_champion ││• Proxy ECL Risk  │               │  Index           │• Inflow Volatility│
│• Digital Tx Share││• Net Contrib (m) │               │• Early Attrition │• Savings Maturity │
│• 90d Momentum    ││• Negative Margin │               │  Risk            │  Pressure         │
│• Spend Concentr. ││• 3-Year CLV Proxy│               │• Disengagement   │                   │
│• Wallet Share    ││• Marginal Yield  │               │  Lead Time Gap   │                   │
└──────────────────┘└──────────────────┘               └──────────────────┘└──────────────────┘
```

---

## 2. Tổng quan nghiên cứu & Bài học thực tiễn từ Week 6 và Week 7

### 2.1. Phân tích chuyên sâu từ Week 6: Banking Customer RFM Analytics
Tài liệu Week 6 triển khai phân tích trên 1,000 khách hàng và 43,711 giao dịch thô với các nguyên lý nghiệp vụ cốt lõi:

1. **Chính sách lọc giao dịch chi tiêu thực tế (Outflow Eligibility Policy):**
   - **Loại bỏ dòng tiền vào (Inflows: `TRANSFER_IN`, `CASH_DEPOSIT`):** Dòng tiền vào là hành động cấp thanh khoản (nhận lương, nạp tiền), không phải hành vi mua sắm/tiêu dùng. Tính cả tiền vào và tiền ra gây lỗi **nhân đôi dòng tiền (Double-Counting)** và phản ánh hành động thụ động của bên thứ ba (người trả lương) thay vì sự chủ động của khách hàng.
   - **Giữ lại 4 loại giao dịch nợ chủ động (Debit Outflows):** `CARD_PAYMENT` (thanh toán POS/online), `BILL_PAYMENT` (hóa đơn thiết yếu), `TRANSFER_OUT` (chuyển tiền thanh toán), và `ATM_WITHDRAWAL` (rút tiền mặt chi tiêu).
2. **Cơ chế xử lý khách hàng ngủ đông hoàn toàn ($F = 0$):**
   - Khách hàng không phát sinh giao dịch nào trong 12 tháng được gán $F=0$, $M=0$ và kiểm duyệt độ trễ $R = 366 \text{ ngày}$ (chiều dài cửa sổ $365 + 1$). Điều này bảo đảm tính tất định: các khách hàng này nhận điểm $R=1, F=1, M=1$ và tự động rơi vào phân khúc `Hibernating`.
3. **Phân vị theo xếp hạng (Rank-based Quintile Scoring):**
   - Do tần suất $F$ và độ trễ $R$ có nhiều giá trị trùng lặp (discrete ties), `pd.qcut` thông thường sẽ báo lỗi bin edges. Week 6 sử dụng `qcut(rank(method='first'))` kết hợp sắp xếp phụ theo `customer_id`, bảo đảm chia đều chính xác 20% khách hàng cho mỗi bậc điểm (1 đến 5).
4. **Phân khúc 5 tầng loại trừ lẫn nhau (Mutually Exclusive Taxonomy):**
   - `Champions` ($R \ge 4, F \ge 4, M \ge 4$): 16.7% khách hàng, chiếm **35.76% tổng giá trị giao dịch**.
   - `Loyal` ($R \ge 3, F \ge 3$): 26.5% khách hàng, chiếm **34.25% tổng giá trị giao dịch**.
   - `Potential` ($R \ge 3$ và không thuộc Champions/Loyal): 16.8% khách hàng, tài khoản mới onboarding cần tập quán hóa số.
   - `At Risk` ($R \le 2 \land [F \ge 3 \lor M \ge 3]$): 18.1% khách hàng, chiếm tới **23.91% tổng giá trị giao dịch** (7.87 tỷ VND). Nhóm này từng gắn bó sâu ($F=37.6$) nhưng độ trễ đã dãn ra trung bình 94.2 ngày không giao dịch $\rightarrow$ **Đây là nguồn tín hiệu rời bỏ (churn signal) mạnh nhất!**
   - `Hibernating` ($R \le 2, F \le 2, M \le 2$): 21.9% khách hàng ngủ đông.
5. **Cảnh báo cốt lõi: Doanh số giao dịch ($M$) không phải là Lợi nhuận ngân hàng:**
   - Giao dịch chuyển tiền liên ngân hàng 24/7 (P2P transfer) có $M$ rất lớn nhưng biên phí 0% và tốn chi phí hạ tầng.
   - Rút tiền ATM có $M$ lớn nhưng gây lỗ biên (chi phí tiếp quỹ tiền mặt, bảo dưỡng ATM, phí chuyển mạch liên ngân hàng).
   - Giao dịch thẻ (`CARD_PAYMENT`) tạo doanh thu phí interchange 1.5% - 2.5%, và số dư tiền gửi không kỳ hạn (CASA) tạo biên lãi ròng (NIM) lớn nhất cho ngân hàng.
6. **Các chỉ số mở rộng của Week 6:**
   - `digital_share`: Tỷ lệ giao dịch Mobile App / Web trên tổng giao dịch.
   - `momentum_90d`: Xung lượng giao dịch 90 ngày gần nhất so với cả năm: $\frac{F_{90d} \times 4}{F_{\text{year}}}$.
   - Khái niệm khách hàng "Lumpy Transactor" (khách hàng thực hiện 1 giao dịch tài sản lớn 100M rồi biến mất) vs "Habitual Spender" (khách hàng tiêu dùng thường nhật 50 giao dịch 100M).

---

### 2.2. Phân tích chuyên sâu từ Week 7: Banking Customer Lifetime Value (CLV)
Tài liệu Week 7 giải quyết bài toán định giá kinh tế tương lai và tối ưu hóa ngân sách giữ chân trên 20,000 khách hàng:

1. **Công thức Đóng góp ròng hàng năm (Annual Net Contribution $m$):**
   $$m = \text{fee\_revenue\_vnd} + \text{net\_interest\_income\_vnd} - \text{servicing\_cost\_vnd} - \text{expected\_credit\_loss\_vnd}$$
   - Khẳng định past acquisition costs (chi phí thu hút quá khứ) là sunk cost, không đưa vào định giá tương lai.
   - NII đã trừ chi phí vốn (funding cost). Chi phí vận hành (`servicing_cost`) và dự phòng rủi ro nợ xấu (`expected_credit_loss`) là các chi phí trừ trực tiếp.
2. **Định giá chiết khấu dòng tiền 3 năm ($CLV_3$):**
   $$CLV_3 = \sum_{t=1}^3 m \cdot \left(\frac{r}{1+d}\right)^t = m \cdot \left[ \frac{r}{1+d} + \left(\frac{r}{1+d}\right)^2 + \left(\frac{r}{1+d}\right)^3 \right]$$
   - Với lãi suất chiết khấu cơ sở $d = 10\%$ ($1.10$) và xác suất duy trì hàng năm $r \in [0, 1]$.
3. **Phát hiện nhóm khách hàng thâm hụt giá trị (Negative-Value Customers):**
   - Có tới **4,386 khách hàng (21.93%)** có $CLV_3 < 0$, gây tổn thất **-9.13 tỷ VND** cho ngân hàng do chi phí phục vụ trực tiếp tại quầy quá cao hoặc chi phí rủi ro tín dụng vượt quá doanh thu mang lại.
4. **Phân nhóm giá trị kinh tế (CLV Business Segments):**
   - `High Value / Stable` ($CLV \ge 5\text{M VND}, r \ge 0.75$): Chiếm 23.18% số lượng, nắm giữ **60.78% tổng CLV toàn ngân hàng** (45.71 tỷ VND).
   - `High Value / At Risk` ($CLV \ge 5\text{M VND}, r < 0.75$): Chiếm 13.73% số lượng, nắm giữ **25.51% tổng CLV**.
   - `Low Positive Value` ($0 < CLV < 5\text{M VND}$): Chiếm 37.37% số lượng.
   - `Zero Value` ($CLV = 0$): 0.30%.
   - `Negative Value` ($CLV < 0$): 21.93%.
5. **Độ nhạy giữ chân và Hiệu ứng trần giữ chân (Retention Ceiling Effect & $\Delta CLV$):**
   - Khi chạy tối ưu hóa ngân sách chiến dịch (Knapsack optimization với chi phí 100k/người, tăng $+3\%$ tỷ lệ giữ chân $r' = \min(1.0, r + 0.03)$), **danh sách Top 100 theo $\Delta CLV$ lệch tới 34% so với Top 100 theo CLV cơ sở**.
   - Lý do: Khách hàng có $r = 97.4\%$ chỉ có thể tăng tối đa $+2.6\%$ trước khi chạm trần 1.0 (trần giữ chân). Phân bổ ngân sách vào người chắc chắn ở lại là lãng phí ("paying for what you get for free"). Nhóm cần nhắm mục tiêu là nhóm có giá trị biên $\Delta CLV$ cao nhất!

---

## 3. Đối chiếu Hiện trạng & Phân tích Khoảng trống (Gap Analysis)

| Hạng mục / Khía cạnh | Hệ thống hiện tại (`intro-to-ba`) | Tài liệu Week 6 & Week 7 | Khoảng trống cần bổ sung (Gaps) |
| :--- | :--- | :--- | :--- |
| **Thước đo RFM** | Chỉ có các biến thô: `recency_days`, `total_transactions`, `total_outflow`. | Chuẩn hóa điểm phân vị `R_score`, `F_score`, `M_score` (1-5) và 5 phân khúc RFM loại trừ lẫn nhau. | Thiếu các bậc phân vị chuẩn hóa giúp mô hình phi tuyến tính và tuyến tính bắt được ngưỡng nhảy vọt; thiếu cờ phân khúc `is_rfm_at_risk`. |
| **Bản chất Chi tiêu** | Đã lọc dòng tiền ra ngoài ngân hàng (`is_ext_out`), nhưng chưa phân tách theo kênh thanh toán. | Phân định rạch ròi thẻ (`CARD_PAYMENT`), hóa đơn (`BILL_PAYMENT`), rút ATM, chuyển tiền P2P. | Cần tạo tỷ trọng chi tiêu thẻ (high margin), tỷ trọng rút ATM (cash economy, margin âm), và tỷ trọng chuyển tiền. |
| **Kinh tế học Đơn vị (Unit Economics)** | Chưa có mô hình hóa doanh thu, chi phí phục vụ, hay lợi nhuận ròng của khách hàng. | Công thức $m = \text{Fee} + \text{NII} - \text{Cost} - \text{ECL}$ phân rã chi tiết 4 thành phần tài chính. | **Hoàn toàn vắng bóng.** Cần xây dựng các biến Proxy: Doanh thu phí ước tính, Thu nhập lãi thuần ước tính từ CASA và Tiết kiệm, Chi phí phục vụ kênh, Chi phí rủi ro tín dụng. |
| **Giá trị Vòng đời (CLV)** | Chưa ước tính giá trị kinh tế tương lai; chỉ dự đoán nhãn nhị phân rời bỏ rời rạc. | Mô hình dòng tiền chiết khấu 3 năm $CLV_3$ kết hợp xác suất tồn tại $r^t$ và suất chiết khấu $d=10\%$. | Cần xây dựng biến proxy $\widehat{CLV}_3$ làm đặc trưng đầu vào để mô hình nắm bắt được tầm vóc kinh tế của khách hàng. |
| **Lệch pha Doanh số vs Lợi nhuận** | `total_outflow` được dùng làm thước đo chi tiêu chung. | Cảnh báo: $M$ cao có thể là P2P miễn phí, không tạo ra đồng lợi nhuận nào cho ngân hàng. | Cần đặc trưng đo độ lệch pha giữa khối lượng giao dịch và đóng góp lợi nhuận ròng ước tính (`volume_to_profit_divergence`). |
| **Xung lượng Ngắn hạn** | Có `trans_trend_ratio` (Q3 / Q2). | Có `momentum_90d` (chuẩn hóa tỷ lệ giao dịch quý gần nhất trên tổng giao dịch năm). | Cần bổ sung `momentum_90d` để đo tốc độ gia tốc/suy giảm giao dịch độc lập với quy mô tài khoản. |
| **Tập quán Kênh số** | Có `primary_device`, `desktop_login_ratio`. | Có `digital_share` = Tỷ lệ giao dịch App/Web trên tổng giao dịch. | Cần bổ sung `digital_tx_share` và đo độ sâu thói quen số (`app_to_trans_ratio` kết hợp kênh giao dịch). |
| **Khách hàng thâm hụt giá trị** | Chưa có khái niệm nhận diện khách hàng âm lợi nhuận. | 21.93% khách hàng mang lại $CLV < 0$ (phục vụ quầy đắt đỏ, rủi ro nợ xấu). | Cần cờ `is_negative_margin_customer` và tỷ lệ chi phí trên thu nhập (`cost_to_income_ratio`). |

---

## 4. Danh mục Chi tiết các Đặc trưng Mới Đề xuất Khai thác (Proposed Feature Catalog)

Dưới đây là 30 đặc trưng được thiết kế tỉ mỉ, chia thành 4 trụ cột nghiệp vụ:

### Trụ cột 1: Nhóm đặc trưng Điểm số & Phân khúc RFM (Khai thác từ Week 6)

#### 1. `rfm_recency_score` (Điểm phân vị Recency: 1 - 5)
- **Công thức:** 
  $$R\_score = \text{qcut}(\text{rank}(R, \text{method}='first', \text{ascending}=\text{False}), 5, \text{labels}=[1, 2, 3, 4, 5])$$
- **Nghiệp vụ:** Khách hàng giao dịch càng gần đây thì điểm càng cao (5 điểm = Top 20% gần nhất). Khách hàng không giao dịch có $R=366$ ngày tự động nhận 1 điểm.
- **Lợi ích mô hình:** Giúp tuyến tính hóa biến `recency_days` (vốn có phân phối lệch phải nghiêm trọng), giảm tác động của outliers đối với Logistic Regression.

#### 2. `rfm_frequency_score` (Điểm phân vị Tần suất: 1 - 5)
- **Công thức:** 
  $$F\_score = \text{qcut}(\text{rank}(F, \text{method}='first', \text{ascending}=\text{True}), 5, \text{labels}=[1, 2, 3, 4, 5])$$
- **Nghiệp vụ:** Số lượng giao dịch chủ động của người dùng càng nhiều thì điểm càng cao.

#### 3. `rfm_monetary_score` (Điểm phân vị Giá trị chi tiêu: 1 - 5)
- **Công thức:** 
  $$M\_score = \text{qcut}(\text{rank}(M, \text{method}='first', \text{ascending}=\text{True}), 5, \text{labels}=[1, 2, 3, 4, 5])$$
- **Nghiệp vụ:** Tổng chi tiêu tiền ra hợp lệ (`total_outflow`) càng lớn thì điểm càng cao.

#### 4. `rfm_composite_index` (Chỉ số RFM tổng hợp)
- **Công thức:**
  $$\text{rfm\_composite\_index} = 0.40 \times R\_score + 0.30 \times F\_score + 0.30 \times M\_score$$
- **Nghiệp vụ:** Đo lường sức sống toàn diện của tài khoản trên thang điểm liên tục từ 1.0 đến 5.0.

#### 5. `is_rfm_at_risk` (Cờ Phân khúc Nguy cơ rời bỏ - Tín hiệu Churn Trọng yếu)
- **Công thức:**
  $$\text{is\_rfm\_at\_risk} = \mathbb{I}\left( R\_score \le 2 \land (F\_score \ge 3 \lor M\_score \ge 3) \right)$$
- **Nghiệp vụ:** Nhận diện chính xác khách hàng từng có giá trị cao hoặc tần suất giao dịch dày đặc trong quá khứ nhưng đã ngưng hoạt động trong thời gian gần đây. Trong Week 6, nhóm này chiếm tới 23.91% giá trị giao dịch của ngân hàng. Đây là nhóm có xác suất rời bỏ cao nhất.

#### 6. `is_rfm_champion` (Cờ Phân khúc Khách hàng Tinh hoa)
- **Công thức:**
  $$\text{is\_rfm\_champion} = \mathbb{I}\left( R\_score \ge 4 \land F\_score \ge 4 \land M\_score \ge 4 \right)$$
- **Nghiệp vụ:** Khách hàng giao dịch thường xuyên, chi tiêu lớn và vừa mới giao dịch. Tỷ lệ rời bỏ tự nhiên của nhóm này cực thấp ($< 1.5\%$).

#### 7. `is_rfm_potential` (Cờ Phân khúc Tiềm năng Onboarding)
- **Công thức:**
  $$\text{is\_rfm\_potential} = \mathbb{I}\left( R\_score \ge 3 \land \neg(\text{Champion} \lor \text{Loyal}) \right)$$
- **Nghiệp vụ:** Khách hàng mới gia nhập hoặc có recency tốt nhưng tần suất và số tiền chưa tích lũy đủ. Cần phân biệt nhóm này với nhóm sắp rời bỏ để tránh phạt oan.

#### 8. `is_rfm_hibernating` (Cờ Phân khúc Ngủ đông)
- **Công thức:**
  $$\text{is\_rfm\_hibernating} = \mathbb{I}\left( R\_score \le 2 \land F\_score \le 2 \land M\_score \le 2 \right)$$
- **Nghiệp vụ:** Tài khoản gần như đã chết lâm sàng hoặc ngừng giao dịch hoàn toàn.

#### 9. `digital_channel_share` (Tỷ lệ Giao dịch qua Kênh số Mobile App)
- **Công thức:**
  $$\text{digital\_channel\_share} = \frac{\text{mobile\_tx\_count}}{\text{total\_transactions} + 1e-5}$$
- **Nghiệp vụ:** Trong Week 6, nhóm Champions có tỷ lệ số hóa 40.1% so với 37.1% ở nhóm Hibernating. Khách hàng gắn bó với ứng dụng ngân hàng di động có độ dính (stickiness) cao hơn và ít rời bỏ hơn.

#### 10. `atm_disintermediation_share` (Tỷ lệ Rút tiền mặt ATM)
- **Công thức:**
  $$\text{atm\_disintermediation\_share} = \frac{\text{cash\_withdrawal\_count}}{\text{total\_transactions} + 1e-5}$$
- **Nghiệp vụ:** Đo lường mức độ phụ thuộc vào tiền mặt vật lý. Khách hàng rút hết tiền mặt ra khỏi ngân hàng để tiêu dùng ngoài xã hội thường có mối liên kết lỏng lẻo với hệ sinh thái số của ngân hàng.

#### 11. `momentum_90d` (Xung lượng Giao dịch 90 Ngày gần nhất)
- **Công thức (chuẩn hóa theo cửa sổ 6 tháng):**
  $$\text{momentum\_90d} = \frac{\text{trans\_count\_3m} \times 2.0}{\text{total\_transactions} + 1.0}$$
- **Nghiệp vụ:** Nếu tốc độ giao dịch duy trì đều đặn thì chỉ số này xấp xỉ 1.0. Nếu khách hàng đang giảm dần hoạt động chuẩn bị rời bỏ, chỉ số này sụt giảm về tiệm cận 0 (trong Week 6, nhóm At Risk sụt xuống 0.10x). Nếu khách hàng đang tăng tốc gắn kết (onboarding), chỉ số sẽ $> 1.3x$.

#### 12. `spend_concentration_ratio` (Tỷ lệ Tập trung Chi tiêu - Nhận diện Lumpy Transactor)
- **Công thức:**
  $$\text{spend\_concentration\_ratio} = \frac{\text{avg\_ticket\_size}}{\text{total\_outflow} + 1.0}$$
- **Nghiệp vụ (Giải quyết Prompt 1 của Week 6):** Phân biệt giữa "Customer A" (chuyển một lệnh lớn 100M rồi thôi $\rightarrow$ tỷ lệ tiệm cận 1.0, rủi ro chuyển tiền mua nhà/đất rồi rút hết tài sản) và "Customer B" (tiêu dùng chia nhỏ 50 giao dịch sinh hoạt hàng ngày $\rightarrow$ tỷ lệ rất thấp $\approx 0.02$, gắn bó bền vững).

#### 13. `wallet_share_ratio` (Thị phần Ví Khách hàng)
- **Công thức:**
  $$\text{wallet\_share\_ratio} = \frac{\text{total\_outflow} \times 2.0}{\text{estimated\_salary} + 1.0}$$
  *(Nhân 2 để quy đổi chi tiêu 6 tháng thành quy mô chi tiêu cả năm).*
- **Nghiệp vụ:** Tỷ lệ chi tiêu qua tài khoản trên thu nhập ước tính. Khách hàng để ngân hàng xử lý phần lớn dòng tiền sinh hoạt của mình có chi phí chuyển đổi (switching cost) rất cao.

---

### Trụ cột 2: Nhóm đặc trưng Kinh tế học Đơn vị & Proxy CLV (Khai thác từ Week 7)

Vì tập dữ liệu `kaggle_bank_churn` không có sẵn báo cáo tài chính nội bộ, ta xây dựng **Mô hình Kinh tế học Đơn vị Ngân hàng Bán lẻ (Retail Banking Unit Economics Proxy Model)** dựa trên cấu trúc lý thuyết của Week 7:
$$m = \text{Fee Revenue} + \text{Net Interest Income} - \text{Servicing Cost} - \text{Expected Credit Loss}$$

#### 14. `proxy_fee_revenue` (Doanh thu Phí Ước tính Hàng năm)
- **Công thức:**
  $$\widehat{\text{Rev}}_{\text{fee}} = \left( 0.015 \times \text{payment\_amount} + \text{bank\_fee\_amount} + 1.0 \times \text{external\_transfer\_out\_count} \right) \times 2.0$$
- **Nghiệp vụ:** 
  - Trong ngân hàng, giao dịch mua hàng/thanh toán hóa đơn (`payment_amount`) mang lại cho ngân hàng phát hành thẻ khoản phí chiết khấu thương mại (Interchange Fee) chuẩn từ 1.5% đến 2.0%.
  - `bank_fee_amount`: Phí thường niên, phí duy trì tài khoản, phạt số dư tối thiểu do ngân hàng trực tiếp thu.
  - Phí chuyển tiền ra ngoài ngân hàng (~1,000 - 2,000 VND / lệnh).

#### 15. `proxy_net_interest_income` (Thu nhập Lãi Thuần Ước tính Hàng năm - NII)
- **Công thức:**
  $$\widehat{\text{NII}} = 0.055 \times \text{everyday\_balance} + 0.018 \times \text{savings\_balance}$$
- **Nghiệp vụ:**
  - **Tài khoản thanh toán (`everyday_balance` - CASA):** Ngân hàng chỉ trả lãi không kỳ hạn rất thấp (0.1% - 0.2%/năm) nhưng có thể đem cho vay với lãi suất 6% - 8%/năm $\rightarrow$ Biên lãi ròng (NIM Spread) cực lớn, ước tính **~5.5%/năm**. Khách hàng có số dư CASA lớn là "mỏ vàng" lợi nhuận của ngân hàng!
  - **Tài khoản tiết kiệm (`savings_balance`):** Ngân hàng phải trả lãi suất huy động có kỳ hạn (3% - 6%/năm), do đó biên lãi ròng sau khi trừ chi phí vốn chỉ còn khoảng **~1.8%/năm**.

#### 16. `proxy_servicing_cost` (Chi phí Phục vụ Ước tính Hàng năm)
- **Công thức:**
  $$\widehat{\text{Cost}}_{\text{servicing}} = \left( 3.5 \times \text{branch\_tx\_count} + 0.8 \times \text{atm\_tx\_count} + 0.05 \times \text{mobile\_tx\_count} + 15.0 \times \text{customer\_service\_complaints} + 25.0 \times \text{unresolved\_complaints\_count} + 5.0 \times \text{total\_accounts} \right) \times 2.0$$
- **Nghiệp vụ (Khớp chính sách Week 7):**
  - Giao dịch tại quầy chi nhánh tốn nhân sự, mặt bằng: ~$3.50/lượt.
  - Giao dịch ATM tốn phí vận hành máy, tiếp quỹ tiền mặt: ~$0.80/lượt.
  - Giao dịch số qua Mobile App chi phí biên gần bằng 0: ~$0.05/lượt.
  - Khiếu nại khách hàng (`complaints`) tiêu tốn chi phí tổng đài viên và bộ phận xử lý tranh chấp: ~$15.0/khiếu nại, khiếu nại chưa xử lý tốn thêm ~$25.0 do leo thang mâu thuẫn.

#### 17. `proxy_expected_credit_loss` (Chi phí Dự phòng Rủi ro Tín dụng Ước tính - ECL)
- **Công thức:**
  $$\widehat{\text{ECL}} = \mathbb{I}(\text{credit\_card} == 1) \times \max\left(0, \frac{680 - \text{credit\_score}}{100}\right)^2 \times (0.02 \times \text{estimated\_salary})$$
- **Nghiệp vụ:** Theo chuẩn mực kế toán IFRS 9 / Basel II, ngân hàng phải trích lập dự phòng tổn thất tín dụng dự kiến đối với khách hàng có thẻ tín dụng. Khách hàng có điểm tín dụng thấp ($< 620$) mang rủi ro vỡ nợ cao, bào mòn nghiêm trọng lợi nhuận của ngân hàng.

#### 18. `proxy_annual_net_contribution` (Ước tính Đóng góp Lợi nhuận Ròng Hàng năm $\hat{m}$)
- **Công thức:**
  $$\hat{m} = \left(\widehat{\text{Rev}}_{\text{fee}} + \widehat{\text{NII}}\right) - \left(\widehat{\text{Cost}}_{\text{servicing}} + \widehat{\text{ECL}}\right)$$
- **Nghiệp vụ:** Giá trị kinh tế ròng thực tế mà khách hàng mang lại trong 1 năm sau khi đã trừ toàn bộ chi phí vận hành kênh và dự phòng rủi ro.

#### 19. `is_negative_margin_customer` (Cờ Khách hàng Gây Lỗ - Value Destroyer)
- **Công thức:**
  $$\text{is\_negative\_margin\_customer} = \mathbb{I}(\hat{m} < 0)$$
- **Nghiệp vụ:** Trong Week 7, có tới 21.93% khách hàng làm hao hụt vốn ngân hàng (-9.13 tỷ VND) do chi phí phục vụ trực tiếp quá đắt đỏ trong khi số dư và phí giao dịch mang lại quá ít. Đặc trưng này giúp mô hình nhận diện nhóm khách hàng có thể ngân hàng chủ động không cần chi ngân sách đắt đỏ để giữ chân.

#### 20. `cost_to_income_ratio` (Hệ số Chi phí trên Thu nhập CIR Ngân hàng)
- **Công thức:**
  $$\text{cost\_to\_income\_ratio} = \frac{\widehat{\text{Cost}}_{\text{servicing}} + \widehat{\text{ECL}}}{\max\left(1.0, \widehat{\text{Rev}}_{\text{fee}} + \widehat{\text{NII}}\right)}$$
- **Nghiệp vụ:** Chỉ số hiệu quả tài chính kinh điển trong ngân hàng (CIR). CIR $> 1.0$ đồng nghĩa khách hàng tạo ra chi phí lớn hơn thu nhập.

#### 21. `fee_penalty_drag` (Tỷ lệ Phí Ngân hàng trên Chi tiêu)
- **Công thức:**
  $$\text{fee\_penalty\_drag} = \frac{\text{bank\_fee\_amount}}{\text{total\_outflow} + 1.0}$$
- **Nghiệp vụ:** Đo lường mức độ khách hàng bị ngân hàng trừ phí (phí quản lý, phí phạt). Khách hàng chịu gánh nặng phí lớn so với lượng tiền họ tiêu sẽ sinh tâm lý ức chế và có xu hướng đóng tài khoản chuyển sang ngân hàng miễn phí (Zero-fee banking).

#### 22. `proxy_retention_rate` (Ước tính Tỷ lệ Duy trì Hàng năm $\hat{r}$)
- **Công thức:**
  $$\hat{r} = \text{clip}\left( 0.65 + 0.15 \times \text{active\_member} + 0.10 \times \text{interaction\_velocity} + 0.05 \times \min\left(1.0, \frac{\text{tenure}}{5}\right) - 0.15 \times \text{is\_high\_friction}, 0.20, 0.98 \right)$$
- **Nghiệp vụ:** Ước tính xác suất khách hàng tiếp tục ở lại ngân hàng trong năm tiếp theo dựa trên trạng thái hoạt động, tốc độ tương tác, thâm niên và ma sát tranh chấp.

#### 23. `proxy_clv_3yr` (Ước tính Giá trị Vòng đời 3 Năm Chiết khấu $\widehat{CLV}_3$)
- **Công thức:**
  $$f = \frac{\hat{r}}{1 + 0.10} \implies \widehat{CLV}_3 = \hat{m} \times \left( f + f^2 + f^3 \right)$$
- **Nghiệp vụ:** Đại lượng định giá kinh tế tương lai của khách hàng. Khách hàng có CLV lớn là tài sản chiến lược của ngân hàng; mất nhóm này gây thiệt hại dòng tiền nghiêm trọng.

#### 24. `is_high_clv_at_risk` (Khách hàng Giá trị Cao Đang Nguy cơ Rời bỏ - Top Priority Retention)
- **Công thức:**
  $$\text{is\_high\_clv\_at\_risk} = \mathbb{I}\left( \widehat{CLV}_3 \ge \text{Median}(\widehat{CLV}_3) \land (\hat{r} < 0.75 \lor \text{is\_rfm\_at\_risk} == 1) \right)$$
- **Nghiệp vụ (Khớp phân khúc Task 3 Week 7):** Đây chính là phân khúc `High Value / At Risk`. Trong ngân hàng, việc phát hiện và cảnh báo sớm nhóm này có ý nghĩa sống còn đối với phòng CRM.

#### 25. `marginal_retention_yield` (Độ co giãn Tối ưu Giữ chân $\Delta CLV$ Proxy)
- **Công thức:**
  $$\text{marginal\_retention\_yield} = \max(0, \hat{m}) \times (1.0 - \hat{r})$$
- **Nghiệp vụ (Giải quyết bài học Task 4 Week 7):** Đo lường tiềm năng gia tăng giá trị khi can thiệp giữ chân. Những khách hàng đã có $\hat{r} \approx 1.0$ (chạm trần giữ chân) có yield bằng 0. Khách hàng có đóng góp $m$ lớn nhưng $\hat{r}$ đang suy giảm có yield cực đại, là đích ngắm hiệu quả nhất của các chiến dịch retention.

---

### Trụ cột 3: Nhóm đặc trưng Giao thoa RFM $\times$ CLV (Cross-Framework Synergy)

#### 26. `volume_to_profit_divergence` (Độ Lệch pha Khối lượng Giao dịch và Lợi nhuận Ròng)
- **Công thức:**
  $$\text{volume\_to\_profit\_divergence} = \frac{\log(1 + \text{total\_outflow})}{\log(1 + \max(0, \hat{m}) + 1.0)}$$
- **Nghiệp vụ (Hiện thực hóa Section 6.4 Week 6):** Bóc trần "bẫy khối lượng". Nếu chỉ số này rất cao, khách hàng tạo ra doanh số chuyển tiền ảo cực lớn nhưng đóng góp lợi nhuận cho ngân hàng bằng 0 $\rightarrow$ Tránh việc mô hình gán nhãn an toàn nhầm cho những khách hàng này.

#### 27. `casa_stability_ratio` (Tỷ lệ Ổn định Dòng tiền Tiền gửi Không kỳ hạn)
- **Công thức:**
  $$\text{casa\_stability\_ratio} = \frac{\text{everyday\_balance}}{\max(0, \text{balance}) + 1.0}$$
- **Nghiệp vụ:** 
  - `everyday_balance` đại diện cho tiền chi tiêu hoạt động sinh hoạt hàng ngày (CASA). Khách hàng để tỷ lệ CASA cao là khách hàng sử dụng ngân hàng làm tài khoản chính (Primary Bank Account), độ dính cực cao.
  - Ngược lại, nếu tỷ lệ CASA gần bằng 0 và toàn bộ tiền nằm ở `savings_balance`, khách hàng chỉ coi ngân hàng là nơi gửi tiết kiệm săn lãi suất (Yield Chaser), sẵn sàng rút sạch tiền sang ngân hàng khác khi hết kỳ hạn nếu đối thủ cạnh tranh có lãi suất cao hơn 0.2%.

#### 28. `savings_attrition_risk` (Rủi ro Chảy máu Tiền gửi Tiết kiệm)
- **Công thức:**
  $$\text{savings\_attrition\_risk} = \text{has\_savings\_account} \times (1 - \text{has\_auto\_renewal}) \times \text{avg\_early\_withdrawal\_rate} \times (\text{is\_high\_friction} + 0.5)$$
- **Nghiệp vụ:** Khách hàng gửi tiết kiệm nhưng **không cài đặt tự động gia hạn** (`renewal_type == manual`), có lãi suất phạt rút trước hạn cao, và vừa gặp sự cố khiếu nại dịch vụ $\rightarrow$ Xu hướng tất toán tiền gửi và chuyển tiền đi ngay khi đáo hạn là rất cao.

#### 29. `disengagement_gap` (Độ lệch Pha Rút lui: Recency Giao dịch vs Đăng nhập)
- **Công thức:**
  $$\text{disengagement\_gap} = \text{recency\_days} - \text{login\_recency\_days}$$
- **Nghiệp vụ:**
  - Nếu $\text{recency\_days} \gg \text{login\_recency\_days}$: Khách hàng vẫn đăng nhập xem ứng dụng gần đây nhưng đã ngừng thanh toán/chuyển tiền từ lâu $\rightarrow$ Dấu hiệu khách hàng đang ngưng sử dụng thẻ, chỉ vào kiểm tra số dư còn lại trước khi chuyển hẳn.
  - Cung cấp tín hiệu cảnh báo sớm trước khi khách hàng đóng hoàn toàn tài khoản.

---

### Trụ cột 4: Nhóm đặc trưng Vòng đời & Thâm niên Tài khoản (Lifecycle & Cohort Dynamics)

#### 30. `precise_tenure_months` (Thâm niên Quan hệ Khách hàng Chính xác theo Tháng)
- **Công thức:**
  $$\text{precise\_tenure\_months} = \frac{(\text{pd.Timestamp('2026-10-01')} - \text{relationship\_start\_date}).\text{dt.total\_seconds}()}{86400 \times 30.4375}$$
- **Nghiệp vụ:** Trong dữ liệu Kaggle gốc, cột `tenure` bị làm tròn thành số nguyên năm (0, 1, 2, ..., 10 năm), làm mất đi tính chi tiết của nhóm khách hàng mới gia nhập trong 12 tháng đầu đời (giai đoạn nhạy cảm nhất với churn). `precise_tenure_months` từ `customer_context.csv` cung cấp độ phân giải chính xác theo tháng.

#### 31. `account_opening_velocity` (Tốc độ Mở Tài khoản theo Thâm niên)
- **Công thức:**
  $$\text{account\_opening\_velocity} = \frac{\text{total\_accounts}}{\text{precise\_tenure\_months} + 1.0}$$
- **Nghiệp vụ:** Khách hàng mở nhiều tài khoản trong thời gian ngắn có thể là nhóm thợ săn khuyến mãi onboarding, dễ rời bỏ sau khi nhận xong quà tặng.

---

## 5. Bảng Ma trận Chi tiết Đặc trưng Mới Đề xuất (Proposed Feature Specifications Matrix)

Bảng tổng hợp 31 đặc trưng mới, phân định rõ công thức, kiểu dữ liệu, nguồn dữ liệu và độ ưu tiên triển khai:

| STT | Tên đặc trưng | Nhóm Pillar | Kiểu DL | Nguồn dữ liệu bảng | Giả thuyết Nghiệp vụ & Tác động Churn | Độ ưu tiên |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| 1 | `rfm_recency_score` | RFM Scoring | int (1-5) | `transactions`, `customer_context` | Điểm độ trễ phân vị. Điểm 1 cảnh báo nguy cơ rời bỏ cao. | **P1 (Cao)** |
| 2 | `rfm_frequency_score` | RFM Scoring | int (1-5) | `transactions` | Điểm tần suất phân vị. Tần suất thấp phản ánh thói quen lỏng lẻo. | **P1 (Cao)** |
| 3 | `rfm_monetary_score` | RFM Scoring | int (1-5) | `transactions` | Điểm chi tiêu phân vị. Đo quy mô tài chính của tài khoản. | **P1 (Cao)** |
| 4 | `rfm_composite_index` | RFM Scoring | float | Kết hợp $R, F, M$ | Thước đo sức sống tổng hợp trên thang 1.0 - 5.0. | **P1 (Cao)** |
| 5 | `is_rfm_at_risk` | RFM Segmentation | int (0/1) | $R, F, M \text{ scores}$ | Tín hiệu Churn mạnh nhất: Từng gắn bó cao nhưng đang ngừng dùng. | **P1 (Cao)** |
| 6 | `is_rfm_champion` | RFM Segmentation | int (0/1) | $R, F, M \text{ scores}$ | Khách hàng VIP cốt lõi, tỷ lệ rời bỏ cực thấp ($< 1.5\%$). | **P1 (Cao)** |
| 7 | `is_rfm_potential` | RFM Segmentation | int (0/1) | $R, F, M \text{ scores}$ | Khách mới onboarding, bảo vệ không bị phạt nhầm tín hiệu churn. | **P1 (Cao)** |
| 8 | `is_rfm_hibernating` | RFM Segmentation | int (0/1) | $R, F, M \text{ scores}$ | Tài khoản đã chết lâm sàng trong suốt cửa sổ quan sát. | **P2 (Trung bình)** |
| 9 | `digital_channel_share`| Channel Mix | float [0, 1]| `transactions` (channel) | Tỷ trọng giao dịch Mobile App; gắn bó số làm giảm churn. | **P1 (Cao)** |
| 10 | `atm_disintermediation_share` | Channel Mix | float [0, 1]| `transactions` (type) | Tỷ lệ rút tiền mặt ATM; gắn liền với rời bỏ hệ sinh thái số. | **P2 (Trung bình)** |
| 11 | `momentum_90d` | Momentum | float $\ge 0$ | `transactions` (datetime) | Xung lượng giao dịch 90 ngày; radar cảnh báo sớm sụt giảm hoạt động. | **P1 (Cao)** |
| 12 | `spend_concentration_ratio` | Behavioral Mix | float [0, 1]| `transactions` | Phân biệt Lumpy Transactor (1 lệnh lớn, dễ bay màu) vs Spender thường nhật. | **P2 (Trung bình)** |
| 13 | `wallet_share_ratio` | Wallet Share | float $\ge 0$ | `transactions`, `kaggle_churn` | Tỷ lệ chi tiêu trên thu nhập; tỷ lệ cao khó rời bỏ do chi phí chuyển đổi. | **P2 (Trung bình)** |
| 14 | `proxy_fee_revenue` | Unit Economics | float $\ge 0$ | `transactions` (payment, fee) | Doanh thu phí ước tính từ Interchange thẻ và phí dịch vụ. | **P1 (Cao)** |
| 15 | `proxy_net_interest_income` | Unit Economics | float $\ge 0$ | `accounts` (everyday, savings) | Thu nhập lãi thuần ước tính; CASA có NIM 5.5%, Tiết kiệm 1.8%. | **P1 (Cao)** |
| 16 | `proxy_servicing_cost` | Unit Economics | float $\ge 0$ | `transactions`, `complaints` | Chi phí phục vụ kênh (Quầy $3.5, ATM $0.8, App $0.05) và khiếu nại. | **P1 (Cao)** |
| 17 | `proxy_expected_credit_loss` | Unit Economics | float $\ge 0$ | `kaggle_churn` (credit score, card)| Dự phòng rủi ro nợ xấu thẻ tín dụng theo điểm tín dụng thấp. | **P2 (Trung bình)** |
| 18 | `proxy_annual_net_contribution`| Unit Economics | float (+/-) | Kết hợp 14, 15, 16, 17 | Đóng góp lợi nhuận ròng hàng năm $\hat{m}$ của khách hàng. | **P1 (Cao)** |
| 19 | `is_negative_margin_customer` | Value Drag | int (0/1) | $\hat{m} < 0$ | Khách hàng thâm hụt kinh tế (chiếm ~22% trong Week 7). | **P1 (Cao)** |
| 20 | `cost_to_income_ratio` | Efficiency | float $\ge 0$ | Proxy Cost / Proxy Revenue | Hệ số chi phí trên thu nhập; CIR $> 1.0$ là gánh nặng vận hành. | **P2 (Trung bình)** |
| 21 | `fee_penalty_drag` | Friction Risk | float [0, 1]| `transactions` (bank_fee) | Gánh nặng phí phạt; khách hàng bị trừ nhiều phí dễ ức chế rời bỏ. | **P2 (Trung bình)** |
| 22 | `proxy_retention_rate` | Actuarial Risk | float [0, 1]| `active_member`, recency, etc. | Ước tính xác suất duy trì tài khoản hàng năm $\hat{r}$. | **P2 (Trung bình)** |
| 23 | `proxy_clv_3yr` | CLV Valuation | float (+/-) | $\hat{m}, \hat{r}, d=0.10$ | Giá trị vòng đời 3 năm chiết khấu; đo lường quy mô tài sản khách hàng. | **P1 (Cao)** |
| 24 | `is_high_clv_at_risk` | CRM Targeting | int (0/1) | $\widehat{CLV}_3, \hat{r}$ | Phân khúc High Value / At Risk: Ưu tiên số 1 của CRM retention. | **P1 (Cao)** |
| 25 | `marginal_retention_yield` | Retention Uplift| float $\ge 0$ | $\hat{m} \times (1 - \hat{r})$ | Độ nhạy giữ chân; tránh bẫy trần giữ chân $r \le 1.0$. | **P2 (Trung bình)** |
| 26 | `volume_to_profit_divergence` | Synergy | float $\ge 0$ | $M$ vs $\hat{m}$ | Độ lệch pha doanh số vs lợi nhuận; bóc trần giao dịch P2P rỗng. | **P2 (Trung bình)** |
| 27 | `casa_stability_ratio` | Synergy | float [0, 1]| `accounts` (balances) | Tỷ lệ số dư CASA / Tổng số dư; khách hàng tài khoản chính vs thợ săn lãi. | **P1 (Cao)** |
| 28 | `savings_attrition_risk` | Synergy | float $\ge 0$ | `savings_cycles`, `accounts` | Rủi ro rút tiết kiệm: Tất toán thủ công + phạt rút sớm + khiếu nại. | **P2 (Trung bình)** |
| 29 | `disengagement_gap` | Early Radar | float (+/-) | `recency_days`, `login_recency` | Độ lệch giữa recency thanh toán và recency login; radar cảnh báo sớm. | **P1 (Cao)** |
| 30 | `precise_tenure_months`| Lifecycle | float $\ge 0$ | `customer_context` | Thâm niên tính chính xác theo tháng thay vì làm tròn năm. | **P1 (Cao)** |
| 31 | `account_opening_velocity` | Lifecycle | float $\ge 0$ | `accounts`, tenure months | Tốc độ mở tài khoản; nhận diện thợ săn khuyến mãi ngắn hạn. | **P3 (Thấp)** |

---

## 6. Kế hoạch Triển khai Kỹ thuật & Nguyên tắc Chống Rò rỉ Dữ liệu (Implementation Roadmap & Governance)

### 6.1. Phân chia trách nhiệm giữa các module trong Codebase
Để bảo đảm tính modular và không làm gãy các bài kiểm thử hiện có (`test_pipeline.py`, các bài ablation study), việc triển khai sẽ được chia thành 2 tầng:

1. **Tầng 1: Trích xuất thô tại `preprocessing/etl.py` (`BankDataETL`):**
   - Bổ sung đếm giao dịch theo từng kênh chủ động:
     - `branch_tx_count`: Số giao dịch có `channel == 'branch'`.
     - `atm_tx_count`: Số giao dịch có `channel == 'atm'`.
     - `mobile_tx_count`: Số giao dịch có `channel == 'mobile_app'`.
   - Tính toán `precise_tenure_months` từ `customer_context.csv`:
     ```python
     ctx["precise_tenure_months"] = (
         (self.snapshot_boundary - pd.to_datetime(ctx["relationship_start_date"])).dt.total_seconds() / (86400.0 * 30.4375)
     )
     ```
   - Xuất các cột thô này vào base dataframe mà không thực hiện phép chia tỷ lệ hoặc binning tại đây.

2. **Tầng 2: Kỹ nghệ đặc trưng nghiệp vụ tại `preprocessing/domain_features.py` (`DomainFeatureEngineer`):**
   - Mở rộng thêm 2 cờ cấu hình con trong `DomainFeatureEngineer`:
     - `enable_rfm_features: bool = True` (Bật/tắt Pillar 1: RFM Scoring & Segmentation).
     - `enable_clv_unit_economics: bool = True` (Bật/tắt Pillar 2 & 3: CLV, Kinh tế học đơn vị & Synergy).
   - Đảm bảo hỗ trợ đầy đủ cho cơ chế **Ablation Study** để đo lường độ tăng ích độc lập (marginal lift) của nhóm RFM và nhóm CLV.

### 6.2. Nguyên tắc Bất khả xâm phạm: Chống Rò rỉ Dữ liệu (Zero Data Leakage Governance)
1. **Phân vị RFM (Quintile Binning):**
   - Trong Week 6, phân vị được tính trên toàn bộ 1,000 khách hàng phục vụ mục đích phân khúc tĩnh.
   - Nhưng trong bài toán Machine Learning Churn Prediction (`intro-to-ba`), **tuyệt đối không được tính `pd.qcut` trên toàn bộ tập dữ liệu (Full Dataset)** vì sẽ gây rò rỉ phân phối tập Test/Validation vào tập Train!
   - **Giải pháp chuẩn khoa học dữ liệu:**
     - Tại hàm `DomainFeatureEngineer.fit(X_train, y_train)`: Tính toán các ngưỡng phân vị $P_{20}, P_{40}, P_{60}, P_{80}$ của $R, F, M$ **độc quyền trên tập `X_train`** và lưu vào thuộc tính `self.r_bins_`, `self.f_bins_`, `self.m_bins_`.
     - Tại hàm `DomainFeatureEngineer.transform(X)`: Áp dụng hàm `pd.cut` với các ngưỡng đã học cố định từ tập Train để gán điểm 1-5 cho `X_train`, `X_val`, và `X_test`.
2. **Ngưỡng CLV Median:**
   - Ngưỡng trung vị để phân nhóm `is_high_clv_at_risk` (`median_clv_`) được tính độc quyền trong `fit()` trên tập Train, sau đó áp cố định sang Validation và Test.
3. **Bảo vệ Khách hàng Mới (New Customers Onboarding):**
   - Duy trì logic chuẩn hóa từ `docs/features.md`: Khách hàng mới có thâm niên $\le 3$ tháng được gán cờ `is_rfm_potential = 1`, không gán nhầm vào nhóm `is_rfm_at_risk` hay phạt sụt giảm chi tiêu.

### 6.3. Dự kiến Đánh giá & Tiêu chuẩn Nghiệm thu Thực nghiệm
Sau khi tài liệu này được thông qua, quy trình triển khai code sẽ trải qua các bước kiểm thử khắt khe:
1. **Chạy kiểm thử toàn bộ hệ thống (`test_pipeline.py`):**
   - Xác nhận 100% các bài test (Full Pipeline, Ablation, MI Score Ranking, Stratified 5-Fold CV, Optuna smoke run) vượt qua không lỗi.
2. **Đánh giá Xếp hạng Mutual Information (`MI_Score`):**
   - Kỳ vọng: Các đặc trưng như `is_rfm_at_risk`, `proxy_annual_net_contribution`, `casa_stability_ratio`, `momentum_90d` sẽ tiến vào **Top 15 đặc trưng quan trọng nhất toàn mô hình**.
3. **Đánh giá Hiệu năng Mô hình (Model Benchmarking):**
   - So sánh trực tiếp AUC-ROC, PR-AUC, F1-Score trên Test set giữa Baseline hiện tại và Mô hình có thêm 31 đặc trưng mới trên cả 3 thuật toán:
     - **XGBoost Classifier**
     - **Random Forest Classifier**
     - **Logistic Regression (L1/SAGA)**
4. **Báo cáo Ablation Lift:**
   - Đo lường chính xác mức đóng góp AUC của `+enable_rfm_features` và `+enable_clv_unit_economics`.

---

## 7. Kết luận & Khuyến nghị Tiếp theo
Tài liệu này cung cấp cơ sở lý thuyết và danh mục đặc trưng toán học hoàn chỉnh, kết nối liền mạch giữa hai chuyên đề môn học (Week 6 - Phân khúc hành vi RFM và Week 7 - Giá trị kinh tế CLV) với bài toán Machine Learning Churn Prediction của Module 5.

**Hành động đề xuất:**
1. Lưu trữ và phê duyệt tài liệu tại `docs/new_feats.md`.
2. Tiến hành triển khai mã nguồn vào `intro-to-ba/preprocessing/etl.py` và `intro-to-ba/preprocessing/domain_features.py`.
3. Cập nhật file cấu hình `config/config.yaml` để quản lý các cờ bật/tắt nhóm đặc trưng mới.
4. Chạy lại benchmark và xuất báo cáo so sánh kết quả.

