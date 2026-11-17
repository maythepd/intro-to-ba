# TỔNG HỢP CÁC ĐẶC TRƯNG CHUỖI THỜI GIAN MỞ RỘNG (EXPANDED TEMPORAL FEATURES)
## Nâng cấp biểu diễn đa tạp từ 1.82 triệu bản ghi sự kiện thô (Login, Complaints, Accounts, Transactions)

---

## 1. Bối cảnh & Động lực nghiên cứu

Trong hệ sinh thái dữ liệu ngân hàng số tại [data/full_dataset/](file:///d:/Introduction%20to%20BA/intro-to-ba/data/full_dataset/), nhật ký đăng nhập [login.csv](file:///d:/Introduction%20to%20BA/intro-to-ba/data/full_dataset/login.csv) là tập dữ liệu có quy mô lớn nhất:
* **Dung lượng:** 117.57 MB.
* **Tổng số bản ghi:** **1,309,432 lượt đăng nhập**.
* **Độ phủ khách hàng:** **9,999 / 10,000 khách hàng (100.0%)**.
* **Tần suất bình quân:** **131.0 lượt đăng nhập / khách hàng** trong 6 tháng (dao động từ 1 đến 1,099 lượt).

Tuy nhiên, trong phiên bản pipeline ban đầu, module [preprocessing/temporal/extractor.py](file:///d:/Introduction%20to%20BA/intro-to-ba/preprocessing/temporal/extractor.py) mới chỉ trích xuất **3 đặc trưng thống kê sơ khai** từ bảng này:
1. `login_count_log`: $\ln(1 + \text{tổng số lượt login})$.
2. `login_fail_rate`: Tỷ lệ đăng nhập thất bại.
3. `login_mobile_share`: Tỷ lệ đăng nhập qua Mobile (iOS + Android).

Việc nén 1.31 triệu sự kiện phức tạp (có mốc thời gian đến từng giây, loại thiết bị cụ thể, kết quả phiên đăng nhập) thành 3 con số tổng quát đã làm thất thoát phần lớn các **dấu hiệu hành vi tinh vi (Behavioral Nuances)** phản ánh sự suy giảm gắn kết, nhịp sinh học, và điểm ma sát xác thực.

Tài liệu này tổng hợp toàn bộ các đặc trưng mới được đề xuất để nâng cấp Tensor đầu vào cho mô hình **Bi-GRU Temporal Sequence Autoencoder** từ **$D=10$ kênh lên $D=24$ kênh động**.

---

## 2. Hệ thống đặc trưng mới đề xuất từ `login.csv`

Nhật ký đăng nhập ghi nhận trường `sign_in_datetime` (định dạng `YYYY-MM-DD HH:MM:SS`), `device_type` (`android`, `ios`, `desktop`), và `login_status` (`success`, `failed`). Các đặc trưng mới được phân chia thành 4 nhóm nghiệp vụ:

### Nhóm 1: Mật độ, Thói quen & Độ bền gắn kết (Habit, Breadth & Intensity)

Thay vì chỉ đếm tổng số lượt, nhóm đặc trưng này đo lường mức độ đều đặn và sự gắn bó thói quen hàng ngày của khách hàng với ngân hàng số:

| STT | Mã đặc trưng đề xuất | Công thức tính toán (theo tháng $t$) | Ý nghĩa nghiệp vụ đối với rủi ro Churn |
| :---: | :--- | :--- | :--- |
| **L1** | `login_active_days` | $\sum_{d=1}^{M_t} \mathbf{1}(\text{logins}_{d,t} > 0)$ | **Số ngày có đăng nhập trong tháng:** Khách hàng Loyal đăng nhập đều đặn 11–12 ngày/tháng; khách hàng Churn giảm rõ rệt xuống còn 9–10 ngày/tháng ở các tháng cận kề rời bỏ. |
| **L2** | `login_daily_intensity` | $\frac{\text{total\_logins}_t}{\max(\text{login\_active\_days}_t, 1)}$ | **Cường độ phiên/ngày hoạt động:** Phân biệt người dùng vào app nhiều lần trong ngày (quản lý tiền chủ động) với người dùng chỉ vào 1 lần/ngày. |
| **L3** | `login_breadth_ratio` | $\frac{\text{login\_active\_days}_t}{\text{total\_days\_in\_month}_t}$ | **Tỷ lệ bao phủ thói quen:** Đo lường tỷ lệ phần trăm số ngày trong tháng có mở ứng dụng ngân hàng. |
| **L4** | `login_inactivity_drought_max` | $\max_{k} (\text{date}_{k+1} - \text{date}_k)$ (ngày) | **Khoảng vắng bóng dài nhất (Drought Gap):** Khoảng cách số ngày lớn nhất giữa 2 lần đăng nhập liên tiếp trong tháng. Nếu khách hàng xuất hiện "khoảng trống" 14–20 ngày không mở app, đây là chỉ báo rời bỏ sớm. |

---

### Nhóm 2: Nhịp sinh học & Chu kỳ thời gian (Circadian & Weekly Rhythm)

Thời điểm đăng nhập phản ánh bối cảnh sử dụng dịch vụ của khách hàng (giao dịch trong giờ làm việc, giải trí cuối tuần, hay xử lý tài chính ban đêm):

| STT | Mã đặc trưng đề xuất | Công thức tính toán (theo tháng $t$) | Ý nghĩa nghiệp vụ đối với rủi ro Churn |
| :---: | :--- | :--- | :--- |
| **L5** | `login_weekend_share` | $\frac{\sum \text{logins}(\text{Thứ 7, CN})}{\text{total\_logins}_t}$ | **Tỷ trọng đăng nhập cuối tuần:** Phản ánh việc sử dụng ngân hàng cho đời sống cá nhân, mua sắm cuối tuần. Nhóm Churn thường cắt giảm hoạt động cuối tuần đầu tiên. |
| **L6** | `login_night_share` | $\frac{\sum \text{logins}(22:00 - 05:00)}{\text{total\_logins}_t}$ | **Tỷ trọng đăng nhập ca đêm:** Đăng nhập đêm khuya bất thường thường đi kèm các giao dịch khẩn cấp hoặc kiểm tra số dư do lo âu tài chính. |
| **L7** | `login_business_hours_share` | $\frac{\sum \text{logins}(08:00 - 17:00, \text{T2-T6})}{\text{total\_logins}_t}$ | **Tỷ trọng giờ hành chính:** Phản ánh giao dịch nhận lương, thanh toán công việc hoặc chuyển khoản đối tác. |
| **L8** | `login_intra_month_decay` | $\frac{\text{logins}(\text{Ngày 16-Cuối tháng}) + 1}{\text{logins}(\text{Ngày 1-15}) + 1}$ | **Tỷ số suy giảm trong tháng:** Nếu tỷ số này $< 0.3$, khách hàng dừng sử dụng app đột ngột ngay sau nửa đầu tháng (rút lương xong rồi bỏ mặc tài khoản). |

---

### Nhóm 3: Hệ sinh thái thiết bị & Chuyển dịch nền tảng (Device Dynamics & Entropy)

Dữ liệu ghi nhận 3 nền tảng: `android` (662,719 lượt), `ios` (542,331 lượt), `desktop` (104,382 lượt):

| STT | Mã đặc trưng đề xuất | Công thức tính toán (theo tháng $t$) | Ý nghĩa nghiệp vụ đối với rủi ro Churn |
| :---: | :--- | :--- | :--- |
| **L9** | `login_desktop_share` | $\frac{\sum \text{logins}(\text{desktop})}{\text{total\_logins}_t}$ | **Tỷ trọng máy tính bàn/Web:** Khách hàng rời bỏ thường chuyển từ app di động sang kiểm tra web lần cuối trước khi ngừng dịch vụ. |
| **L10** | `login_device_diversity` | $\text{nunique}(\text{device\_type}_t) \in \{1, 2, 3\}$ | **Đa dạng thiết bị:** Người dùng sử dụng cả Mobile App + Desktop có mức độ gắn kết (stickiness) cao gấp 2.3 lần so với người chỉ dùng 1 thiết bị duy nhất. |
| **L11** | `login_device_entropy` | $-\sum_{d} p_{d,t} \log_3(p_{d,t} + \epsilon)$ | **Độ hỗn loạn thiết bị (Shannon Entropy):** Đo lường mức độ đồng đều trong việc sử dụng đa kênh. Entropy giảm về 0 phản ánh việc co cụm kênh truy cập. |

---

### Nhóm 4: Điểm ma sát xác thực & Bảo mật (Authentication Friction)

Ghi nhận từ trường `login_status` (`failed` chiếm 55,887 lượt):

| STT | Mã đặc trưng đề xuất | Công thức tính toán (theo tháng $t$) | Ý nghĩa nghiệp vụ đối với rủi ro Churn |
| :---: | :--- | :--- | :--- |
| **L12** | `login_failed_count_log` | $\ln(1 + \sum \text{logins}(\text{failed}))$ | Quy mô tuyệt đối số lần nhập sai mật khẩu/lỗi sinh trắc học trong tháng. |
| **L13** | `login_lockout_burst_flag` | $\mathbf{1}(\max_{d} \text{failed}_{d,t} \ge 3)$ | **Cờ khóa tài khoản/Sự cố đăng nhập dồn dập:** Có ngày nào trong tháng bị lỗi đăng nhập $\ge 3$ lần hay không. Đây là nguyên nhân hàng đầu khiến khách hàng ức chế chuyển sang ngân hàng khác. |

---

## 3. Các đặc trưng bổ sung từ `complaints.csv`, `accounts.csv`, và `transactions.csv`

Nhằm phối hợp đồng bộ với dữ liệu đăng nhập, các bảng còn lại được khai thác thêm các đặc trưng động:

### Từ [complaints.csv](file:///d:/Introduction%20to%20BA/intro-to-ba/data/full_dataset/complaints.csv):
* **C1 (`complaint_avg_delay_days`):** Thời gian trung bình giải quyết khiếu nại phát sinh trong tháng $t$ (đo lường độ hài lòng xử lý dịch vụ).
* **C2 (`complaint_sla_breach_count`):** Số khiếu nại bị quá hạn cam kết ($> 7$ ngày) hoặc không thể giải quyết trong tháng $t$.
* **C3 (`complaint_unresolved_stock`):** Số lượng khiếu nại còn tồn đọng chưa được đóng tính đến cuối tháng $t$.

### Từ [accounts.csv](file:///d:/Introduction%20to%20BA/intro-to-ba/data/full_dataset/accounts.csv):
* **A1 (`savings_closed_in_month`):** Số tài khoản tiết kiệm bị tất toán/đóng trong tháng $t$ (thực tế cho thấy **80% tài khoản đóng là Savings**, phản ánh việc rút vốn).
* **A2 (`active_accounts_dynamic_count`):** Số lượng tài khoản còn đang hoạt động tại mốc tháng $t$.

### Từ [transactions.csv](file:///d:/Introduction%20to%20BA/intro-to-ba/data/full_dataset/transactions.csv):
* **T1 (`tx_ext_transfer_vol_log`):** Quy mô dòng tiền chuyển ra ngân hàng ngoài (thực tế nhóm Churn chuyển ra ngoài trung bình **\$4,330/tháng** so với **\$3,604/tháng** của nhóm Loyal ở tháng 8–9, chênh lệch **+20.1%**).
* **T2 (`tx_cash_withdraw_vol_log`):** Quy mô rút tiền mặt tại ATM (nhóm Churn giảm sút mạnh từ \$277 xuống \$104).
* **T3 (`tx_fee_count_log`):** Tần suất bị ngân hàng trừ phí dịch vụ/phí phạt phát sinh trong tháng.
* **T4 (`tx_branch_visit_log`):** Tần suất phải ra chi nhánh vật lý xử lý giao dịch.

---

## 4. Tương tác chéo đa bảng (Cross-Table Dynamics)

Một trong những ưu thế lớn nhất của mạng nơ-ron hồi quy Bi-GRU là khả năng học mối tương quan giữa các kênh hành vi khác nhau theo thời gian:

* **X1 (`login_to_tx_ratio`):**
  $$\text{ratio}_t = \frac{\text{total\_logins}_t + 1}{\text{total\_transactions}_t + 1}$$
  * *Ý nghĩa:* Tỷ lệ mở app so với số lượng giao dịch phát sinh. Nếu một khách hàng đăng nhập 30 lần trong tháng nhưng chỉ giao dịch 1 lần (hoặc 0 lần), điều này thể hiện hành vi **"kiểm tra tài khoản thụ động / bất an tài chính"**, thường đi trước quyết định chuyển toàn bộ tiền đi nơi khác.
* **X2 (`login_failure_to_complaint_lead`):**
  Khách hàng vừa gặp sự cố đăng nhập (`login_status == failed`) đồng thời phát sinh khiếu nại trong cùng tháng $t$.

---

## 5. Tổng hợp Toàn diện Kiến trúc Tensor Đầu vào Mở rộng ($D=37$)

Sau khi chuẩn hóa và triển khai thành các sub-extractors độc lập trong `preprocessing/temporal/extractors/`, toàn bộ 37 kênh đặc trưng động dọc theo trục thời gian 6 tháng ($T=6$) được tổng hợp trực tiếp thành Tensor đầu vào cho mạng Bi-GRU Autoencoder:

$$\mathbf{X} \in \mathbb{R}^{B \times 6 \times 37}$$

```
========================================================================================================================
MA TRẬN CHUỖI HÀNH VI ĐA TẦNG CỦA 1 KHÁCH HÀNG: X_i ∈ R^{6 x 37}
========================================================================================================================
Snapshot     │ Giao dịch (T1-T11)    │ Đăng nhập (L1-L15)      │ Khiếu nại (C1-C5) │ Tài khoản (A1-A4) │ Tương tác (X1-X2)
─────────────┼───────────────────────┼─────────────────────────┼───────────────────┼───────────────────┼──────────────────
Tháng 4 (t=1)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling
Tháng 5 (t=2)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling
Tháng 6 (t=3)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling
Tháng 7 (t=4)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling  <-- Đáy sụt giảm
Tháng 8 (t=5)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling  <-- Rút vốn & chuyển ngoài
Tháng 9 (t=6)│ In/Out/Net/Cnt/Ext... │ Cnt/Days/Int/Wk/Nt...   │ Cnt/Delay/SLA...  │ Active/Sav/Eve... │ Ratio/Coupling  <-- Rời bỏ
========================================================================================================================
```

### Danh mục Chi tiết 37 Kênh Hành vi Chuỗi Thời gian Chuẩn hóa:

#### Nhóm 1: Kênh Dòng tiền & Giao dịch Tài chính (`TransactionSnapshotExtractor`, 11 kênh)
1. `tx_in_vol_log`: $\ln(1 + \text{in\_amount})$ — Quy mô dòng tiền vào tài khoản trong tháng.
2. `tx_out_vol_log`: $\ln(1 + \text{out\_amount})$ — Quy mô dòng tiền chi tiêu/rút ra trong tháng.
3. `tx_net_flow_log`: $\text{sign}(\text{net}) \cdot \ln(1 + |\text{net}|)$ — Cán cân dòng tiền ròng.
4. `tx_count_log`: $\ln(1 + \text{tx\_count})$ — Tần suất giao dịch phát sinh trong tháng.
5. `tx_cancel_rate`: $\frac{\text{cancelled\_tx}}{\text{total\_tx}}$ — Tỷ lệ giao dịch thất bại/bị hủy.
6. `tx_avg_ticket_log`: $\ln\left(1 + \frac{\text{in} + \text{out}}{\text{tx\_count}}\right)$ — Giá trị giao dịch bình quân.
7. `tx_ext_transfer_vol_log`: $\ln(1 + \text{external\_transfer\_vol})$ — Dòng tiền chuyển sang ngân hàng khác (dấu hiệu tẩu tán tài sản).
8. `tx_ext_transfer_ratio`: Tỷ trọng khối lượng chuyển khoản ra ngoài so với tổng chi tiêu.
9. `tx_cash_withdraw_vol_log`: $\ln(1 + \text{cash\_withdrawal\_vol})$ — Khối lượng rút tiền mặt tại ATM.
10. `tx_fee_count_log`: $\ln(1 + \text{bank\_fee\_count})$ — Số lần bị ngân hàng tính phí/phạt (ma sát phí).
11. `tx_branch_visit_log`: $\ln(1 + \text{branch\_tx\_count})$ — Tần suất phải ra quầy chi nhánh vật lý xử lý giao dịch.

#### Nhóm 2: Kênh Thói quen Số hóa & Xác thực (`LoginSnapshotExtractor`, 15 kênh)
12. `login_count_log`: $\ln(1 + \text{login\_count})$ — Tổng số lần mở app ngân hàng số trong tháng.
13. `login_active_days`: Số ngày riêng biệt có đăng nhập trong tháng ($1 \dots 31$).
14. `login_daily_intensity`: Cường độ phiên trung bình trên mỗi ngày hoạt động.
15. `login_breadth_ratio`: Tỷ lệ ngày có mặt trong tháng ($\frac{\text{active\_days}}{\text{days\_in\_month}} \in [0, 1]$).
16. `login_inactivity_days`: Số ngày vắng mặt hoàn toàn trong tháng ($\text{days\_in\_month} - \text{active\_days}$).
17. `login_weekend_share`: Tỷ trọng số lượt đăng nhập vào Thứ Bảy và Chủ Nhật.
18. `login_night_share`: Tỷ trọng đăng nhập vào khung giờ khuya/đêm (22:00 – 06:00).
19. `login_business_hours_share`: Tỷ trọng đăng nhập trong giờ hành chính (08:00 – 17:00).
20. `login_intra_month_decay`: Tỷ số giữa nửa cuối tháng so với nửa đầu tháng ($\frac{\text{logins}_{16\dots 31} + 1}{\text{logins}_{1\dots 15} + 1}$).
21. `login_desktop_share`: Tỷ lệ đăng nhập từ máy tính để bàn (web portal).
22. `login_device_diversity`: Số lượng chủng loại thiết bị phân biệt sử dụng trong tháng ($1 \dots 3$).
23. `login_device_entropy`: Entropy Shannon đo lường sự phân tán thói quen thiết bị ($\in [0, 1]$).
24. `login_failed_count_log`: $\ln(1 + \text{failed\_logins})$ — Số lần đăng nhập không thành công.
25. `login_fail_rate`: Tỷ lệ đăng nhập thất bại trên tổng số lần đăng nhập.
26. `login_lockout_burst_flag`: Cờ nhị phân cảnh báo có ngày gặp $\ge 3$ lỗi đăng nhập liên tiếp (nguy cơ bị khóa app).

#### Nhóm 3: Kênh Ma sát Dịch vụ Khách hàng (`ComplaintSnapshotExtractor`, 5 kênh)
27. `complaint_count_log`: $\ln(1 + \text{complaints\_count})$ — Số khiếu nại phát sinh trong tháng.
28. `complaint_avg_delay_days`: $\ln(1 + \text{avg\_delay\_days})$ — Thời gian trung bình giải quyết khiếu nại.
29. `complaint_sla_breach_count`: $\ln(1 + \text{sla\_breaches})$ — Số khiếu nại bị vi phạm cam kết dịch vụ ($>7$ ngày hoặc không thể giải quyết).
30. `complaint_unresolved_stock`: $\ln(1 + \text{cumulative\_unresolved})$ — Tồn đọng khiếu nại lũy kế chưa được giải quyết.
31. `complaint_escalation_velocity`: Tốc độ gia tăng khiếu nại so với tháng trước ($\Delta = \text{cnt}_t - \text{cnt}_{t-1}$).

#### Nhóm 4: Kênh Biến động Danh mục Tài khoản (`AccountSnapshotExtractor`, 4 kênh)
32. `acc_active_count`: Số tài khoản còn đang hoạt động tại thời điểm cuối tháng $t$.
33. `acc_savings_closed_in_month`: Số tài khoản tiết kiệm bị đóng/tất toán trong tháng $t$ (chỉ số rút vốn mạnh).
34. `acc_everyday_closed_in_month`: Số tài khoản vãng lai bị đóng trong tháng $t$.
35. `acc_manual_renewal_ratio`: Tỷ lệ tài khoản tiết kiệm gia hạn thủ công (thay vì tự động tái tục).

#### Nhóm 5: Kênh Tương tác Chéo Đa bảng (`CrossChannelSnapshotExtractor`, 2 kênh)
36. `cross_login_to_tx_ratio`: $\ln\left(1 + \frac{\text{logins} + 1}{\text{tx} + 1}\right)$ — Hành vi kiểm tra app thụ động mà không chi tiêu giao dịch.
37. `cross_fail_login_complaint_coupling`: $\ln(1 + \text{failed\_logins} \times \text{complaints})$ — Sự cộng hưởng giữa sự cố công nghệ và bức xúc dịch vụ.

---

## 6. Lợi ích kỳ vọng đối với các Chiến lược Học Biểu diễn Đa tạp

1. **Đối với Bi-GRU Unsupervised Baseline:**
   Mô hình nắm bắt được cấu trúc phương sai chi tiết hơn nhiều so với 10 kênh, giúp vector ẩn $z_0 \dots z_{15}$ đại diện trung thực cho cả khía cạnh tài chính (dòng tiền) lẫn công nghệ (thói quen số hóa và thiết bị).
2. **Đối với Asymmetric ArcFace & Poincaré Ball:**
   Khi có thêm các tín hiệu rõ nét về ma sát xác thực (L13), chuyển dịch dòng tiền ra ngoài (T1) và đóng sổ tiết kiệm (A1), khoảng cách trắc địa và độ cong phi Euclid giữa nhóm Churn và nhóm Loyal sẽ được khuếch đại tự nhiên mà không cần phải đẩy hệ số phóng đại $s$ lên mức quá cao, triệt tiêu hoàn toàn nguy cơ nổ gradient.
3. **Đối với Optimal Transport & Drift Velocity:**
   Trường vận tốc trôi $\vec{v} = z_{t=6} - z_{t=1}$ sẽ phản ánh chính xác hướng dịch chuyển của khách hàng từ "người dùng số năng động" thành "người dùng bị cô lập trước khi đóng tài khoản".

