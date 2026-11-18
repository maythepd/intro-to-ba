# Kế hoạch hoàn thiện Module 7 — Model Monitoring

Ngày lập: 17/11/2026. Phạm vi: API FastAPI và dashboard React hiện có của BaBank.
Đây là kế hoạch ban đầu, kèm cập nhật triển khai bên dưới. Hướng dẫn vận hành hiện tại nằm trong `docs/module-7-monitoring.md`.

Cập nhật sau khi chuyển mặc định sang Module 5: API đã có adapter checkpoint, preprocessing riêng và manifest tái lập đúng metrics nghiên cứu. Các hạng mục monitoring, lưu lịch sử và trigger bên dưới vẫn cần thực hiện.

## Tiến độ ngày 18/11/2026

- Đã triển khai baseline, PostgreSQL/SQLite store, logging single/batch, delayed labels, drift/performance rules, API, dashboard và HTML/JSON report.
- Supabase project riêng `exifwzbacchvvtzckxcw` đã có schema private/RLS; ba kịch bản replay đã được ghi, tính và đọc lại sau khi kết nối lại.
- Đã có notebook nộp bài, kiểm thử local và workflow CI có PostgreSQL test service.
- Theo lựa chọn của người dùng, workflow chỉ gọi API đã được cài riêng lên `main` của fork tại commit `db81e0d`, có GitHub Secret được mã hóa; code ứng dụng vẫn phát triển trên nhánh Module 7 và hướng về `demo`.
- Chưa xác nhận lượt deploy Render của Module 7. Đang chờ `RENDER_API_KEY` để cấu hình/redeploy bằng API; Computer Use dừng do không xác định được URL trình duyệt Windows đủ tin cậy, nên chưa kiểm tra trực quan desktop/mobile.

## 1. Mục tiêu theo đề bài

Căn cứ Module 7 và Evaluation Focus trong [thông báo project](../../Business_Analytics_Project_Announcement.pdf):

1. So sánh phân phối dữ liệu đầu vào mới với dữ liệu huấn luyện để phát hiện data drift.
2. Theo dõi chất lượng dự đoán theo thời gian khi có nhãn thực tế.
3. Có dashboard hoặc report monitoring và các điều kiện kích hoạt xem xét tái huấn luyện.

Kết quả cần nộp: trang `/monitoring` trên live link, báo cáo có thể tải về, cấu hình ngưỡng, kiểm thử và hướng dẫn demo tái lập được.

## 2. Hiện trạng khi lập kế hoạch và các phụ thuộc

| Thành phần | Hiện trạng | Việc cần làm |
| --- | --- | --- |
| `api/main.py` | `/api/monitoring` chỉ trả trạng thái model và số inference trong RAM; drift/performance đều false | Thu thập dữ liệu, tính chỉ số thật, trả lịch sử và trạng thái có giải thích |
| `frontend/src/routes/monitoring.tsx` | Biểu đồ `ILLUS` tạo số minh họa | Thay bằng dữ liệu monitoring API, có trạng thái thiếu mẫu/thiếu nhãn |
| `api/prepare_module5.py`, `api/module5.py` | Nạp checkpoint Module 5, dựng lại preprocessing trên split gốc và tái lập 6 metrics; đã có hash/manifest | Dùng manifest hiện có làm registry/baseline cho monitoring |
| `/api/metrics` | Đọc báo cáo Module 5; thêm checkpoint hash và trạng thái tái lập | Phân biệt baseline nghiên cứu với metrics của các window có nhãn mới |
| Single/batch scoring | Chưa có lịch sử đầu vào và prediction bền vững | Ghi sự kiện dự đoán, nhận diện nguồn, tránh nhân đôi cached batch |
| Render | Build đã được cấu hình chuẩn bị preprocessing Module 5 và clustering artifacts | Xác nhận phiên bản live sau redeploy; lưu lịch sử ngoài filesystem của web service |

Thay đổi cục bộ sẵn có trong `api/main.py` và `frontend/package-lock.json` cần được giữ khi bắt đầu nhánh `feature/module7-monitoring`.

## 3. Kiến trúc và quyết định triển khai

Luồng chính:

`Export model + baseline -> Predict API -> Event store -> Monitoring runner -> Reports/alerts -> Monitoring API -> Dashboard`

- Dùng Postgres qua `DATABASE_URL` để lưu lịch sử. Chốt nhà cung cấp còn hạn sử dụng tới ngày chấm bài trước khi deploy. SQLite chỉ dùng cho phát triển/kiểm thử cục bộ.
- Render free có filesystem tạm; dữ liệu ghi lúc chạy có thể mất sau restart, redeploy hoặc spin down. Render Postgres free hết hạn sau 30 ngày. Đây là ràng buộc cần tính vào lựa chọn DB và lịch demo.
- Tính drift bằng NumPy/SciPy đã có trong repo; xuất JSON và HTML từ cùng kết quả. Evidently là lựa chọn mở rộng nếu cần báo cáo phong phú hơn; đề bài đưa Evidently làm ví dụ, không bắt buộc thư viện này.
- Chạy monitoring theo ngày bằng GitHub Actions gọi endpoint vận hành có token; có `workflow_dispatch` để chạy ngay khi demo. Không phụ thuộc bộ hẹn giờ trong tiến trình web đang chạy.
- Tách nguồn `live`, `snapshot_batch`, `study_replay` và từng `scenario_id`. Batch chấm lại cùng snapshot không được trình bày thành các đợt khách hàng mới.
- Chỉ công khai kết quả tổng hợp. Thao tác nhập nhãn, tạo demo và chạy job dùng credential phía server; credential không nằm trong biến `VITE_*`.

## 4. Các bước thực hiện

### Bước 1 — Đóng gói model, baseline và tập đánh giá

**Ước lượng: 0,5–1 ngày.**

- Giữ nguyên split gốc của checkpoint Module 5, xấp xỉ 80% train/10% validation/10% test; manifest ghi kích thước và seed chính xác. Không đổi split rồi lắp preprocessing mới vào checkpoint đã train.
- Dùng preprocessing và metrics đã được `api.prepare_module5` kiểm chứng. Replay dùng các khách hàng ngoài tập train, gắn nguồn/scenario rõ ràng. Khi huấn luyện một phiên bản mới, giữ tập đánh giá/replay ngoài dữ liệu fit và chọn model.
- Xuất baseline phân phối từ các feature thô của train trước SMOTE và mã hóa. Loại ID, nhãn và timestamp vận hành khỏi danh sách feature drift.
- Kế thừa checkpoint/preprocessor hash, hash dữ liệu, schema/features, split manifest, seed, threshold và metrics đã có; bổ sung ID registry và reference cho monitoring.
- Baseline chất lượng dùng đúng checkpoint và đường scoring của API. Chỉ gắn baseline cho phiên bản live sau khi `/health` xác nhận checkpoint hash và `study_reproduced=true`.
- Segmentation có version và reference riêng theo model/preprocessor đang deploy. Theo dõi tỷ lệ cluster với model được giữ cố định; ghi rõ phép gán hierarchical ngoài mẫu fit đang là xấp xỉ.

**Hoàn thành khi:** có thể truy ngược mỗi report tới đúng model, dữ liệu, reference và tập đánh giá; các split không giao nhau theo customer ID.

### Bước 2 — Ghi lịch sử dự đoán và kết quả thực tế

**Ước lượng: 0,5–1 ngày.**

- Các nhóm dữ liệu lưu trữ: `model_versions`, `prediction_events`, `outcomes`, `monitoring_runs`, `monitoring_alerts`.
- Prediction event gồm ID sự kiện, thời gian, model/segment version, nguồn, feature đầu vào cần theo dõi, probability, predicted class, segment nếu có, latency và trạng thái xử lý.
- Outcome gắn với đúng prediction, có nhãn churn, nguồn nhãn, thời điểm quan sát và horizon kết thúc. Chỉ dùng nhãn đã đủ thời gian quan sát cho quality metrics.
- Kết quả inference không đưa nhãn thật vào preprocessing. Nhãn chỉ được nối ở bước đánh giá.
- Ghi nhận single scoring và batch mới thành công; cached batch không sinh thêm một tập sự kiện giống hệt. Chạy lại job cùng model/window/config không tạo bản ghi trùng.
- Bộ đếm lưu mọi request; tập dùng tính drift/performance xử lý khách hàng lặp trong cùng window theo quy tắc cố định, ví dụ lấy lần dự đoán cuối cùng.
- Không tự lấy nhãn churn cũ của snapshot làm outcome cho một prediction mới có feature hoặc horizon khác.
- Nếu DB lỗi: ghi trạng thái monitoring suy giảm, thời điểm report cuối và lỗi thu thập; không hiển thị report cũ như report mới. Có timeout và giới hạn ghi rõ ràng.

**Hoàn thành khi:** lịch sử còn sau restart/redeploy và có thể nối nhãn đúng prediction, đúng phiên bản.

### Bước 3 — Data drift, prediction drift và chất lượng dữ liệu

**Ước lượng: 0,5 ngày.**

- Theo dõi feature hành vi như `recency_days`, `monthly_transactions`, `trans_amount_3m`, `trans_trend_ratio`, `app_logins_monthly`, complaints và các feature còn lại của model.
- Numeric: PSI với bins cố định từ reference; xử lý riêng giá trị ngoài biên, missing và bin không có quan sát.
- Categorical: Jensen–Shannon distance, căn chỉnh category và xử lý category mới rõ ràng.
- Tính riêng drift của churn probability và tỷ lệ segment. Thay đổi phân phối prediction/cluster là tín hiệu cần điều tra, chưa tự chứng minh chất lượng model giảm.
- Theo dõi missing rate, lỗi schema/range và category lạ; request bị validation từ chối được thống kê riêng.
- Mỗi window hiển thị số mẫu, thời gian, nguồn, reference/model version và các feature thay đổi mạnh nhất.
- Chưa đủ mẫu trả `insufficient_data` cùng số mẫu hiện tại; không chuyển thành trạng thái bình thường.

**Hoàn thành khi:** cohort ổn định và cohort dịch chuyển tạo ra kết quả phù hợp, có thể giải thích bằng phân phối reference/current.

### Bước 4 — Hiệu năng theo thời gian và trigger

**Ước lượng: 0,5 ngày.**

- Tính ROC-AUC, Average Precision (ghi rõ đây là cách tính chỉ số PR), Precision, Recall, F1 và confusion matrix theo window đã có nhãn.
- Hiển thị số mẫu đủ horizon, số mẫu có nhãn, label coverage, số positive/negative và threshold phân loại.
- Chưa có nhãn trả `awaiting_labels`; chỉ có một class thì metric không xác định để `null` kèm lý do.
- Đối chiếu với baseline độc lập của cùng model; khi deploy model mới mở chuỗi version mới và lưu lịch sử cũ.
- Ưu tiên classifier quality để đáp ứng Module 7. Với clustering, theo dõi tỷ lệ cluster và khoảng cách tới centroid như chỉ số bổ sung; không dùng AUC/F1 để chấm model không có nhãn cluster chuẩn.

**Ngưỡng khởi đầu đề xuất — hiệu chỉnh trên validation/các cohort kiểm chuẩn trước khi chốt:**

| Tín hiệu | Điều kiện đề xuất | Hành động |
| --- | --- | --- |
| Chưa đủ mẫu drift | Dưới 500 mẫu hợp lệ/window | `insufficient_data`, tiếp tục thu thập |
| PSI từng numeric feature | Từ 0,10 cảnh báo; từ 0,25 thay đổi mạnh | Hiển thị feature, score và phân phối |
| JS distance từng categorical feature | Từ 0,10 cảnh báo; từ 0,20 thay đổi mạnh | Hiển thị category đóng góp lớn; chốt rõ distance, không nhầm divergence |
| Drift kéo dài | Ít nhất 30% feature theo dõi ở mức mạnh, hoặc feature hành vi trọng yếu PSI >= 0,25, trong 2 window không chồng lấn đủ mẫu | `review_required`; mở điều tra chất lượng dữ liệu và nhu cầu nhãn |
| Hiệu năng giảm | Recall hoặc Average Precision giảm >= 0,05 điểm tuyệt đối so với baseline, trong 2 window không chồng lấn đủ nhãn | `retraining_required` |
| Đủ điều kiện xét hiệu năng | Ít nhất 500 nhãn trưởng thành, >= 50 positive, >= 50 negative và coverage >= 80%/window | Cho phép áp dụng rule hiệu năng; không đạt thì báo thiếu bằng chứng |
| Dữ liệu lỗi | Invalid/missing rate vượt 5% trên ít nhất 100 quan sát | `data_quality_alert`, xử lý pipeline/nguồn dữ liệu trước |

Các ngưỡng trên là cấu hình riêng của project, không phải ngưỡng do đề bài quy định hay chuẩn áp dụng cho mọi ngân hàng.

Quy trình sau trigger: kiểm tra dữ liệu và nhãn -> chuẩn bị tập mới -> train candidate -> đánh giá trên tập độc lập và so với model hiện hành -> người phụ trách chọn phiên bản -> deploy hoặc rollback có lưu biên bản. Cảnh báo lưu cả bằng chứng, rule version và trạng thái xử lý.

### Bước 5 — Monitoring API và dashboard

**Ước lượng: 0,5–1 ngày.**

- Giữ `/api/monitoring` làm summary có trạng thái, version, thời điểm cập nhật và khả năng tính từng nhóm metric.
- Thêm API đọc lịch sử/run detail/report; API vận hành chạy job và nhập outcome được bảo vệ bằng token phía server.
- Dashboard có bộ lọc nguồn, model version và thời gian; mặc định ưu tiên nguồn live. Khi chưa có live data, hiển thị trạng thái tương ứng và lựa chọn xem study replay rõ ràng.
- Hiển thị: health và độ mới dữ liệu; bảng feature drift; biểu đồ distribution; đường metrics theo window; coverage nhãn; prediction/cluster distribution; danh sách alert và hành động đề xuất.
- Loại các số `ILLUS` khỏi nội dung trình bày như phép đo; chart lấy dữ liệu từ cùng API/report.
- Bổ sung trạng thái loading, thiếu mẫu, chờ nhãn, report cũ và backend/DB không sẵn sàng. Bổ sung xử lý cold start của Render cho luồng monitoring.

**Hoàn thành khi:** người xem biết đang xem dữ liệu nào, model nào, có đủ bằng chứng chưa và vì sao một alert xuất hiện.

### Bước 6 — Replay, kiểm thử, deploy và tài liệu nộp

**Ước lượng: 0,5–1 ngày.**

- Tạo 3 kịch bản tái lập bằng seed: ổn định, drift hành vi, và suy giảm hiệu năng có kiểm soát.
- Replay chỉ dùng tập đã giữ ngoài huấn luyện, được gắn `study_replay` và scenario riêng. Các window trong mỗi scenario không chồng lấn; tái sử dụng cohort giữa các scenario phải được ghi rõ, không cộng thành bằng chứng thời gian thực.
- Kịch bản suy giảm tính metric từ dự đoán/nhãn của kịch bản; mọi thay đổi feature/nhãn phục vụ stress test đều ghi lại. Kết quả cảnh báo và biểu đồ phải được tính, không hardcode.
- Test: sai model/reference version, trùng sự kiện, cache, DB lỗi/restart, không có nhãn, một class, ít mẫu, bins rỗng/category mới, shift rõ ràng, threshold boundaries và 2 window độc lập cho trigger.
- Chạy các API regression tests hiện có và build frontend. Kiểm tra live CORS, API summary/history/report, dữ liệu sau redeploy và trạng thái trên desktop/mobile.
- Tạo CI cho monitoring; pin dependency sau khi xác minh tương thích; ghi secrets vận hành trong GitHub/Render settings.
- Nộp report JSON/HTML cùng hướng dẫn demo: xem normal -> chạy drift -> xem lý do cảnh báo -> nhập outcome/đánh giá performance -> xem retraining trigger -> restart và xem lịch sử còn nguyên.

## 5. Các file dự kiến

| File/thư mục | Vai trò |
| --- | --- |
| `api/prepare_module5.py`, `api/module5.py` | Kế thừa checkpoint/preprocessor manifest và baseline đã được kiểm chứng |
| `monitoring/baseline.py` | Xuất reference và baseline chất lượng |
| `monitoring/store.py`, `monitoring/schema.sql` | Lưu trữ và migration |
| `monitoring/drift.py`, `monitoring/performance.py` | Tính metric và trạng thái thiếu bằng chứng |
| `monitoring/runner.py` | Chạy theo window, áp dụng rule, lưu report/alert |
| `monitoring/simulate.py` | Replay và stress scenarios có nguồn rõ ràng |
| `config/monitoring.yaml` | Feature list, window, sample size, threshold, retention |
| `api/main.py` | Instrument inference, API đọc/vận hành monitoring |
| `frontend/src/lib/api.ts` | Kiểu dữ liệu và lời gọi monitoring API |
| `frontend/src/routes/monitoring.tsx` | Dashboard dùng kết quả thật |
| `tests/test_monitoring_*.py` | Kiểm thử drift, labels, store, rules và API |
| `.github/workflows/monitoring-ci.yml` | Kiểm thử/build và kiểm tra hợp đồng API |
| `.github/workflows/monitoring-run.yml` | Chạy lịch và chạy thủ công |
| `render.yaml`, `.env.example` | Cấu hình deploy/env theo dịch vụ thực tế |
| `docs/module-7-monitoring.md` | Thiết kế, ngưỡng, giới hạn, runbook và hướng dẫn demo |

## 6. Lịch thực hiện và nghiệm thu

Ước lượng 4–6 ngày làm việc cho một người quen repo, với quyền cấu hình DB và Render sẵn có. Thứ tự phụ thuộc: baseline -> lưu trữ -> metrics/rules -> API/dashboard -> kiểm thử/deploy/demo.

| Mốc | Kết quả |
| --- | --- |
| Ngày 1 | Model version, split, baseline và schema lưu trữ |
| Ngày 2 | Ghi inference/outcome, drift và data quality |
| Ngày 3 | Performance, windows, trigger và report |
| Ngày 4 | API, dashboard và replay |
| Ngày 5–6 | Kiểm thử lỗi/regression, deploy, persistence check và tài liệu/demo |

Module 7 được tính là hoàn thành khi:

- [ ] Có drift đo được từ reference và incoming/replay được phân biệt rõ.
- [ ] Có metrics theo nhiều window đã có nhãn, đúng model và không dùng dữ liệu fit để công bố chất lượng độc lập.
- [ ] Chưa có nhãn hoặc chưa đủ mẫu được hiển thị đúng trạng thái.
- [ ] Có trigger định lượng, bằng chứng kích hoạt và quy trình tái huấn luyện.
- [ ] Live dashboard và report dùng cùng kết quả tính toán.
- [ ] Lịch sử còn sau restart/redeploy; job được chạy lại không nhân đôi kết quả.
- [ ] Ba scenario demo, kiểm thử, CI, frontend build và kiểm tra live đều hoàn tất.
- [ ] Có báo cáo/runbook và giải thích giới hạn dữ liệu nghiên cứu, nhãn trễ và dữ liệu mô phỏng.

## 7. Tài liệu tham khảo

- [Thông báo project](../../Business_Analytics_Project_Announcement.pdf): Module 7, Deliverables, Evaluation Focus.
- [Render — Deploy for Free](https://render.com/docs/free): filesystem tạm, spin down và thời hạn Postgres free.
- [Evidently — Data Drift](https://docs.evidentlyai.com/metrics/preset_data_drift): reference/current, drift khi chưa có ground truth và cấu hình phương pháp/ngưỡng.
