# Synthetic demo monitoring

## Đánh giá hướng dẫn và kế hoạch thực hiện

Triển khai theo `Synthetic_Demo_Monitoring_Implementation_Guide.md` ngày 24/11/2026 trên fork `nthne/Intro-to-Business-Analysis`, nhánh `demo`. Thiết kế tái sử dụng Module 7 hợp lý: giữ checkpoint, preprocessing và reference cố định; nhãn synthetic tách khỏi đầu vào; cửa sổ cùng scenario để xét streak; không retrain để làm đẹp demo.

Các điều chỉnh sau kiểm tra code thực tế:

- `monitoring.drift` tính **Jensen–Shannon distance base 2**, tức căn bậc hai divergence. Giữ nguyên phương pháp và thresholds hiện có; không dùng công thức divergence rồi so với ngưỡng distance.
- Generator upstream dùng công thức momentum khác serving fork. Profile churn chuyển sang `model_development.preprocessing`; diagnostic successful engagement giữ riêng. Sửa lời gọi `risk_level` của route demo cho đúng chữ ký fork.
- Mỗi dataset/model/label policy/seed/config/anchor có scenario `synthetic_demo_<digest>` riêng. Nhãn thay đổi không ghi đè outcome cũ hoặc trộn với drift-only.
- Batch metadata nằm trong report JSON; không cần migration schema. Event, outcome, report, alert của một batch commit/rollback cùng transaction. PostgreSQL dùng transaction advisory lock cho cùng scope.
- Performance synthetic so tương đối với **pool ba batch normal** cùng policy, khi đủ nhãn. Study baseline vẫn hiển thị nhưng khác nguồn nhãn, không dùng để kích hoạt performance alert synthetic. Baseline normal không tự đánh giá suy giảm so với chính nó.
- Checkpoint study tại threshold 0,5 dự đoán churn toàn bộ test cohort: accuracy 0,204, recall 1,0. Giữ nguyên checkpoint/threshold; đây là hạn chế của model hiện có.
- Build Render khác runtime: [Starter pipeline có 8 GB RAM](https://render.com/docs/build-pipeline). Có đường build từ release 3 được pin, không cài torch/optuna, và đường private artifact nếu cần. Replay không tự chạy khi server khởi động.

| Giai đoạn | Công việc | Điều kiện kiểm chứng |
|---|---|---|
| A | Python 3.11, kiểm tra fork/demo, đủ 7 bảng release 3, prepare artifacts | Study reproduced, 133 model features |
| B | Generate/validate 10.000 mẫu, sửa tương thích | 5 × 2.000, API parity bằng 0 |
| C | Drift-only, outcome policy, transaction replay, API/UI | No leakage, idempotency, rollback, source isolation |
| D | SQLite và PostgreSQL test riêng, frontend build | Suite pass, không dùng production trong tests |
| E | Render đúng fork, backend rồi frontend, replay, persistence | Health/model ID, 5 report/mode trên Supabase, UI lấy API thật |

## Chạy local

Từ repo root, dùng Python 3.11. Có thể gọi thẳng `.venv-synthetic\Scripts\python.exe` nếu PowerShell chặn activation. Bộ serving đã pin phiên bản tại `requirements-serving.txt`; research/dev dependencies nằm trong `requirements-demo.txt`.

```powershell
python -m pip install -r requirements-serving.txt -r requirements-demo.txt
python -m tools.prepare_demo_source
python -m api.prepare_module5 --source data/full_dataset
python -m api.official_data --source data/full_dataset
python -m api.segmentation_service
python -m demo_data.generate --multiplier 10 --max-samples 10000 --seed 42
# Dùng SQLite riêng cho API parity; không trỏ database production để test.
$env:DATABASE_URL = 'sqlite:///artifacts/monitoring/validation.sqlite3'
python -m demo_data.validate
```

`tools.prepare_demo_source --source <thu_muc_release_3>` dùng cache đã tải, kiểm tra đủ 7 bảng và snapshot. Không ghi đè source đã tồn tại. Nguồn và output demo đều gitignore.

```powershell
$env:DATABASE_URL = 'sqlite:///artifacts/monitoring/synthetic-demo.sqlite3'
# Chỉ đo drift, không tạo nhãn:
python -m monitoring.synthetic_replay --data-dir data/demo --all --export-dir artifacts/synthetic-drift-reports
# Performance minh họa với nhãn độc lập:
python -m demo_data.labels --seed 4242
python -m monitoring.synthetic_replay --data-dir data/demo --all --labels data/demo/synthetic_outcomes.csv --export-dir artifacts/synthetic-demo-reports
# Có thể chạy một batch; cửa sổ vẫn giữ nguyên vị trí trong chuỗi:
python -m monitoring.synthetic_replay --data-dir data/demo --batch engagement_shift --labels data/demo/synthetic_outcomes.csv
```

Không nhãn: metric null, trạng thái `awaiting_labels`. Có nhãn: xác minh checksum, policy và join một-một, sau đó lưu source `synthetic_stress`. Nhãn không bao giờ đi vào 63 model inputs. Policy `config/synthetic_outcome_v1.json` có intercept, weights, fixed scales/caps và smoothing công bố trước; Bernoulli dùng randomness riêng theo ID/seed, không đọc probability XGBoost hoặc donor churn. Thay đổi giả định cần version policy mới.

Anchor mặc định lấy ngày đăng ký model trừ horizon + 7 ngày, UTC midnight. Mỗi batch một ngày; timestamps là giả lập. Adapter từ chối cửa sổ tương lai, quá retention hoặc nhãn chưa đủ horizon. `--anchor` cho phép mốc UTC cố định mới; thay anchor tạo run identity mới. Scenario tùy chỉnh phải bắt đầu `synthetic_demo`; không được tái dùng cho identity khác.

Replay kiểm tra/scoring toàn dataset trước lần ghi đầu tiên. Mỗi batch là một transaction; lỗi giữa batch rollback toàn batch. Chạy lại cùng identity tạo 0 event/outcome mới và cập nhật đúng 5 report IDs. Thay chế độ nhãn tạo scope riêng, vì vậy chạy cả hai chế độ lưu 20.000 events, 10.000 outcomes và 10 reports. Chỉ chạy chế độ có nhãn lưu 10.000/10.000/5.

## API và dashboard

Đặt `DEMO_DATA_DIR=data/demo` trước startup. Admin route cần `ENABLE_MONITORING_DEMO=true`, database sẵn sàng và `Authorization: Bearer <MONITORING_ADMIN_TOKEN>`.

```text
POST /api/monitoring/synthetic-replay
{"batch":"normal_01","with_labels":false}
{"all":true,"with_labels":true}
```

Chọn đúng một batch hoặc `all=true`. HTTP body không nhận đường dẫn tùy ý. Route labels chung từ chối gắn nhãn vào synthetic demo; dùng adapter versioned để giữ provenance. Dashboard public không lưu admin token hay gọi replay. Đối với hosting chậm, chạy lần lượt từng batch; nếu timeout, retry cùng batch an toàn.

API trả scenario thực trong response, không giả định tên luôn là `synthetic_demo`:

```text
GET /api/monitoring?source=study_replay&scenario=<scenario_thuc>&days=30
GET /api/monitoring/runs/<id>
GET /api/monitoring/runs/<id>/report?format=json
GET /api/monitoring/runs/<id>/report?format=html
```

Trong `/monitoring`, chọn Study replay, scenario Synthetic Demo (Drift only hoặc Simulated labels), model đúng và một batch 1–5. UI có coverage, accuracy/F1/AP/ROC-AUC, feature PSI/JS distributions, prediction drift, segment distribution, cảnh báo và provenance. Metric chưa có hiển thị N/A. Thời gian là giả lập, latency replay không đại diện Render. Quality chỉ tính validated rows và missingness; không đại diện mọi lỗi request thật. Cảnh báo là minh họa, không tự retrain/deploy.

## Kiểm thử

```powershell
python -m pytest tests -q
# Windows bị chặn thư mục pytest mặc định: tạo .test-tmp rồi dùng --basetemp .test-tmp/run-moi
# Optional: MONITORING_TEST_DATABASE_URL phải là PostgreSQL disposable riêng.
cd frontend
npm.cmd run build
```

`tests/conftest.py` mặc định cô lập API telemetry vào SQLite tạm, không dùng DATABASE_URL từ `.env` cá nhân. PostgreSQL integration tests chỉ dùng `MONITORING_TEST_DATABASE_URL`. CI monitoring chạy trên push `demo`, chuẩn bị pinned serving artifacts, generate/validate đầy đủ, pytest và frontend build.

## Render

Giữ nguyên hai dịch vụ hiện có, xác nhận repo là fork `nthne/Intro-to-Business-Analysis`, branch `demo`. Không deploy upstream. Script `tools.configure_render_monitoring` có chế độ đọc, lưu rollback snapshot trước khi đổi; `--relink-fork` chỉ chấp nhận đúng upstream đã biết/branch demo.

Build: `pip install -r requirements-serving.txt && python -m tools.prepare_serving`. Python 3.11.15. Start: migration hiện có rồi Uvicorn. Không thay checkpoint hoặc reset database.

Hai phương án dữ liệu:

1. `DEMO_BUILD_MODE=generate`, `DEMO_DATA_DIR=data/demo`: tải release 3, prepare, generate 10.000, validate API parity trong SQLite build riêng và tạo nhãn seed 4242. Không mở database production hay replay trong build.
2. Private artifact: `python -m tools.demo_artifact --pack artifacts/synthetic-serving-bundle.zip`, đưa bundle vào private HTTPS storage, cấu hình `DEMO_ARTIFACT_URL`, `DEMO_ARTIFACT_SHA256`, `DEMO_DATA_DIR=data/demo`. Bundle kèm model preprocessing/reference/segmentation và hai snapshot đúng hash, không kèm 7 bảng nguồn. Installer chỉ chấp nhận whitelist đường dẫn và hash, kiểm tra Python/package versions và inference. Không public bundle/dataset hoặc commit credentials.

Backend cần DATABASE_URL, MONITORING_ADMIN_TOKEN, writer role hiện có, CORS tới frontend. Frontend build dùng `VITE_API_BASE_URL=https://babank.onrender.com` và SPA rewrite về index.html. Deploy backend trước, health pass rồi frontend. Replay gọi rõ ràng sau deploy, kiểm tra API/Supabase cùng model ID/reference. Rerun không nhân đôi, reload server vẫn giữ reports trên PostgreSQL.

## Kết quả đo local ngày 24/11/2026

Generator/validator: 10.000 ID riêng, 5 × 2.000, không copy nguyên feature vector nguồn; 50 requests classifier, 50 KMeans, 50 Ward surrogate đều parity pass; max probability delta 0. Model ID local sau kiểm tra build đầy đủ: `3264bc15320fadfd54d4c9acb106b917bdddb5a14b34c3b2a15ab6a9217a8948`.

Dataset SHA256 `8a54c89258e7742ee060d100d38c39e4df41085fd21309f2ed5f61270d70d3cd`, generation seed 42; label policy `synthetic_outcome_v1`, label seed 4242. Window dates 19–24/08/2026 là giả lập. Đây là kết quả local, không phải latency hoặc performance khách hàng thực tế của Render.

| Batch | N | Critical share | Prediction PSI | ROC-AUC | AP | Accuracy | Recall | F1 | Kết quả |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| normal_01 | 2000 | 0 | 0,0559 | 0,5842 | 0,3364 | 0,2805 | 1 | 0,4381 | stable |
| normal_02 | 2000 | 0 | 0,0672 | 0,5946 | 0,3477 | 0,2755 | 1 | 0,4320 | stable |
| normal_03 | 2000 | 0 | 0,0740 | 0,6034 | 0,3531 | 0,2865 | 1 | 0,4454 | stable |
| engagement_shift | 2000 | 0,0476 | 0,2178 | 0,5598 | 0,4128 | 0,3700 | 1 | 0,5401 | critical feature drift |
| friction_shift | 2000 | 0,0476 | 0,1935 | 0,5889 | 0,4456 | 0,3855 | 1 | 0,5565 | review_required: missingness |

Mỗi batch có coverage 100%, đủ hai lớp; precision bằng accuracy vì checkpoint dự đoán toàn bộ positive tại threshold 0,5. Không có performance/retraining alert theo baseline synthetic. Friction có missing excess khoảng 0,2996 do `avg_resolution_delay_days` thiếu khi chỉ có complaint chưa giải quyết; đây là đặc tính synthetic cần xem xét, không phải request failure. Feature critical share nhỏ nên không có chuỗi data-drift alert đủ điều kiện.

File aggregate đầy đủ: `artifacts/synthetic-demo-reports/summary.csv`, `features.csv`, từng report JSON/HTML; drift-only ở `artifacts/synthetic-drift-reports/`. Rerun cùng identity đã xác nhận 0 event/outcome mới. Những artifact này chỉ nằm local, không theo dõi trong Git. Kết quả triển khai được ghi riêng bên dưới.

Suite cuối: **68 passed**, gồm PostgreSQL 17 Docker riêng, không còn test skip. Frontend TypeScript/Vite build pass. Full build `DEMO_BUILD_MODE=generate` pass; peak working set quan sát khoảng 1,22 GB. Đã bổ sung regression cho import khi bật demo và đường dẫn segmentation độc lập với Module 5.

## Kết quả triển khai ngày 25/11/2026

Backend và frontend hiện có đã chuyển từ upstream sang `nthne/Intro-to-Business-Analysis`, branch `demo`, và deploy application commit `6000e21bb9ed5ccec58329f61ac0b8c25b100151`. [CI Linux](https://github.com/nthne/Intro-to-Business-Analysis/actions/runs/45396962697) pass 68 tests, generate/validate 10.000 mẫu và frontend build. Hai secret được cấu hình trực tiếp trong backend Render sau xác nhận của người dùng; không ghi vào Git hoặc frontend.

- Backend: https://babank.onrender.com, deploy `dep-dc35t7k0916s73fqu4a0`.
- Frontend: https://babank-dashboard.onrender.com/monitoring, deploy `dep-dc35v9qfngtc73d1gnk0`.
- Build Render dùng pinned Kaggle release 3, `DEMO_BUILD_MODE=generate`, không cài research stack torch/optuna. Generate, hai fitted adapters và validation/API parity đều pass.
- `/health`: model, customers, segmentation, study reproduction đều true; feature count 133; PostgreSQL storage ready/persistent và reference available đều true; không có ingest error.
- Serving model ID: `00621aa7242712b7f388cf8cf0126614fb0d791d13d3fdc74ab71ee80778bcb2`.
- Fixed reference SHA256: `6f5271ebe88ce6cf90335de0b6044451f010fcfe9495c314d3edc690ea9a9eef`.
- Dataset SHA256 khớp local: `8a54c89258e7742ee060d100d38c39e4df41085fd21309f2ed5f61270d70d3cd`.
- Scenario production: `synthetic_demo_6bdec224083a`; generation seed 42; label seed 4242; policy `synthetic_outcome_v1`.

Chạy admin API lần lượt đủ năm batch, mỗi batch 2.000 dòng. SQL kiểm chứng trên Supabase xác nhận đúng **10.000 events, 10.000 unique prediction IDs/subjects, 10.000 outcomes**, tất cả outcome source `synthetic_stress`, và **5 reports**. Coverage nhãn và KMeans của cả năm batch đều 100%; cùng serving model và reference. Không có synthetic subjects trong scope ngoài `study_replay`. Metrics và cảnh báo từng batch khớp bảng kết quả local phía trên; execution là replay trên backend Render, không phải phép đo serving latency hoặc độ chính xác trên khách hàng thật.

Chạy lại `normal_01` tạo 0 events và 0 outcomes. Restart backend qua Render API, xác nhận process mới khởi động lúc 04:13:38 UTC, sau đó `/health` vẫn persistent/ready và history vẫn đúng **5 report IDs** trước restart. Không tự replay khi startup.

`/monitoring` trả HTTP 200 nhờ SPA rewrite. Bundle đã publish có selector Synthetic Demo, batch metadata và đúng API URL; không chứa giá trị DATABASE_URL hoặc MONITORING_ADMIN_TOKEN. Admin replay thiếu token trả 401. Kiểm tra hình ảnh UI trực tiếp chưa thực hiện được do browser automation host không sẵn sàng; kiểm chứng frontend build, asset và API đã pass.

Để xem: mở `/monitoring`, chọn **Study replay / stress tests**, rồi scenario **Synthetic Demo · Simulated labels (seed 4242) · 6bdec224083a**, giữ Current serving model và chọn batch 1–5. Scope Incoming API requests vẫn tách riêng.

Report JSON/HTML tải từ backend được lưu tại `artifacts/deployed-synthetic-reports/`, kèm `summary.csv`, `verification.json` và `after-restart.json`. Rollback snapshot của cấu hình/deploy trước thay đổi ở `artifacts/monitoring/render-before-synthetic.json`. Các file này không được commit. Script cấu hình đã sửa để bỏ biến artifact tùy chọn thay vì gửi chuỗi rỗng mà Render từ chối, và lọc secret trong thông báo lỗi API.
