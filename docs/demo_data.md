# Dữ liệu demo Customer Churn & Segmentation

Chạy ở thư mục gốc của repo, trong môi trường Python đã cài `requirements.txt`. Các lệnh đọc bảy bảng từ `data/full_dataset/` và các artifact phục vụ hiện có.

```powershell
python -m demo_data.generate --multiplier 10 --max-samples 10000 --seed 42
python -m demo_data.validate
python -m pytest tests/test_demo_data.py -q
```

## Dữ liệu được sinh như thế nào

Mỗi mẫu là một khách hàng tổng hợp tại snapshot **01/10/2026 00:00** (mốc loại trừ). Bộ sinh đối chiếu ID trong bảng khách hàng gốc với `customer_context`, rồi tính `N_demo = min(10 × N_base, 10.000)` theo tham số mặc định. Seed 42 điều khiển việc chọn mẫu và biến đổi. Với nguồn hiện tại, 10.000 lịch sử được dùng làm mẫu tham chiếu, mỗi lịch sử một lần. Bộ sinh cấp ID demo mới, chia đều vào năm lô và phân số dư lần lượt từ `normal_01`.

1. Ghép khối tuổi, điểm tín dụng, trạng thái thẻ và lương từ một hồ sơ khác nếu có khách hàng tương đồng theo quốc gia, giới tính, trạng thái hoạt động, số sản phẩm, nhóm tenure và nhóm lương/số dư. Nếu nhóm chỉ có một hồ sơ, dùng khối của hồ sơ đó và tiếp tục biến đổi.
2. Biến đổi số tiền theo hệ số chung cho mỗi danh mục khách hàng, với mức tăng hoặc giảm từ 4% đến 12%. Tính số dư khách hàng từ số dư tài khoản đã biến đổi. Hai vế chuyển nội bộ dùng cùng số tiền và thời điểm. Với chuyển liên khách hàng, hai vế giữ cùng số tiền và được nối lại nếu cùng có mặt trong lô.
3. Cấp ID mới cho các thực thể và sự kiện. Dời giờ trong ngày bằng cùng một phép biến đổi tăng dần để giữ thứ tự, các mốc theo ngày và thời điểm của các cặp chuyển khoản. Chọn 4% số ngày hoạt động đủ điều kiện để bỏ bớt sự kiện chủ động và lượt đăng nhập, đồng thời giữ các phiên cần cho giao dịch di động còn lại. Đây là lịch sử tổng hợp cơ sở của cả năm lô.

| Lô | Mẫu | Cách xử lý lịch sử nguồn |
| --- | ---: | --- |
| `normal_01`, `normal_02`, `normal_03` | 2.000 mỗi lô | Dùng lịch sử tổng hợp cơ sở để giữ phân phối gần nguồn tham chiếu. |
| `engagement_shift` | 2.000 | Chọn các ngày trong quý 3 của từng khách hàng với xác suất 55%, cộng 21 ngày cuối cửa sổ. Trong các ngày được chọn, bỏ lượt đăng nhập và giao dịch chi ra được phép bỏ. Giữ chuyển khoản liên kết, giao dịch thụ động, tài khoản tiết kiệm hoặc đã đóng, và phiên đăng nhập cần cho giao dịch di động còn lại. |
| `friction_shift` | 2.000 | Với xác suất 65% trên khách hàng có lượt đăng nhập thành công phù hợp trong quý 3, thêm một lần thất bại ngay trước lần thành công và một khiếu nại chưa xử lý tại thời điểm đó. |

ETL tính lại số đếm, số tiền, recency và tỷ lệ từ sự kiện đã sinh. Recency khi chưa có sự kiện thành công dùng thời gian đã quan sát. Tỷ lệ thất bại để trống khi mẫu số bằng 0, và thời gian giải quyết để trống khi chỉ có khiếu nại chưa xử lý. JSON biểu diễn các giá trị thiếu bằng `null`. Validator bỏ ID và metadata khỏi phép so khớp, rồi đối chiếu cả 10 thuộc tính hồ sơ và 63 đầu vào churn với nguồn. `manifest.json` ghi mapping hai mẫu tham chiếu, tỷ lệ trùng, hash và thống kê so với nguồn cũng như so với lịch sử tổng hợp trước khi áp dụng drift.

## Dùng với API

```powershell
$env:DEMO_DATA_DIR = "data/demo"
python -m uvicorn api.main:app
```

Mở `/docs` để gọi các route khi chạy local.

- `GET /api/demo/customers` tìm ID demo. `GET /api/demo/customers/{id}` trả về hai payload cho `POST /api/predict` và `POST /api/segments/predict`.
- Mỗi dòng `data/demo/api_examples.jsonl` là một body cho `POST /api/predict`, lấy từ `customer_features.csv`. Request chuyển đủ 63 aggregate thô trong `raw_features` và các trường bắt buộc của `CustomerInput`.
- `POST /api/demo/batches/{batch_id}/score` chấm điểm đúng lô được chọn và trả về phân bố cụm, xác suất churn trung bình và thống kê đầu vào đối chiếu với nguồn (trung bình, P10, trung vị, P90, số giá trị thiếu). Thêm `?include_predictions=true` để lấy kết quả từng khách hàng. `POST /api/predict/batch` tiếp tục chấm điểm snapshot chính thức.

### Schema và phạm vi kiểm chứng

- XGBoost Module 5 dùng 63 trường bắt buộc trước preprocessing, gồm 59 số và 4 trường phân loại `country`, `gender`, `preferred_channel`, `primary_device`. Trong dữ liệu demo, aggregate đầu vào có thể thiếu là `avg_resolution_delay_days`. Preprocessor đã fit tạo 133 trường cho checkpoint. `monthly_transactions` của profile Module 5 đo toàn bộ giao dịch đã ghi sổ, còn `monthly_initiated_transactions` dùng riêng cho monitoring tương tác chủ động.
- Bundle segmentation đang phục vụ dùng 17 trường, gồm 15 số và 2 trường phân loại. Các trường profile bắt buộc của API phải có giá trị. Bundle áp dụng imputation, log, scaling, one-hot và trọng số nhóm đã lưu. Các cột `seg_*` giữ cách chiếu hành vi dùng để huấn luyện bundle, gồm cách tính tần suất và recency của snapshot đó. Dữ liệu demo loại lượt nhận thụ động khỏi tương tác và dùng thời gian thực nếu recency dưới một ngày. Ward gán khách hàng mới bằng tâm cụm gần nhất.
- Cả hai mô hình dùng đặc trưng tổng hợp tại snapshot. Lịch sử sự kiện được dùng ở khâu sinh, còn inference chạy trên aggregate đã dựng. Các file đầu ra không chứa embedding, nhãn churn hoặc kết quả huấn luyện lại.

Đây là dữ liệu tổng hợp để kiểm thử inference, API và monitoring. Bảng khách hàng gốc là nguồn của nghiên cứu, được chia train/validation/test khi xây dựng mô hình. Artifact đã huấn luyện được dùng lại trực tiếp khi chấm điểm demo. Số dư demo xuất phát từ tài khoản tham chiếu đã biến đổi, và lịch sử chỉ phục vụ tính aggregate, chưa tái dựng sổ cái. Thống kê diagnostic trong manifest dùng toàn bộ nguồn; Module 7 dùng fixed reference của training split cùng checkpoint. Năm lô là các đợt nạp mô phỏng của cùng một snapshot. Để đánh giá độ chính xác trên khách hàng mới cần dữ liệu holdout độc lập có nhãn churn quan sát được.

Profile churn hiện dùng công thức `model_development.preprocessing` của fork để khớp serving; các cột diagnostic engagement vẫn dùng số lượt thành công. Risk level có ngưỡng riêng 0,3/0,7, độc lập với decision threshold.

Dashboard monitoring đã hỗ trợ lịch sử 5 batch, drift, segmentation và metric nhãn mô phỏng. Xem [hướng dẫn replay, nhãn độc lập, triển khai và kết quả](synthetic-demo-monitoring.md).
