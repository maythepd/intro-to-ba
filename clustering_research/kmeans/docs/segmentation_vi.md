# Module 5B: Phân khúc khách hàng bằng K-Means

[Bản tiếng Anh](segmentation.md)

**Mục tiêu:** kiểm tra dữ liệu hành vi mô phỏng có bổ sung thông tin hữu ích
cho phân khúc khách hàng hay làm thay đổi cấu trúc vốn có của dữ liệu gốc.

## 1. Các thí nghiệm và thiết lập

| Thí nghiệm | Dữ liệu đầu vào | Mục đích |
|---|---|---|
| **A — Chỉ dữ liệu gốc** | 10 đặc trưng Kaggle: tuổi, số dư, điểm tín dụng, số sản phẩm… | Xác định phân khúc ban đầu làm mốc so sánh. |
| **B — Chỉ hành vi mô phỏng** | 24 đặc trưng giao dịch, đăng nhập, khiếu nại và tiết kiệm. | Xác định các nhóm khách hàng theo hành vi. |
| **C — Kết hợp** | 34 đặc trưng từ A+B, cân bằng đóng góp hai nhóm. | Kiểm tra giá trị bổ sung của hành vi so với dữ liệu gốc. |
| **D — Đối chứng hoán vị** | Giữ nguyên A, tráo nguyên hàng hành vi B giữa khách hàng trong từng tập dữ liệu. | Kiểm tra lợi ích của C có đến từ liên hệ khách hàng–hành vi hay chỉ do thêm chiều dữ liệu. |

So sánh **C với A** để đánh giá tác động của việc thêm hành vi; so sánh **C với D**
để đánh giá thông tin từ việc ghép đúng khách hàng và hành vi. Cả bốn dùng cùng
khách hàng và cách chia tập; D không chứng minh dữ liệu mô phỏng sát thực tế.

| Thiết lập | Giá trị |
|---|---|
| Dữ liệu | 10.000 khách hàng từ [bộ dữ liệu công khai của nhóm](https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba). |
| Chia tập | Train: 7.999; validation: 1.001; test độc lập: 1.000; seed 42; không phân tầng theo churn. |
| Cửa sổ quan sát | Từ 01/04/2026 đến trước 01/10/2026. |
| Tiền xử lý | Điền giá trị thiếu, biến đổi log cho biến số lệch, StandardScaler và one-hot encoding; chỉ học tham số trên train. |
| K-Means | k=2–10; n_init=10; max_iter=300; seed 42, 137, 2026; tối đa 1.000 khách hàng khi tính silhouette. |
| Trọng số C/D | Nhóm gốc 50%, nhóm hành vi 50% đóng góp kỳ vọng vào bình phương khoảng cách; kiểm tra thêm 25%/75% và 75%/25%. |
| Đối chứng D | 5 seed hoán vị: 101, 211, 307, 401, 503; hoán vị riêng trong train/validation/test. |

Chọn k nhỏ nhất có silhouette validation trung bình cách mức tốt nhất không quá
**0,02**, với tỷ lệ cụm nhỏ nhất trên train trung bình ≥ **2%** và ARI ổn định ≥ **0,8**.
Nếu không có ứng viên đạt cả hai ngưỡng, dùng silhouette hợp lệ và đánh dấu trong CSV.
**Churn không tham gia huấn luyện hoặc chọn k; chỉ được dùng sau khi cố định các mô hình.**

## 2. Kết quả trên các chỉ số phân cụm

Inertia lấy từ mô hình train được lưu với seed 42. Silhouette và Davies–Bouldin
là trung bình validation qua ba seed; ARI ổn định đo mức nhất quán giữa các lần
khởi tạo. D lấy trung bình năm lần hoán vị. Ký hiệu ↑/↓ là chiều tốt hơn của chỉ số.

| Thí nghiệm | k được chọn | Inertia train ↓ | Silhouette validation ↑ | Davies–Bouldin validation ↓ | ARI ổn định ↑ |
|---|---:|---:|---:|---:|---:|
| A | 2 | 6973.56 | 0.1407 | 2.4514 | 1.0000 |
| B | 2 | 6297.41 | 0.5118 | 1.1689 | 1.0000 |
| C | 2 | 6982.48 | 0.1385 | 2.3912 | 1.0000 |
| D — trung bình 5 lần | 2–3 | 6722.18 | 0.1426 | 2.3942 | 0.9975 |

Các chỉ số hình học mô tả từng không gian đặc trưng. Không thể chỉ dựa vào
silhouette, DB hoặc inertia để kết luận B/C tốt hơn A khi dùng đặc trưng khác nhau.

## 3. So sánh trên tập test độc lập

Tỷ lệ churn chung trên test là **20,3%**. Chỉ số phân tách churn dưới đây bằng
phương sai tỷ lệ churn giữa các cụm, có trọng số theo số khách hàng, chia cho
phương sai nhãn churn toàn tập. Giá trị cao hơn thể hiện liên hệ mô tả rõ hơn;
đây không phải độ chính xác dự đoán. So sánh cùng k giúp kiểm soát số lượng cụm.

| Thí nghiệm | Phân tách churn tại k được chọn | Phân tách churn khi cùng k=4 |
|---|---:|---:|
| A | 0.01208 | 0.05288 |
| B | 0.01109 | 0.03359 |
| C | 0.03055 | 0.04028 |
| D — trung bình 5 lần | 0.00280 | 0.00606 (0.00384–0.00973) |

Tại k=2, C tạo hai nhóm **ít tương tác** và **đang hoạt động**, có tỷ lệ churn
lần lượt **27,52%** và **13,45%**. A chủ yếu tách khách hàng theo mức số dư,
còn C chuyển sang phân nhóm theo hoạt động.

Mức tương đồng giữa các cách phân cụm trên test:

| Cặp thí nghiệm | ARI | NMI |
|---|---:|---:|
| A – B | -0.00151 | 0.00003 |
| A – C | -0.00008 | 0.00042 |
| B – C | 0.03460 | 0.12741 |

ARI/NMI không phụ thuộc tên nhãn cụm. ARI A–C gần 0 cho thấy cấu trúc phân nhóm
thay đổi đáng kể; mức tương đồng thấp tự nó chưa chứng minh tốt hay xấu.

## 4. Kết luận và giới hạn

- **C phân tách churn rõ hơn A tại k=2, nhưng kém A khi cùng k=4.**
- C vượt cả năm đối chứng D; ước lượng xác suất đuôi hoán vị có hiệu chỉnh là
  **0,1667**, nên năm lần hoán vị chưa cung cấp bằng chứng thống kê mạnh.
- Khi loại nhóm giao dịch/tiết kiệm, chỉ số phân tách churn của C còn **0,00050**.
  Hành vi mô phỏng phụ thuộc cờ hoạt động gốc, nên quy tắc sinh dữ liệu có thể
  giải thích lợi ích quan sát được.

**Chưa có bằng chứng dữ liệu mô phỏng cải thiện phân khúc một cách nhất quán.**
Không có mã nguồn bộ sinh hoặc dữ liệu hành vi thực để đối chiếu. Churn của quý
sau là giả định của dự án; kết quả chưa xác nhận được mức sát thực tế của mô phỏng.

## 5. Cách chạy và nơi lưu kết quả

Chạy từ thư mục gốc dự án, sử dụng môi trường đã chuẩn bị:

```powershell
.\.venv\Scripts\python.exe main.py --mode segmentation
.\.venv\Scripts\python.exe -m pytest -q
```

Kết quả CSV/biểu đồ: `results/segmentation/`; mô hình: `models/saved/segmentation/`.
Xem [hướng dẫn từng file kết quả](segmentation_outputs.md).
**16 kiểm thử đạt**; bộ kiểm thử pipeline hiện có cũng đã đạt. Mô hình sau khi
lưu và tải lại cho kết quả gán cụm khớp với bản đã xuất.
