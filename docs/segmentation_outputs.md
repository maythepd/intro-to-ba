# Hướng dẫn đọc kết quả segmentation

Thư mục `results/segmentation/` có **27 file** khi chạy đầy đủ với biểu đồ.
Nên đọc `comparative_summary.csv` → `personas.csv` → `common_k_comparison.csv` trước.
Các bảng chi tiết dùng cột `experiment` để phân biệt A/B/C, các seed D và sensitivity.

## 1. Kết quả tổng quan và chọn k

| File | Dùng để làm gì? |
|---|---|
| `comparative_summary.csv` | Bảng A–D ngắn gọn: k, inertia, silhouette, Davies–Bouldin, stability và mức phân tách churn. D là trung bình các seed. |
| `comparison.csv` | Metric chi tiết của từng lần chạy, gồm từng seed D và các thay đổi trọng số/nhóm đặc trưng. |
| `common_k_comparison.csv` | So sánh A/B/C và mọi seed D khi cùng k; đây cũng là bảng đối chứng hoán vị đầy đủ. |
| `k_selection.csv` | Metric cho mọi k và thí nghiệm, gồm cờ `selected`/fallback; lọc theo `experiment` để xem từng trường hợp. |
| `k_selection.png` | Một hình gộp đường inertia và silhouette của A/B/C và trung bình D; đường đứt đánh dấu k được chọn. |

Silhouette/DB/inertia mô tả hình học trong từng không gian đặc trưng, không tự
chứng minh một nhóm đặc trưng tốt hơn nhóm khác. Churn chỉ dùng sau khi phân cụm.

## 2. Khách hàng, personas và churn

| File | Dùng để làm gì? |
|---|---|
| `cluster_assignments.csv` | Một dòng mỗi khách hàng: ID, partition, cụm của từng thí nghiệm và trạng thái chuyển A→C sau ghép nhãn. |
| `cluster_sizes.csv` | Số lượng/tỷ lệ khách hàng trong mỗi cụm, cho cả train/validation/test và mọi thí nghiệm. |
| `cluster_profiles.csv` | Đặc trưng đại diện của cụm trên train: trung bình, trung vị, phân vị, tỷ lệ danh mục và độ lệch chuẩn hóa. |
| `personas.csv` | Tên và đặc điểm cụm A/B/C, hành động đề xuất và churn trên test. |
| `churn_by_cluster.csv` | Số khách hàng, số churn, tỷ lệ churn, lift và khoảng tin cậy của từng cụm trên test, tại k được chọn. |
| `churn_common_k.csv` | Cùng các chỉ số churn nhưng tại common k; gộp A/B/C và mọi seed D vào một bảng. |
| `cluster_sizes.png` | Tỷ lệ cụm trên test cho A/B/C và seed D đầu tiên; xem đủ các seed trong `cluster_sizes.csv`. |
| `churn_by_cluster.png` | Churn theo cụm A/B/C, khoảng tin cậy 95% và mức churn chung. |

## 3. So sánh cấu trúc cụm

| File | Dùng để làm gì? |
|---|---|
| `agreement.csv` | ARI/NMI cho A–B, A–C, B–C; đo mức giống nhau của cách phân cụm, không phụ thuộc tên nhãn. |
| `cluster_transitions.csv` | Số khách hàng và tỷ lệ đi từ từng cụm nguồn sang cụm đích cho ba cặp thí nghiệm. Gộp bảng đếm và bảng tỷ lệ cũ. |
| `cluster_comparison.png` | Ba heatmap trong một hình, thể hiện số lượng và tỷ lệ chuyển/đối chiếu cụm. |
| `original_distribution_profiles.csv` | Thành phần thuộc tính gốc trong từng cụm: phân vị, độ lệch chuẩn hóa và KS. Dữ liệu gốc không đổi, chỉ cách nhóm thay đổi. |
| `original_feature_distributions.png` | Phân phối tuổi, điểm tín dụng, số dư, số sản phẩm, hoạt động và quốc gia trong cụm A so với C. |

## 4. Kiểm tra dữ liệu synthetic

| File | Dùng để làm gì? |
|---|---|
| `feature_provenance.csv` | Nguồn gốc, kiểu dữ liệu, phụ thuộc và lý do loại các đặc trưng. |
| `synthetic_distributions.csv` | Phân phối, phân vị, số giá trị khác nhau và tỷ lệ thiếu của các đặc trưng hành vi trên train. |
| `synthetic_original_correlations.csv` | Tương quan Spearman giữa đặc trưng gốc và hành vi; giúp phát hiện thông tin lặp lại/phụ thuộc. |
| `activity_conditioning.csv` | Kiểm tra giao dịch synthetic theo cờ `active_member` gốc để nhận diện tác động của quy tắc sinh dữ liệu. |
| `permutation_donors.csv` | Ánh xạ người nhận–người cho hàng hành vi của D; cần để tái lập đúng dữ liệu đầu vào cho model D đã lưu. |
| `permutation_control.png` | So sánh mức phân tách churn của A/C với các đối chứng D tại cùng k; không chứng minh dữ liệu synthetic sát thực tế. |

## 5. Hướng dẫn và tái lập

| File | Dùng để làm gì? |
|---|---|
| `README.md` | Bản hướng dẫn này được chép vào thư mục kết quả sau mỗi lần chạy. |
| `report.md` | Diễn giải chi tiết của lần chạy: phương pháp, kết quả, personas và giới hạn. Báo cáo ngắn nằm ở `docs/segmentation.md` trong repo. |
| `run_manifest.json` | Settings, seed, partition, phiên bản thư viện, Git revision, đặc trưng và hash nguồn dữ liệu để tái lập. |

Model `.joblib` nằm riêng trong `models/saved/segmentation/`.
CSV kết quả và biểu đồ không được commit lên GitHub.
