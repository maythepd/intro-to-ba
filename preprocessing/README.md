# Preprocessing & Feature Engineering Package

Mô-đun hóa toàn bộ quy trình biến đổi dữ liệu và kỹ nghệ đặc trưng (Feature Engineering) từ notebook `04_Data_Transformation.ipynb` cho bài toán **Customer Churn Prediction**.

---

## 1. Cấu trúc thư mục

```
preprocessing/
├── __init__.py                    # Export các transformer, processor và pipeline chính
├── base.py                        # BaseFeatureTransformer (chuẩn fit/transform & BaseManager)
├── pipeline.py                    # FeatureEngineeringPipeline & load_dataset: Điều phối toàn bộ quy trình
│
├── domain/                        # Feature Engineering nghiệp vụ ngân hàng
│   ├── __init__.py
│   ├── bundling.py                # BundlingFeatureEngineer (sản phẩm dịch vụ)
│   ├── friction.py                # FrictionFeatureEngineer (khiếu nại, sự cố)
│   ├── interactions.py            # InteractionFeatureEngineer (tương tác chéo)
│   ├── momentum.py                # MomentumFeatureEngineer (xu hướng giao dịch)
│   └── manager.py                 # DomainFeatureManager tổng hợp các module domain
│
├── gnn/                           # Graph Neural Network Embeddings
│   ├── __init__.py
│   ├── graph_builder.py           # CustomerGraphBuilder (xây dựng đồ thị khách hàng)
│   ├── models.py                  # GraphSAGE / GAT architectures
│   └── manager.py                 # GNNEmbeddingManager trích xuất vector không gian
│
├── ingestion/                     # Nhập liệu & Phân chia tập dữ liệu
│   ├── __init__.py
│   ├── etl.py                     # BankDataETL: Ingest & aggregate 7 bảng quan hệ từ full_dataset
│   ├── kaggle_downloader.py       # Tải dataset từ Kaggle
│   └── splitting.py               # DataSplitter: Stratified Train/Val/Test split & K-Fold CV
│
├── quant_signals/                 # Tín hiệu định lượng & vi cấu trúc tài chính
│   ├── __init__.py
│   ├── entropy.py                 # EntropySignals (độ hỗn loạn hành vi)
│   ├── microstructure.py          # MicrostructureSignals (nhịp độ giao dịch)
│   ├── statistical_arbitrage.py   # StatisticalArbitrageSignals (lệch chuẩn chi tiêu)
│   └── manager.py                 # QuantSignalManager
│
├── selection/                     # Tuyển chọn đặc trưng (Feature Selection)
│   ├── __init__.py
│   ├── correlation.py             # CorrelationSelector (lọc đa cộng tuyến)
│   ├── importance.py              # ImportanceSelector (Mutual Info, RF Importance)
│   └── manager.py                 # FeatureSelectionManager
│
├── temporal/                      # Trích xuất chuỗi thời gian & Attention
│   ├── __init__.py
│   ├── manager.py                 # TemporalSequenceManager
│   ├── model.py                   # PyTorch Temporal Sequence Model
│   ├── raw_extractor.py           # Trích xuất sự kiện chuỗi thô
│   ├── extractors/                # Extractors chuyên biệt (transaction, login, complaint, account)
│   ├── modules/                   # Attention blocks & Loss functions
│   └── strategies/                # ArcFace, Poincare, Optimal Transport, Neyman-Pearson
│
└── transforms/                    # Các biến đổi toán học & mã hóa
    ├── __init__.py
    ├── categorical_encoding.py    # CategoricalEncoder: One-Hot & Smoothed Target Encoding
    ├── scaling.py                 # FeatureScaler: RobustScaler & StandardScaler
    └── skewness_transform.py      # SkewnessTransformer: Log1p, zero-inflation hurdle, Yeo-Johnson
```

---

## 2. File cấu hình YAML duy nhất: `config/config.yaml`

Tất cả cấu hình được gom vào `config/config.yaml`. Bạn chỉ cần mở file này và sửa `true` / `false` để chạy các thử nghiệm ablation:

```yaml
dataset:
  data_path: "data/cleaned_bank_data.csv"
  target_col: "churn"

features:
  # 1. BẬT / TẮT CÁC MÔ-ĐUN CHÍNH (Ablations)
  enable_skewness: true             # Biến đổi Log1p, Zero-balance hurdle, Yeo-Johnson
  enable_scaling: true              # Chuẩn hóa RobustScaler & StandardScaler
  enable_domain_features: true      # Tạo các đặc trưng nghiệp vụ ngân hàng
  enable_one_hot_encoding: true     # One-Hot Encoding cho các cột phân loại
  enable_target_encoding: true      # Smoothed Target Encoding (m-estimate)
  enable_smote: true                # Cân bằng mẫu thiểu số bằng SMOTE trên tập Train
  enable_feature_selection: false   # Lọc đặc trưng Top-K theo độ quan trọng

  # 2. CHI TIẾT ĐẶC TRƯNG NGHIỆP VỤ
  domain_features:
    enable_behavioral_momentum: true  # velocity, spending collapsing, ticket size, app logins ratio
    enable_customer_friction: true    # friction_index, is_high_friction
    enable_product_bundling: true     # is_product_sweetspot, is_product_danger_zone
    enable_risk_interactions: true    # wealth_deceleration_risk, germany_high_balance, middle-aged

model:
  name: "xgboost"
  params:
    learning_rate: 0.05
    max_depth: 5
```

---

## 3. Hướng dẫn sử dụng

### 3.1. Chạy với cấu hình YAML mặc định

`FeatureEngineeringPipeline` tự động tìm và đọc cấu hình từ file `config/config.yaml`:

```python
import pandas as pd
from preprocessing import FeatureEngineeringPipeline

# Đọc dữ liệu sạch
df = pd.read_csv("data/cleaned_bank_data.csv")

# Khởi tạo pipeline (tự động nạp config/config.yaml)
pipe = FeatureEngineeringPipeline()
result = pipe.prepare_train_test(df)

# Kết quả sẵn sàng huấn luyện mô hình
X_train = result.X_train               # (8000, 55)
X_test = result.X_test                 # (2000, 55)
y_train = result.y_train               # (8000,)
y_test = result.y_test                 # (2000,)
X_train_smote = result.X_train_resampled  # (12740, 55)
y_train_smote = result.y_train_resampled  # (12740,)
```

### 3.2. Chạy ablation hoặc huấn luyện qua main.py

```bash
# Chạy đầy đủ 7 kịch bản benchmark và xuất báo cáo results/xgboost_ablation_report.md
python main.py --mode ablation

# Huấn luyện 1 mô hình đơn lẻ
python main.py --mode train
```

### 3.3. Xuất các file Artifacts

```python
pipe = FeatureEngineeringPipeline()
res = pipe.prepare_train_test(df)

# Xuất ra thư mục artifacts giống notebook 04
pipe.export_artifacts(res, output_dir="artifacts")
```
