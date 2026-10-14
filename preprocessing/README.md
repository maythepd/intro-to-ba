# Preprocessing & Feature Engineering Package

Mô-đun hóa toàn bộ quy trình biến đổi dữ liệu và kỹ nghệ đặc trưng (Feature Engineering) từ notebook `04_Data_Transformation.ipynb` cho bài toán **Customer Churn Prediction**.

---

## 1. Cấu trúc thư mục

```
intro-to-ba/
├── config/
│   ├── __init__.py                # Export PipelineConfig và load_config
│   ├── pipeline_config.py         # Dataclass cấu hình & load YAML
│   └── feature_engineering.yaml   # FILE CẤU HÌNH YAML CHÍNH (Sửa true/false để bật/tắt features)
├── preprocessing/
│   ├── __init__.py                # Export các transformer và pipeline
│   ├── base.py                    # BaseFeatureTransformer (chuẩn fit/transform)
│   ├── skewness_transform.py      # SkewnessTransformer: Log1p, zero-inflation hurdle, Yeo-Johnson
│   ├── scaling.py                 # FeatureScaler: RobustScaler & StandardScaler
│   ├── domain_features.py         # DomainFeatureEngineer: Momentum, Friction, Bundling, Interaction
│   ├── categorical_encoding.py    # CategoricalEncoder: One-Hot & Smoothed Target Encoding (m-estimate)
│   ├── balancing.py               # ClassBalancer: SMOTE resampling cho tập Train
│   ├── feature_selector.py        # FeatureSelector: Xếp hạng Mutual Info / Random Forest & lọc Top-K
│   ├── pipeline.py                # FeatureEngineeringPipeline: Lớp điều phối tổng thể
│   └── README.md                  # Hướng dẫn chi tiết
└── test_pipeline.py               # Script kiểm thử và demo
```

---

## 2. File cấu hình YAML: `config/feature_engineering.yaml`

Bạn chỉ cần mở file `config/feature_engineering.yaml` và đổi giá trị giữa `true` và `false` để chạy các thử nghiệm ablation:

```yaml
# 1. BẬT / TẮT CÁC MÔ-ĐUN CHÍNH (Ablations)
modules:
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
```

---

## 3. Hướng dẫn sử dụng

### 3.1. Chạy với cấu hình YAML mặc định

`FeatureEngineeringPipeline` tự động tìm và đọc cấu hình từ file `config/feature_engineering.yaml`:

```python
import pandas as pd
from preprocessing import FeatureEngineeringPipeline

# Đọc dữ liệu sạch
df = pd.read_csv("artifacts/cleaned_bank_data.csv")

# Khởi tạo pipeline (tự động nạp config/feature_engineering.yaml)
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

### 3.2. Truyền file YAML tùy chỉnh hoặc đường dẫn khác

```python
pipe = FeatureEngineeringPipeline("path/to/my_ablation_experiment.yaml")
result = pipe.prepare_train_test(df)
```

### 3.3. Ghi đè cấu hình qua code Python khi cần

```python
from config import PipelineConfig
from preprocessing import FeatureEngineeringPipeline

# Tạo cấu hình với các tham số ghi đè
cfg = PipelineConfig(
    enable_domain_features=False, # Tắt đặc trưng nghiệp vụ
    enable_smote=False,           # Tắt SMOTE
)

pipe = FeatureEngineeringPipeline(cfg)
res = pipe.prepare_train_test(df)
```

### 3.4. Xuất các file Artifacts

```python
pipe = FeatureEngineeringPipeline()
res = pipe.prepare_train_test(df)

# Xuất ra thư mục artifacts giống notebook 04
pipe.export_artifacts(res, output_dir="artifacts")
```
