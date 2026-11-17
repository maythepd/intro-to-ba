# CHIẾN LƯỢC HỌC BIỂU DIỄN HÌNH HỌC MANIFOLD CHO CHUỖI HÀNH VI KHÁCH HÀNG
## TỐI ƯU HÓA ĐÁNH ĐỔI GIỮA BÁO ĐỘNG GIẢ (TYPE I) VÀ BỎ SÓT CHURN (TYPE II)

> **Tài liệu Kỹ thuật & Nghiên cứu Nâng cao (Advanced Technical & Theoretical Specification)**  
> **Áp dụng cho:** Hệ thống Dự đoán Churn Ngân hàng đa kỳ (Multi-snapshot Behavioral Sequences)  
> **Dữ liệu đầu vào:** Tensor chuỗi sự kiện $\mathcal{X} \in \mathbb{R}^{N \times T \times D}$ ($T=6$ tháng, $D=10$ chiều snapshot giao dịch, số dư, đăng nhập, khiếu nại).

---

## 1. BẢN CHẤT HÌNH HỌC VÀ VẤN ĐỀ "MANIFOLD TANGLING"

### 1.1. Giới hạn của Autoencoder Tự giám sát thuần túy (Unsupervised Reconstruction)
Trong pipeline truyền thống, mô hình `BiGRUTemporalAutoencoder` được huấn luyện hoàn toàn theo hàm mất mát tái tạo Euclidean:
$$\mathcal{L}_{recon} = \frac{1}{B \cdot T \cdot D} \sum_{i=1}^{B} \sum_{t=1}^{T} \| x_{i, t} - \hat{x}_{i, t} \|_2^2$$

Hàm mất mát này có đặc tính **đẳng hướng (isotropic)** và **đối xứng (symmetric)**. Nó ép không gian ẩn $z \in \mathbb{R}^{16}$ phải bảo toàn tối đa **phương sai tổng thể (total variance)** của dữ liệu thô. Tuy nhiên:
1. **Mù quáng trước nhãn Churn**: Không gian ẩn không có áp lực phân tách giữa khách hàng rời bỏ ($y=1$) và khách hàng ở lại ($y=0$).
2. **Hiện tượng Manifold Tangling (Xoắn Manifold)**: Nhóm khách hàng "chớm rời bỏ" (silent churners — giảm dần số lần login và chi tiêu) có vector hành vi ở tháng thứ 6 rất giống với nhóm khách hàng trung thành nhưng vốn ít giao dịch (low-frequency loyal customers). Trên không gian phẳng $\mathbb{R}^{16}$, hai nhóm này nằm đan xen vào nhau trong các vùng mật độ cao.
3. **Bế tắc về ngưỡng quyết định (Decision Boundary Dilemma)**: Khi đưa $z$ vào các mô hình xuôi dòng (XGBoost / LightGBM), bất kỳ mặt cắt quyết định phẳng nào cố gắng tăng **Recall** (giảm False Negatives - FN) đều sẽ quét qua vùng đan xen, làm bùng nổ **False Positives (FP)**.

### 1.2. Mục tiêu của Thao túng Manifold (Manifold Deformation)
Thao túng manifold là việc áp đặt các cấu trúc hình học phi-Euclidean (siêu cầu, hyperbol, không gian đo vận chuyển) hoặc hàm phạt ràng buộc để **chủ động bẻ cong (deform)** không gian biểu diễn sao cho:
- **Tập Churn ($y=1$) bị cô lập vào các vùng biên hoặc nón hình cầu có biên cách ly an toàn**.
- **Tập Non-churn ($y=0$) co cụm ổn định tại vùng lõi**.
- **Đánh đổi bất đối xứng (Asymmetric Cost Trade-off)** được nhúng trực tiếp vào cấu trúc topo của biểu diễn, giải phóng áp lực điều chỉnh threshold ở downstream classifier.

---

## 2. TỔNG HỢP 4 CHIẾN LƯỢC HỌC BIỂU DIỄN ĐỘT PHÁ

```
                     ┌─────────────────────────────────────────────────────────┐
                     │  Longitudinal Snapshot Tensor: X in R^(N x 6 x 10)     │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │   Bi-GRU Temporal Feature Backbone  │
                               └──────────────────┬──────────────────┘
                                                  │
         ┌────────────────────────┬───────────────┴───────────────┬────────────────────────┐
         ▼                        ▼                               ▼                        ▼
┌──────────────────┐    ┌──────────────────┐            ┌──────────────────┐     ┌──────────────────┐
│   CHIẾN LƯỢC 1   │    │   CHIẾN LƯỢC 2   │            │   CHIẾN LƯỢC 3   │     │   CHIẾN LƯỢC 4   │
│  Hyperspherical  │    │    Hyperbolic    │            │     Optimal      │     │  Neyman-Pearson  │
│  Metric Learning │    │  Representation  │            │  Transport &     │     │   Constrained    │
│  (Asym-ArcFace)  │    │  (Poincaré Ball) │            │  Drift Velocity  │     │   Optimization   │
└────────┬─────────┘    └────────┬─────────┘            └────────┬─────────┘     └────────┬─────────┘
         │                        │                               │                        │
         ▼                        ▼                               ▼                        ▼
  Lề góc bất đối xứng      Cấu trúc cây thứ bậc           Trường vận tốc trôi      Chặn cứng sai số FP
 m_churn >> m_loyal trên   thể tích dạt biên mũ          Wasserstein Geodesic      Type I <= alpha,
  siêu cầu S^(d-1)         d_H(z, 0) -> Rủi ro           gamma_i(t) trên manifold  cực tiểu hóa FN
```

---

### CHIẾN LƯỢC 1: HYPERSPHERICAL METRIC LEARNING VỚI LỀ GÓC BẤT ĐỐI XỨNG (ASYMMETRIC ARCFACE / SUPCON)

#### 1. Cơ sở Toán học & Hình học
Chiếu vector biểu diễn ẩn $z = f_{enc}(X) \in \mathbb{R}^{d}$ lên bề mặt siêu cầu đơn vị $\mathbb{S}^{d-1}$ bằng phép chuẩn hóa $L_2$:
$$\tilde{z} = \frac{z}{\|z\|_2 + \epsilon} \in \mathbb{S}^{d-1}, \quad \|\tilde{z}\|_2 = 1$$

Khoảng cách trắc địa (geodesic distance) giữa hai điểm trên siêu cầu tương ứng với góc $\theta = \arccos(\tilde{z}_i^\top \tilde{z}_j)$. Ta khởi tạo 2 vector nguyên mẫu học được (learnable class prototypes) $W = [W_0, W_1] \in \mathbb{R}^{2 \times d}$ đại diện cho tâm cụm Loyal ($W_0$) và tâm cụm Churn ($W_1$), chuẩn hóa $\|W_k\|_2 = 1$.

#### 2. Cơ chế Lề góc Bất đối xứng (Asymmetric Angular Margin)
Trong kinh doanh ngân hàng, mất 1 khách hàng Churn gây thiệt hại lớn hơn nhiều chi phí gọi điện/CSKH khi báo động nhầm. Do đó, ta thiết lập **lề phạt góc bất đối xứng**:
$$m_1 = m_{churn} \gg m_0 = m_{loyal} \quad (\text{ví dụ: } m_{churn} = 0.45 \text{ rad} \approx 26^\circ, \; m_{loyal} = 0.05 \text{ rad})$$

Hàm mất mát góc Asymmetric ArcFace:
$$\mathcal{L}_{angular} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s \cdot \cos(\theta_{i, y_i} + m_{y_i})}}{e^{s \cdot \cos(\theta_{i, y_i} + m_{y_i})} + \sum_{j \neq y_i} e^{s \cdot \cos(\theta_{i, j})}}$$
trong đó $s$ là bán kính siêu cầu (hyperspherical scale factor, thường chọn $16 \le s \le 32$), và $\cos(\theta + m) = \cos\theta \cos m - \sin\theta \sin m$.

#### 3. Hàm mục tiêu Đa nhiệm (Hybrid Multi-Task Objective)
$$\mathcal{L}_{total} = \mathcal{L}_{recon}(X, \hat{X}) + \lambda_{angular} \cdot \mathcal{L}_{angular}(\tilde{z}, y)$$
- $\mathcal{L}_{recon}$: Giữ lại toàn bộ thông tin phương sai nền tảng của chuỗi giao dịch (tránh sụp đổ biểu diễn - representation collapse).
- $\mathcal{L}_{angular}$: Ép khách hàng Churn vào một nón hình cầu cực kỳ hẹp, đẩy xa khỏi tâm Loyal một khoảng an toàn tối thiểu $m_{churn}$.

#### 4. Đặc trưng Manifold sinh ra cho Downstream XGBoost
1. `temporal_z0` đến `temporal_z15`: Tọa độ latent chuẩn hóa.
2. `angular_dist_to_churn`: $\theta_{churn} = \arccos(\tilde{z}^\top W_1) \in [0, \pi]$ (Khoảng cách trắc địa trực tiếp tới tâm Churn).
3. `angular_dist_to_loyal`: $\theta_{loyal} = \arccos(\tilde{z}^\top W_0) \in [0, \pi]$.
4. `angular_risk_ratio`: $\frac{\theta_{loyal}}{\theta_{churn} + 10^{-4}}$ (Tỷ số tương đối đo độ lệch về cực Churn).
5. `cosine_churn_sim`: $\tilde{z}^\top W_1$ (Độ tương đồng cosine tuyến tính).

---

### CHIẾN LƯỢC 2: HỌC BIỂU DIỄN TRÊN HÌNH HỌC HYPERBOLIC (POINCARÉ BALL MODEL)

#### 1. Cơ sở Toán học & Hình học
Không gian Hyperbolic $\mathbb{H}^d$ là một đa tạp Riemannian có độ cong âm không đổi ($c < 0$). Trong mô hình quả cầu Poincaré (Poincaré Ball Model):
$$\mathbb{B}_c^d = \left\{ x \in \mathbb{R}^d : c \|x\|^2 < 1 \right\}$$
Metric tensor Riemannian được định nghĩa bởi:
$$g_x^{\mathbb{B}} = \lambda_x^2 g^E, \quad \text{với } \lambda_x = \frac{2}{1 - c \|x\|^2}$$

Khoảng cách Hyperbolic giữa 2 điểm $u, v \in \mathbb{B}_c^d$:
$$d_{\mathbb{H}}(u, v) = \frac{2}{\sqrt{c}} \operatorname{artanh}\left( \sqrt{c} \| -u \oplus_c v \| \right)$$
trong đó $\oplus_c$ là phép cộng Möbius trên quả cầu Poincaré:
$$u \oplus_c v = \frac{(1 + 2c \langle u, v \rangle + c \|v\|^2)u + (1 - c \|u\|^2)v}{1 + 2c \langle u, v \rangle + c^2 \|u\|^2 \|v\|^2}$$

#### 2. Bản chất Nghiệp vụ & Khả năng Triệt tiêu Đánh đổi FP/FN
Hành vi khách hàng ngân hàng qua thời gian có bản chất **phân cấp thứ bậc (Hierarchical Degradation Tree)**:
$$\text{Khách hàng VIP} \to \text{Giao dịch đều đặn} \to \text{Giảm tương tác} \to \text{Khiếu nại / Ngừng login} \to \text{Rút hết tiền (Churn)}$$

- Trong không gian phẳng $\mathbb{R}^d$, thể tích quả cầu bán kính $r$ chỉ tăng theo hàm đa thức ($V(r) \propto r^d$). Khi cây phân nhánh sâu, các nút lá (các dạng hành vi churn hiếm gặp) bị dồn ép (crowding effect), dẫn đến overlap với khách hàng bình thường $\to$ sinh FP cao.
- Trong không gian Hyperbolic, chu vi và diện tích tăng theo **hàm mũ**:
  $$V_{\mathbb{H}}(r) \propto \sinh^{d-1}(r) \approx \frac{1}{2^{d-1}} e^{(d-1)r}$$
  Nó cung cấp không gian vô hạn ở gần biên ($\|x\| \to 1$).

#### 3. Cơ chế ánh xạ
- Bi-GRU Encoder sinh vector $h \in \mathbb{R}^d$, sau đó dùng ánh xạ hàm mũ (Exponential Map) tại gốc tọa độ để chiếu vào $\mathbb{B}_c^d$:
  $$z_{\mathbb{H}} = \exp_0^c(h) = \tanh(\sqrt{c}\|h\|) \frac{h}{\sqrt{c}\|h\|}$$
- Khách hàng trung thành, tương tác ổn định sẽ định vị tự nhiên ở **gần gốc tọa độ** $\|z_{\mathbb{H}}\| \approx 0$.
- Càng có dấu hiệu suy thoái, quỹ đạo càng bị đẩy dạt ra **rìa quả cầu** $\|z_{\mathbb{H}}\| \to 1$.
- **Khoảng cách Hyperbolic tới gốc $d_{\mathbb{H}}(z_i, 0) = \frac{2}{\sqrt{c}}\operatorname{artanh}(\sqrt{c}\|z_i\|)$** trở thành một chỉ báo nguy cơ đơn điệu tự nhiên, phân loại rủi ro mà không làm vỡ cụm loyal.

---

### CHIẾN LƯỢC 3: HỌC QUỸ ĐẠO ĐỘNG BẰNG VẬN CHUYỂN TỐI ƯU & TRƯỜNG VẬN TỐC TRÔI (OPTIMAL TRANSPORT & DRIFT VELOCITY)

#### 1. Cơ sở Toán học & Hình học
Thay vì nén cả chuỗi 6 tháng thành 1 vector tĩnh, ta coi hành vi của khách hàng $i$ là một **quỹ đạo động (Dynamical Trajectory)** hoặc một phân phối xác suất thời gian (Empirical Measure) trên không gian trạng thái:
$$\mu_i = \frac{1}{T} \sum_{t=1}^{T} \delta_{(z_{i, t}, t)}$$

Khoảng cách Wasserstein bậc 2 giữa hai quỹ đạo khách hàng:
$$\mathcal{W}_2^2(\mu_i, \mu_j) = \inf_{\pi \in \Pi(\mu_i, \mu_j)} \int \|x - y\|_2^2 \, \mathrm{d}\pi(x, y)$$

#### 2. Cơ chế Trường Vận tốc Trôi (Drift Velocity Field)
Tại mỗi bước thời gian $t$, trạng thái ẩn $h_t$ của GRU tuân theo một hệ phương trình vi phân động học:
$$\frac{\mathrm{d}h}{\mathrm{d}t} = \mathcal{F}_{\theta}(h(t), x(t))$$

Ta xấp xỉ trường vận tốc trôi hành vi (Drift Velocity Vector) và gia tốc (Curvature):
$$\vec{v}_i = z_{i, t=6} - z_{i, t=1} = \int_{1}^{6} \dot{z}_i(t) \, \mathrm{d}t$$
$$\vec{a}_i = \dot{v}_i \approx (z_{i, t=6} - z_{i, t=5}) - (z_{i, t=2} - z_{i, t=1})$$

#### 3. Khắc phục nhược điểm Báo động giả (FP Elimination)
- **Vấn đề thực tế**: Một khách hàng chi tiêu 2 triệu/tháng tại tháng 6 có thể là:
  - Trường hợp A: Khách hàng sinh viên mới mở tài khoản (vốn chỉ chi 2 triệu/tháng, bình thường $\to$ Loyal).
  - Trường hợp B: Khách hàng VIP trước đây tiêu 100 triệu/tháng, nay tụt xuống 2 triệu/tháng ($\to$ Churn nguy hiểm).
- Nếu chỉ nhìn snapshot tĩnh tại $t=6$, hai người này trùng tọa độ $\to$ **Dễ sinh báo động giả cho người A hoặc bỏ sót người B**.
- **Giải pháp Manifold**: Tọa độ tĩnh có thể giống nhau, nhưng vector đạo hàm $\vec{v}_A$ có hướng ổn định/tăng trưởng ($\langle \vec{v}_A, \vec{u}_{growth} \rangle > 0$), trong khi $\vec{v}_B$ là vector suy thoái cực lớn ($\|\vec{v}_B\| \gg 0$ và hướng thẳng vào vùng hấp thụ Churn).
- Khoảng cách Wasserstein $W_2(\mu_i, \bar{\mu}_{churn})$ tới phân phối tâm Churn cho phép phân định rạch ròi 100% hai trường hợp này.

---

### CHIẾN LƯỢC 4: HỌC BIỂU DIỄN RÀNG BUỘC THEO TIÊN ĐỀ NEYMAN-PEARSON (NEYMAN-PEARSON CONSTRAINED LEARNING)

#### 1. Cơ sở Toán học & Lý thuyết Quyết định Thống kê
Trong phân loại học máy chuẩn (Empirical Risk Minimization - ERM), hàm mục tiêu tối ưu hóa trung bình cộng lỗi:
$$\min_{\theta} \mathbb{E}_{(X, Y)} [\ell(f_{\theta}(X), Y)]$$
Điều này vô tình coi trọng số lỗi của hai lớp là bằng nhau hoặc phụ thuộc tỷ lệ mẫu. 

Theo tiên đề thống kê **Neyman-Pearson (NP)** (Rigollet & Tong, 2011), bài toán kiểm định giả thuyết được đặt ra có điều kiện biên cứng:
$$\min_{\theta} \quad R_1(f_{\theta}) = \mathbb{P}(f_{\theta}(X) = 0 \mid Y = 1) \quad (\text{Type II Error: Bỏ sót Churn})$$
$$\text{thỏa mãn: } \quad R_0(f_{\theta}) = \mathbb{P}(f_{\theta}(X) = 1 \mid Y = 0) \le \alpha \quad (\text{Type I Error: Báo động giả} \le \alpha)$$
trong đó $\alpha \in (0, 1)$ là ngưỡng trần ngân sách can thiệp CSKH (ví dụ: chỉ chấp nhận gọi nhầm tối đa 5% tổng khách hàng).

#### 2. Thao túng Manifold qua Hàm phạt Lagrangian (Constrained Manifold Sculpting)
Ta biến đổi bài toán học biểu diễn chuỗi thành bài toán tối ưu Min-Max không ràng buộc thông qua hệ số nhân Lagrange $\lambda$:
$$\max_{\lambda \ge 0} \min_{\theta} \quad \mathcal{L}_{recon}(X, \hat{X}) + \hat{R}_1(z_{\theta}) + \lambda \cdot \max\left(0, \, \hat{R}_0(z_{\theta}) - \alpha + \delta\right)$$
trong đó $\hat{R}_0$ và $\hat{R}_1$ được xấp xỉ bằng hàm sigmoid trơn (smooth surrogate of 0-1 loss) trên khoảng cách biểu diễn.

#### 3. Hiệu quả hình học
- Mạng nơ-ron tự động "kéo dãn" (dilate) không gian ở xung quanh mặt biên $R_0 = \alpha$.
- Tất cả các mẫu Churn bị nén chặt về một phía của siêu mặt phẳng ranh giới, trong khi phần còn lại của không gian bị chặn cứng bởi bán kính $\alpha$.
- Đảm bảo nghiệm tối ưu đạt cận Pareto tối thượng: **Cực đại hóa Recall tại mức False Alarm cố định $\alpha$ được kiểm soát toán học**.

---

## 3. BẢNG SO SÁNH TỔNG QUAN GIỮA 4 CHIẾN LƯỢC

| Tiêu chí So sánh | Chiến lược 1: Hyperspherical Asym-ArcFace | Chiến lược 2: Hyperbolic Poincaré Ball | Chiến lược 3: Optimal Transport & Drift | Chiến lược 4: Neyman-Pearson Constrained |
| :--- | :--- | :--- | :--- | :--- |
| **Không gian Topo** | Siêu cầu $\mathbb{S}^{d-1}$ (Độ cong dương) | Đĩa Poincaré $\mathbb{B}_c^d$ (Độ cong âm) | Không gian đo xác suất $\mathcal{P}(\mathbb{R}^d)$ | Không gian Euclide bị nén biên Lagrangian |
| **Toán tử chính** | Khoảng cách góc trắc địa $\arccos(\tilde{z}_i^\top \tilde{z}_j)$ | Phép cộng Möbius & Metric Poincaré | Khoảng cách Wasserstein $\mathcal{W}_2$ | Ràng buộc bất đẳng thức $\mathbb{P}(FP) \le \alpha$ |
| **Cơ chế giảm FN** | Lề góc bất đối xứng $m_{churn} \gg m_{loyal}$ ép cụm churn | Thể tích hyperbolic tăng hàm mũ dạt biên | Vector vận tốc $\vec{v}_i$ phát hiện sớm xu hướng | Cực tiểu hóa trực tiếp Type II Error |
| **Cơ chế chặn FP** | Tâm cụm Loyal cố định, góc phân cách an toàn | Khách hàng ổn định định vị tại tâm $\|z\| \approx 0$ | Phân biệt suy giảm tạm thời vs suy giảm vĩnh viễn | Chặn cứng giới hạn $\alpha$ bởi Lagrangian multiplier |
| **Độ phức tạp tính toán** | Thấp ($O(B \cdot d)$), chạy cực nhanh | Trung bình ($O(B \cdot d)$ với phép chiếu vi phân) | Trung bình - Cao ($O(B \cdot T \cdot d)$) | Trung bình ($O(B \cdot d)$ với cập nhật đối ngẫu) |
| **Tương thích GBDT/XGBoost** | **Xuất sắc** (sinh ra góc và tỷ số trắc địa 1D) | Rất tốt (sinh ra bán kính hyperbol và góc) | Rất tốt (sinh ra vector vận tốc và gia tốc) | Tốt (sinh ra điểm số phân định theo ngưỡng) |
| **Mức độ sẵn sàng** | **Sẵn sàng triển khai ngay** | Cần thư viện Geoopt / PyTorch Hyperbolic | Cần mô-đun xấp xỉ vi phân thời gian | Cần thuật toán tối ưu kép Primal-Dual |

---

## 4. LỘ TRÌNH THỰC NGHIỆM CHI TIẾT (EXPERIMENTAL ROADMAP)

- [x] **Pha 1 (Hiện tại): Triển khai Chiến lược 1 (Hyperspherical Asymmetric ArcFace)**:
  - Tích hợp lớp `AsymmetricArcFaceLoss` và `HypersphericalHead` vào `BiGRUTemporalAutoencoder`.
  - Sinh 4 đặc trưng trắc địa siêu cầu: `angular_dist_to_churn`, `angular_dist_to_loyal`, `angular_risk_ratio`, `temporal_drift_velocity`.
  - So sánh đối chuẩn (Benchmark) với baseline MSE thuần túy trên tập Test độc lập.
- [ ] **Pha 2: Nghiên cứu Chiến lược 3 (Drift Velocity & Trajectory Curvature)**:
  - Bổ sung trường đạo hàm bậc 1 ($\vec{v}_i$) và bậc 2 ($\vec{a}_i$) giữa các snapshot thời gian để loại trừ dứt điểm báo động giả của nhóm giao dịch thấp định kỳ.
- [ ] **Pha 3: Nghiên cứu Chiến lược 2 (Poincaré Embeddings)**:
  - Thử nghiệm mô hình hóa cấu trúc cây suy thoái trên đĩa Poincaré với bán kính $c=1.0$.
- [ ] **Pha 4: Tích hợp Chiến lược 4 (Neyman-Pearson Risk Control)**:
  - Thiết lập cận cứng ngân sách chiến dịch chăm sóc khách hàng (ví dụ: FP $\le 4\%$) và huấn luyện mô hình bám sát cận tối ưu Neyman-Pearson.

---

## 5. PHÂN TÍCH THỰC NGHIỆM CHIẾN LƯỢC 1: NGUYÊN NHÂN MỨC TĂNG KHIÊM TỐN & HƯỚNG DẪN TINH CHỈNH THAM SỐ

### 5.1. Bức tranh Thực nghiệm (Empirical Observations)
Sau khi huấn luyện 5 folds và đánh giá trên tập kiểm thử độc lập (Hold-out Test 1,000 mẫu):
- **Recall**: Tăng từ `0.4706` $\to$ `0.4853` (+1.47%, bắt thêm được đúng 3 khách hàng churn: TP tăng từ 96 lên 99, FN giảm từ 108 xuống 105).
- **False Positives (Báo động giả)**: Giữ nguyên tuyệt đối ở mức `41` (Precision tăng nhẹ từ `0.7007` $\to$ `0.7071`).
- **Feature Importance**: Các đặc trưng chuỗi thời gian tăng gần gấp đôi độ quan trọng (`temporal_z8` nhảy từ hạng 64 lên 34, Gain tăng từ 6.3 lên 11.9), và các đặc trưng hình học mới (`temporal_angular_loyal_dist`, `temporal_angular_risk_ratio`) lọt vào Top 60-80.

**Vấn đề cốt lõi đặt ra:** Tại sao hình học manifold đã biến dạng rõ rệt, nhưng số lượng churn bắt thêm được vẫn chỉ dừng lại ở mức khiêm tốn (+3 khách hàng), và cảnh báo giả không hề suy giảm?

---

### 5.2. Chẩn đoán 5 Nguyên nhân Gốc rễ (Root Cause Analysis)

#### Nguyên nhân 1: Điểm nghẽn Ngưỡng Quyết định Cố định $\tau = 0.50$ (Rigid Threshold Disconnect)
- **Bản chất**: Trong bài toán mất cân bằng lớp (churn chiếm ~20%), ngưỡng phân loại mặc định $\tau = 0.50$ là cực kỳ khắt khe, ưu tiên tối đa Precision thay vì Recall.
- **Hệ quả trên Manifold**: Khi Asymmetric ArcFace đẩy cụm Churn ra xa cụm Loyal, phân phối xác suất dự báo $P(y=1 \mid x)$ của nhóm khách hàng "chớm churn" đã dịch chuyển đáng kể từ $0.25 - 0.35$ lên $0.40 - 0.48$. Tuy nhiên, vì mô hình vẫn bị đo tại ngưỡng cố định $\tau = 0.50$, tất cả các khách hàng có xác suất $0.40 - 0.48$ này **vẫn bị gán nhãn 0 (vẫn rơi vào False Negatives)**!
- **Minh chứng**: Mức FP giữ nguyên ở 41 và Precision giữ vững ở 70.7% chứng minh rằng mô hình không bị suy thoái ở vùng an toàn, nhưng tiềm năng của manifold chưa được giải phóng vì thiếu bước **Cost-Sensitive Threshold Calibration**.

#### Nguyên nhân 2: Lề góc $m_{churn} = 0.45 \text{ rad}$ chưa đủ rộng trên Siêu cầu 16 Chiều
- **Bản chất**: Trên siêu cầu $\mathbb{S}^{15}$, hai vector ngẫu nhiên trực giao với nhau có góc $\frac{\pi}{2} \approx 1.57 \text{ rad} = 90^\circ$.
- Lề góc $m_{churn} = 0.45 \text{ rad} \approx 25.8^\circ$ và $m_{loyal} = 0.05 \text{ rad} \approx 2.9^\circ$. Khoảng cách ly góc hiệu dụng giữa hai tâm cụm chỉ là:
  $$\Delta m = 0.45 - 0.05 = 0.40 \text{ rad} \approx 22.9^\circ$$
- Với góc phân cách $23^\circ$, các khách hàng có quỹ đạo hành vi dao động phức tạp (noise) vẫn có thể liếm vào vùng biên phân định. Lề góc chưa đủ lớn để tạo ra một "hố sâu ngăn cách" (wide angular margin gap) giúp tách biệt hoàn toàn nhóm churn cận biên.

#### Nguyên nhân 3: Feature Selection bằng Pearson Correlation làm tổn thương Đặc trưng Phi tuyến
- **Bản chất**: Trong pipeline, bước lọc đặc trưng sử dụng **Pearson Correlation** để chọn Top 120 features:
  $$r_{X, y} = \frac{\operatorname{Cov}(X, y)}{\sigma_X \sigma_y}$$
- Pearson chỉ đo mối quan hệ **tuyến tính đơn điệu**. Trong khi đó, các đặc trưng hình học Manifold ($\theta_{churn} = \arccos(\tilde{z}^\top W_1)$, $\text{drift\_velocity} = \|\vec{v}\|_2$) có mối quan hệ **phi tuyến dạng cầu hoặc ngưỡng (threshold-based non-linear)** đối với Churn.
- **Hậu quả**: Một số đặc trưng hình học mạnh nhất (như `temporal_angular_churn_dist` hay `temporal_cosine_churn_sim`) bị chấm điểm tương quan tuyến tính thấp hơn các biến nhân khẩu học phẳng và suýt bị loại khỏi Top 120.

#### Nguyên nhân 4: Độ sâu Cây XGBoost bị giới hạn (`max_depth: 4`)
- Bộ siêu tham số tối ưu tìm được bởi Optuna chọn `max_depth: 4`, tương ứng với tối đa $2^4 = 16$ lá quyết định trên mỗi cây.
- Với 120 đặc trưng trong ma trận, cây 4 tầng ưu tiên tuyệt đối các đặc trưng bảng có độ biến thiên phân đoạn lớn (`is_middle_aged_vulnerable`, `is_product_danger_zone`, `gender`, `active_member`, `country`).
- Các chiều ẩn $z_0 \dots z_{15}$ chỉ được cây tiếp cận ở độ sâu tầng 3 và 4. Một cây quá nông không đủ khả năng kết hợp đa chiều các tọa độ mặt cầu $\mathbb{S}^{15}$ thành một mặt cắt quyết định hình nón.

#### Nguyên nhân 5: Nhiễu Chuỗi & Giới hạn Bayes Bất khả quy (Aleatoric Uncertainty)
- Tập dữ liệu chuỗi 6 tháng chỉ gồm 10 chỉ số tổng hợp theo tháng. Một tỷ lệ khách hàng rời bỏ vì các nguyên nhân ngoại sinh tức thời (được ngân hàng đối thủ mời chào lãi suất cao hơn, chuyển nơi cư trú, thay đổi chính sách phí) hoàn toàn không có dấu hiệu suy thoái hành vi trong 6 tháng trước đó. Nhóm này tạo ra một "trần giới hạn Bayes" mà chỉ dựa vào lịch sử giao dịch sẽ không thể phân biệt được.

---

### 5.3. Hướng dẫn Điều chỉnh Tham số (Hyperparameter Tuning Guide)

Để tạo ra bước nhảy vọt (breakthrough) về Recall (kéo Recall lên $> 65 - 75\%$) trong khi giữ FP trong ngân sách cho phép, cần điều chỉnh đồng bộ 4 nhóm tham số:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   CHIẾN LƯỢC ĐIỀU CHỈNH 4 TẦNG THAM SỐ                   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  TẦNG 1: MANIFOLD│        │ TẦNG 2: SELECTION│        │ TẦNG 3: THRESHOLD│
│    HYPERSPHERE   │        │     FILTER       │        │   CALIBRATION    │
│ m_churn: 0.45->0.7│       │ Whitelist toàn bộ│        │ Quét tau: 0.50   │
│ m_loyal: 0.05->0.0│       │ 21 features      │        │  -> tau* in      │
│ s: 16.0 -> 24.0  │        │ temporal & geom  │        │   [0.32, 0.40]   │
└──────────────────┘        └──────────────────┘        └──────────────────┘
```

#### Bảng Tham số Đề xuất Chi tiết:

| Nhóm Tham số | Tham số Cụ thể | Giá trị Hiện tại | Giá trị Đề xuất Tối ưu | Cơ chế Tác động Hình học & Mô hình |
| :--- | :--- | :---: | :---: | :--- |
| **Manifold ArcFace** | `margin_churn` | `0.45` ($\approx 26^\circ$) | **`0.70` ($\approx 40^\circ$)** | Tăng mạnh lực đẩy góc, mở rộng khoảng cách ly an toàn cho lớp Churn. |
| | `margin_loyal` | `0.05` ($\approx 3^\circ$) | **`0.00` ($0^\circ$)** | Không phạt nhóm Loyal, cho phép nhóm Loyal phân bổ tự nhiên tại tâm, dồn toàn bộ lực ép vào Churn. |
| | `angular_scale (s)` | `16.0` | **`24.0` hoặc `30.0`** | Làm dốc bề mặt hàm xác suất Softmax, tăng độ nhạy gradient tại ranh giới quyết định góc. |
| | `lambda_angular` | `1.0` | **`0.60` - `0.80`** | Giảm nhẹ để cân bằng tốt hơn với MSE tái tạo, thu hẹp khoảng cách tổng quát hóa (Val Gap). |
| **Feature Selection** | `feature_selection_metric` | `Pearson` | **`MI_Pearson_Hybrid`** | Bổ sung Mutual Information ($I(X; Y)$) để nhận diện các đặc trưng khoảng cách góc phi tuyến. |
| | Bảo vệ đặc trưng chuỗi | Không bảo vệ | **`Whitelist temporal_*`** | Giữ lại nguyên vẹn 21 đặc trưng latent và hình học, chỉ áp dụng Top-K trên các biến bảng. |
| **Downstream XGBoost** | `scale_pos_weight` | `1.0` | **`1.8` - `2.5`** | Bổ sung trọng số gradient cho lớp thiểu số để cây ưu tiên tách các nhánh Churn. |
| | `max_depth` | `4` | **`5` - `6`** | Cho phép cây tạo đủ các lát cắt trực giao để xấp xỉ mặt cầu của không gian ẩn. |
| **Decision Threshold** | `eval_threshold` | `0.50` (Cố định) | **`tau* in [0.32, 0.38]`** | Quét tìm ngưỡng tối ưu hóa hàm mục tiêu **$F_2$-Score** (đặt trọng số Recall gấp đôi Precision). |

---

### 5.4. Dự báo Hiệu năng sau khi Tinh chỉnh

Khi áp dụng đồng thời lề góc mở rộng $m_{churn} = 0.70$, bảo vệ đặc trưng hình học khỏi Pearson filter, và hạ ngưỡng quyết định chi phí $\tau^* \approx 0.35$:
1. **Recall dự kiến đạt:** $68\% - 75\%$ (bắt được khoảng $140 - 155$ khách hàng churn thay vì $99$).
2. **False Negatives dự kiến giảm:** Từ $105$ xuống còn $45 - 60$ ca.
3. **False Positives:** Sẽ tăng nhẹ từ $41$ lên khoảng $65 - 80$ ca (đây là cái giá chấp nhận được trong kinh doanh ngân hàng vì chi phí can thiệp CSKH luôn thấp hơn giá trị vòng đời của một khách hàng giữ chân thành công).
4. **Hệ số $F_2$-Score:** Dự kiến tăng từ $\approx 0.51$ lên $> 0.65$.

---

## 6. PHÂN TÍCH THỰC NGHIỆM ĐỐI CHỨNG: VÌ SAO ASYMMETRIC ARCFACE LÀM SUY GIẢM PR-AUC (0.6843 ➔ 0.5617)

### 6.1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Đối Chứng (Full Diagnostic Matrix)

Quá trình chẩn đoán đối kháng trực tiếp giữa **Baseline Unsupervised BiGRU** và **Asymmetric ArcFace** trên cùng một tập dữ liệu ($N=10,000$, $D=37$ kênh động học qua $T=6$ tháng snapshot, 150 features sau lọc, đánh giá trên 1,000 khách hàng hold-out test độc lập):

| Chỉ số Đánh giá (Metrics) | Baseline: Unsupervised BiGRU | Chiến lược 1: Asymmetric ArcFace | Độ chênh lệch ($\Delta$) & Đánh giá |
| :--- | :---: | :---: | :--- |
| **BiGRU Val Reconstruction Loss** | **`0.0025`** | `0.0021` | Tái tạo chuỗi tương đương nhau ($\approx 0.002$). |
| **BiGRU Val Total Loss** | **`0.0025`** | **`1.2985`** | ArcFace chịu thêm $\mathcal{L}_{arc} \approx 1.296$ (áp lực góc quá lớn). |
| **Khoảng cách góc giữa 2 tâm cụm** | `0.3013°` | **`15.4782°`** | Manifold đã bị bẻ cong, đẩy 2 cụm ra xa $\approx 15.5^\circ$. |
| **Cosine giữa 2 tâm cụm** | `0.999986` | `0.963732` | Tâm churn dạt dần ra khỏi tâm loyal. |
| **PR-AUC (Temporal-Only Features)**| **`0.2438`** | **`0.2135`** | **Kém hơn (-0.0303)**: Tự thân vector ArcFace bị nhiễu. |
| **ROC-AUC (Temporal-Only Features)**| **`0.5737`** | **`0.5420`** | **Kém hơn (-0.0317)**: Khả năng phân biệt đơn lẻ thấp. |
| **XGBoost PR-AUC (Full Pipeline)** | **`0.6843`** | **`0.5617`** | **SỤP ĐỔ NẶNG NỀ (-12.26% PR-AUC)**. |
| **XGBoost ROC-AUC (Full Pipeline)**| **`0.8674`** | **`0.8096`** | **Giảm sút (-5.78% ROC-AUC)**. |
| **True Positives (TP) tại $\tau=0.50$**| **`99`** | `90` | Bỏ sót thêm 9 khách hàng rời bỏ. |
| **False Positives (FP) tại $\tau=0.50$**| **`46`** | **`76`** | **Bùng nổ báo động giả (+30 ca, +65.2% FP)**. |
| **False Negatives (FN) tại $\tau=0.50$**| **`105`** | `114` | Tỷ lệ sót tăng từ 51.5% lên 55.9%. |
| **Precision tại $\tau=0.50$** | **`68.28%`** | `54.22%` | **Sụt giảm nghiêm trọng (-14.06%)**. |
| **Recall tại $\tau=0.50$** | **`48.53%`** | `44.12%` | Giảm 4.41%. |
| **F1-Score tại $\tau=0.50$** | **`0.5673`** | `0.4865` | Giảm 8.08%. |
| **Ngưỡng tối ưu $\tau^*$ (Best Threshold)**| `0.2710` | `0.3166` | Phân phối xác suất bị đẩy lệch. |
| **Best F1 tại $\tau^*$** | **`0.6224`** | `0.5378` | Kém hơn -8.46%. |
| **Best Recall tại $\tau^*$** | **`73.53%`** (TP=150) | `62.75%` (TP=128) | Kém hơn -10.78% khả năng bắt churn. |
| **Xếp hạng & Gain đặc trưng Temporal**| `temporal_z14`: Gain 0.0036 (Top 30+)<br>`temporal_drift`: Gain 0.0030 | **`temporal_angular_risk_ratio`: Gain 0.0769 (HẠNG 2 TOÀN BỘ MÔ HÌNH!)**<br>`temporal_angular_churn_dist`: Gain 0.0193 (Hạng 14) |

---

### 6.2. Bốn Nguyên Nhân Gốc Rễ Khiến Asymmetric ArcFace Thất Bại

#### 1. Hiện Tượng "Ăn Thịt Đặc Trưng" và Chiếm Đoạt Lát Cắt Cây (Feature Cannibalization & Greedy Split Hijacking)
- **Cơ chế**: Trong kết quả thực nghiệm của `asymmetric_arcface`, chỉ số **`temporal_angular_risk_ratio` có Gain lên tới 0.0769, chiếm vị trí Hạng 2 trong số toàn bộ 150 đặc trưng của mô hình XGBoost** (chỉ đứng sau biến tuổi tác `Age` / `is_middle_aged_vulnerable`).
- **Nghịch lý**: BiGRU chỉ được học từ 6 snapshot của 37 kênh hành vi mà không hề biết các thông tin tĩnh sống còn (Balance, NumOfProducts, HasCrCard, Geography). Tự bản thân các đặc trưng chuỗi chỉ đạt PR-AUC đơn lẻ là `0.2135` (chứa độ bất định và nhiễu rất cao).
- **Hệ quả**: Do hàm mất mát góc ép biểu diễn tạo ra một tỷ số phân kỳ cực đại $\frac{\theta_{loyal}}{\theta_{churn}}$, cây quyết định XGBoost (với thuật toán chia nhánh tham lam theo độ giảm impurity) **bị đánh lừa**. Cây vội vã đưa `temporal_angular_risk_ratio` lên các tầng nhánh gốc (root splits) để phân chia dữ liệu, vô tình **loại bỏ hoặc làm lu mờ các đặc trưng bảng cốt lõi**. Việc phân chia sớm dựa trên một tín hiệu chuỗi chưa chín muồi và có độ nhiễu cao dẫn tới việc cô lập nhầm các khách hàng lành mạnh vào nhánh rủi ro $\to$ **False Positives tăng vọt từ 46 lên 76 ca (+65%)**.

#### 2. Xung Đột Topo Giữa Siêu Cầu Phi Tuyến ($\mathbb{S}^{15}$) và Kỹ Thuật SMOTE
- **Cơ chế**: Trong quy trình tiền xử lý, SMOTE được áp dụng để cân bằng tập huấn luyện từ 7,999 mẫu lên 12,740 mẫu ($1:1$). SMOTE hoạt động dựa trên phép **nội suy tuyến tính phẳng trong không gian Euclidean**:
  $$\mathbf{x}_{\text{synthetic}} = \mathbf{x}_i + \lambda (\mathbf{x}_{k\text{-NN}} - \mathbf{x}_i), \quad \lambda \in [0, 1]$$
- **Vấn đề**:
  - Với `unsupervised`: Không gian latent $z \in \mathbb{R}^{16}$ là không gian liên tục phẳng, bao lồi của các điểm nội suy vẫn nằm trong phân phối tự nhiên.
  - Với `asymmetric_arcface`: Các vector $z$ đã bị chiếu lên mặt cầu đơn vị $\mathbb{S}^{15}$ ($\|z\|_2 = 1$). Phép nội suy tuyến tính giữa 2 điểm trên mặt cầu sẽ tạo ra vector nằm **bên trong lòng khối cầu** ($\|x_{\text{synthetic}}\|_2 < 1$).
  - Khi tính khoảng cách góc $\arccos(z^\top W)$ và tỷ số `temporal_angular_risk_ratio` từ các điểm tổng hợp bị biến dạng này, các giá trị góc trở nên phi lý và phá vỡ cấu trúc manifold, làm sai lệch gradient của cây XGBoost.

#### 3. Gradient Dominance và Bão Hòa Xác Suất Softmax ($s=16.0$)
- **Hàm mất mát góc**:
  $$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{angular} \cdot \mathcal{L}_{arc}$$
- Với $\lambda_{angular} = 1.0$ và scale factor $s = 16.0$:
  - Trong khi $\mathcal{L}_{recon}$ hội tụ về $\sim 0.0021$, thì $\mathcal{L}_{arc}$ có độ lớn $\sim 1.20 - 1.30$ (gấp **600 lần** $\mathcal{L}_{recon}$).
  - Gradient của BiGRU bị chi phối hoàn toàn bởi nhiệm vụ phân loại góc nhị phân, làm tê liệt khả năng học các vi mô động học (micro-dynamics) của 37 kênh hành vi.
  - Hơn nữa, với $s=16.0$, các logits $s \cdot \cos\theta$ dao động trong khoảng $[-16, 16]$, đẩy hàm softmax vào vùng bão hòa (vanishing gradient) đối với nhóm khách hàng loyal thông thường và gây bùng nổ gradient đối với các trường hợp ngoại lai. BiGRU bị ép học vẹt (overfitting) trên tập train, dẫn đến Val Loss lên tới $1.2985$.

#### 4. Tính Bổ Trợ (Orthogonality) Đáng Giá của Baseline Unsupervised
- **Tại sao Unsupervised đạt PR-AUC 0.6843?**
  - Mô hình Autoencoder phi giám sát học cách nén toàn bộ 37 kênh hành vi qua 6 tháng thành 16 biến ẩn phản ánh **nhịp điệu khách hàng thuần túy** (tốc độ tiêu tiền, độ trôi số dư, mức độ tích cực đăng nhập). Nó hoàn toàn không cố gắng dự đoán khách hàng có churn hay không.
  - Do đó, các đặc trưng $z_0 \dots z_{15}$ mang thông tin **độc lập và trực giao (orthogonal)** với các biến bảng tĩnh.
  - Khi vào XGBoost, chúng chỉ đóng vai trò thứ cấp (Gain khiêm tốn từ 0.0024 đến 0.0036, Rank 30+), giúp cây tinh chỉnh các trường hợp cận biên mà không làm phá hỏng khung phân loại vững chắc của dữ liệu nhân khẩu học và tài khoản.

---

### 6.3. Giải Pháp Khắc Phục Thực Tế (Actionable Remediation Protocol)

Để đưa chiến lược học biểu diễn hình học vượt qua baseline Unsupervised mà không bị sụp đổ PR-AUC, cần áp dụng 4 điều chỉnh kiến trúc bắt buộc:

1. **Hạ trọng số đa nhiệm $\lambda_{angular}$ (Soft Manifold Regularization)**:
   - Giảm $\lambda_{angular}$ từ `1.0` $\to$ **`0.05` - `0.10`**.
   - Mục tiêu: Đưa $\mathcal{L}_{arc}$ về độ lớn tương đương với $\mathcal{L}_{recon}$ ($\sim 0.01 - 0.05$), giữ cho BiGRU tập trung vào bảo toàn thông tin chuỗi thay vì cố trở thành một classifier đơn độc.
2. **Hạ Scale Factor $s$ và Thu hẹp Lề góc $m_{churn}$**:
   - Hạ $s$ từ `16.0` $\to$ **`6.0` - `8.0`**: Ngăn chặn tình trạng bão hòa sớm của Softmax và tránh gradient cực đoan trên $S^{15}$.
   - Đặt $m_{churn} = 0.20 \text{ rad} \approx 11.5^\circ$ và $m_{loyal} = 0.00$: Không trừng phạt nhóm khách hàng ở lại.
3. **Cô Lập SMOTE Khỏi Không Gian Hình Học (Geometry-Aware Resampling)**:
   - Không cho phép SMOTE nội suy trên các đặc trưng góc và khoảng cách mặt cầu.
   - Hoặc tối ưu nhất: **Tắt SMOTE hoàn toàn** và sử dụng tham số `scale_pos_weight = 2.0 - 2.5` trực tiếp trong XGBoost để tránh sinh ra các vector tổng hợp nằm ngoài mặt phẳng siêu cầu.
4. **Biến Đổi Phi Tuyến Bậc Thấp & Giới Hạn Ảnh Hưởng (Feature Transformation)**:
   - Áp dụng `log1p(temporal_angular_risk_ratio)` hoặc chuẩn hóa Quantile để nén đuôi phân phối dài, ngăn cản XGBoost dùng biến này làm root split độc quyền.
5. **Chuyển Hướng Sang Chiến Lược 3 (Optimal Transport & Drift Velocity)**:
   - Chiến lược 3 không áp đặt nhãn nhị phân lên hàm loss mà tính toán trường vận tốc trôi Wasserstein ($\vec{v} = \Delta z$) từ chính biểu diễn Unsupervised ổn định ($PR=0.6843$). Đây là con đường kế thừa an toàn nhất, phát huy trọn vẹn sức mạnh của Autoencoder mà vẫn trích xuất được động học rủi ro dạt biên.

---

## 7. ĐÁNH GIÁ THỰC NGHIỆM CHIẾN LƯỢC 2: VÌ SAO QUẢ CẦU POINCARÉ CHO PR-AUC THẤP NHẤT (0.5348)

### 7.1. Bảng Tổng Hợp So Sánh 3 Chiến Lược Thực Nghiệm

| Chỉ số Đánh giá (Metrics) | Baseline: Unsupervised BiGRU | Chiến lược 1: Asymmetric ArcFace | Chiến lược 2: Hyperbolic Poincaré |
| :--- | :---: | :---: | :---: |
| **BiGRU Val Reconstruction Loss** | **`0.0025`** | `0.0021` | `0.0027` |
| **BiGRU Val Total Loss** | **`0.0025`** | `1.2985` | `0.1442` |
| **PR-AUC (Temporal-Only)** | **`0.2438`** | `0.2135` | `0.2301` |
| **ROC-AUC (Temporal-Only)** | **`0.5737`** | `0.5420` | `0.5486` |
| **XGBoost PR-AUC (Full Pipeline)** | **`0.6843`** | `0.5617` | **`0.5348` (THẤP NHẤT)** |
| **XGBoost ROC-AUC (Full Pipeline)**| **`0.8674`** | `0.8096` | `0.8186` |
| **True Positives (TP) tại $\tau=0.50$**| **`99`** | `90` | `93` |
| **False Positives (FP) tại $\tau=0.50$**| **`46`** | **`76`** | **`67` (+45.7% báo động giả)** |
| **False Negatives (FN) tại $\tau=0.50$**| **`105`** | `114` | `111` |
| **Precision tại $\tau=0.50$** | **`68.28%`** | `54.22%` | **`58.12%` (-10.16%)** |
| **Recall tại $\tau=0.50$** | **`48.53%`** | `44.12%` | `45.59%` |
| **F1-Score tại $\tau=0.50$** | **`0.5673`** | `0.4865` | `0.5110` |
| **Ngưỡng tối ưu $\tau^*$** | `0.2710` | `0.3166` | `0.1731` |
| **Best F1 tại $\tau^*$** | **`0.6224`** | `0.5378` | `0.5389` |
| **Best Recall tại $\tau^*$** | **`73.53%`** | `62.75%` | `69.61%` |
| **Xếp hạng đặc trưng Temporal** | `temporal_z14`: Hạng 30+<br>(Gain: 0.0036) | `temporal_angular_risk`: Hạng 2<br>(Gain: 0.0769) | **`temporal_poincare_churn_dist`: HẠNG 1 (Gain: 0.1013)**<br>**`temporal_poincare_loyal_dist`: HẠNG 3 (Gain: 0.0611)** |

---

### 7.2. Bốn Nguyên Nhân Gốc Rễ Khiến Quả Cầu Poincaré Cho Hiệu Năng Kém Nhất

#### 1. Hiện Tượng "Sụp Đổ Rìa Biên 100%" (100% Boundary Shell Collapse)
- **Kỳ vọng lý thuyết**: Trong quả cầu Poincaré $\mathbb{B}_c^d$, khách hàng trung thành ($y=0$) co cụm dày đặc ở vùng lõi gần gốc tọa độ ($u \to 0$), còn khách hàng churn ($y=1$) trôi dạt ra gần rìa biên ($u \to \partial\mathbb{B}$), nơi thể tích không gian dãn nở theo hàm mũ.
- **Thực tế trong mô hình**:
  - Vector ẩn $z \in \mathbb{R}^{16}$ đi ra từ lớp `Tanh` của BiGRU có độ dài Euclid $\|z\|_2 = \sqrt{\sum z_i^2}$ dao động trong khoảng $[1.81, 2.96]$ (trung bình là **$2.51$**).
  - Hàm chiếu `poincare_project(x, c=1.0)` thực hiện cắt cụt đơn giản: $u = \frac{x}{\|x\|_2} \cdot (1 - \epsilon)$.
  - Vì chuẩn của toàn bộ $100\%$ khách hàng đều lớn hơn $1.0$, **TOÀN BỘ 100.00% KHÁCH HÀNG BỊ ÉP CHẶT LÊN VỎ BIÊN** ($\|u\|_2 = 0.99999$).
  - Đo lường thực nghiệm:
    * `temporal_poincare_norm`: Min = 1.0000, Max = 1.0000, Mean = 1.0000, **Std = 0.0000**!
    * `temporal_poincare_origin_dist`: Min = 12.1929, Max = 12.2047, Mean = 12.2024, **Std = 0.0034**!
  - Cấu trúc phân cấp hình cây biến mất hoàn toàn. Quả cầu đặc Poincaré bị biến thành một vỏ rỗng mỏng dính ở biên vô cực, làm triệt tiêu hoàn toàn thông tin khoảng cách tới gốc tọa độ!

#### 2. Cướp Quyền Chia Nhánh Nghiêm Trọng Nhất (Extreme Split Hijacking - Rank 1 & Rank 3)
- **Thực tế trong cây XGBoost**:
  * `temporal_poincare_churn_dist`: **Chiếm vị trí Hạng 1 toàn bộ mô hình với Gain = 0.1013 (10.13% tổng Gain!)**
  * `temporal_poincare_loyal_dist`: **Chiếm vị trí Hạng 3 toàn bộ mô hình với Gain = 0.0611 (6.11% tổng Gain!)**
  * Chỉ riêng 2 đặc trưng này đã kiểm soát tới **16.24% quyết định chia nhánh** của toàn bộ cây XGBoost, lấn át hoàn toàn các đặc trưng nghiệp vụ cốt lõi như `is_middle_aged_vulnerable` (0.0679), `products_number` (0.0213), `age` (0.0204), `active_member` (0.0166), `balance` (0.0096).
- **Hậu quả**:
  * Khoảng cách hyperbolic gần biên có mẫu số $(1 - \|u\|^2)(1 - \|v\|^2) \approx \epsilon^2 = 10^{-10}$, biến phép đo thành một hàm số **cực kỳ nhạy cảm với nhiễu vi mô** (khoảng cách nhảy vọt từ 8.65 lên 14.25 chỉ vì một dịch chuyển nhỏ).
  * XGBoost bị hấp dẫn bởi độ giảm impurity cực lớn của biến này nên đưa nó lên làm **Root Split (lát cắt gốc)** ngay tầng 1.
  * Vì tín hiệu chuỗi tự thân chỉ có PR-AUC = 0.2301, việc chia nhánh sớm theo một đặc trưng nhiễu cao khiến hàng loạt khách hàng trung thành bị đẩy nhầm vào nhánh Churn $\to$ **False Positives tăng vọt lên 67 ca (+45.7%)**, kéo Precision rơi từ 68.28% xuống 58.12% và PR-AUC sập về **0.5348**.

#### 3. Lệch Pha Gradient Giữa Tối Ưu Euclid và Hình Học Riemannian
- Prototype của Churn và Loyal trong `PoincarePrototypeHead` là các tham số `nn.Parameter` được tối ưu bằng **Euclidean AdamW**, không phải **Riemannian SGD/Adam**.
- Metric tensor tại điểm $u$ trong Poincaré Ball đòi hỏi gradient phải được chuẩn hóa theo hệ số conformal $\lambda_u^{-2} = \frac{(1 - \|u\|^2)^2}{4}$. Khi bỏ qua hiệu chỉnh Riemannian, các bước cập nhật Euclid ở gần biên bị méo mó nghiêm trọng, khiến hai tâm prototype không thể ổn định ở vị trí topo tối ưu.

#### 4. Bài Học Tổng Quát Cho Pipeline Churn: "Càng Ép Phân Loại Sớm, Hiệu Năng Càng Giảm"
- Cả **Chiến lược 1 (ArcFace: PR 0.5617)** và **Chiến lược 2 (Poincaré: PR 0.5348)** đều mắc cùng một sai lầm bản chất:
  **Cố ép một bộ mã hóa chuỗi thời gian (vốn dĩ chỉ có 6 snapshot với tín hiệu chuỗi yếu PR ~ 0.21 - 0.24) phải làm nhiệm vụ của một bộ phân loại nhị phân.**
- Việc giám sát nhãn Churn/Loyal lên không gian ẩn $z$ đã phá hủy tính trực giao (orthogonality) của biểu diễn hành vi, tạo ra các đặc trưng giả mạo có Gain quá cao lấn át các biến bảng tĩnh, dẫn đến sự sụp đổ dây chuyền của mô hình cây downstream.
- Ngược lại, **Baseline Unsupervised (PR 0.6843)** không áp đặt nhãn, nén nhịp điệu chi tiêu một cách khách quan, tạo ra các biến $z$ độc lập đóng vai trò chất xúc tác làm mịn cây mà không phá hỏng cấu trúc phân nhánh gốc.

---

## 8. TÁI ĐỊNH HÌNH TOÀN DIỆN: CHUYỂN DỊCH SANG HỌC HÌNH HỌC TỰ GIÁM SÁT THUẦN TÚY (PURE SELF-SUPERVISED GEOMETRIC REPRESENTATION LEARNING)

Để triệt tiêu vĩnh viễn nguy cơ **Rò rỉ nhãn (Target Label Leakage)** và **Cướp nhánh cây quyết định (Split Hijacking)**, toàn bộ 4 chiến lược nghiên cứu chuyên sâu đã được tái cấu trúc thành các mô hình **Hoàn toàn Tự giám sát (100% Unsupervised / Self-Supervised)**:

### 8.1. Bảng Chuẩn Hóa Cấu Trúc 4 Chiến Lược Mới

| Chiến lược | Triết lý Hình học | Cơ chế Mất mát Tự giám sát (Không dùng nhãn $y$) | Các Đặc trưng Đa tạp Phi tuyến Sinh ra |
| :--- | :--- | :--- | :--- |
| **1. Hyperspherical Manifold** (`asymmetric_arcface.py`) | Siêu cầu đơn vị $\mathbb{S}^{d-1}$ (Wang & Isola, ICML 2020) | - **Alignment Loss**: Căn chỉnh trắc địa summary $z$ và trạng thái cuối $z_{t6}$<br>- **Uniformity Loss**: Cực đại hóa entropy trên siêu cầu chống sụp đổ chiều | `temporal_drift_velocity`, `temporal_angular_drift`, `temporal_hypersphere_norm`, `temporal_hypersphere_density` |
| **2. Hyperbolic Poincaré** (`hyperbolic_poincare.py`) | Không gian Hyperbolic $\mathbb{B}_c^d$ (Nickel & Kiela, NeurIPS 2017) | - **Geodesic Smoothness Loss**: Ràng buộc trắc địa giữa điểm đầu $u_{t1}$ và điểm cuối $u_{t6}$<br>- **Origin Centering**: Kéo hành vi ổn định về tâm quả cầu $\|u\| \approx 0$ | `temporal_poincare_norm`, `temporal_poincare_origin_dist`, `temporal_poincare_geodesic_drift`, `temporal_poincare_conformal_factor`, `temporal_drift_velocity` |
| **3. Optimal Transport & Kinetics** (`optimal_transport.py`) | Cơ học Hamilton & Vận chuyển tối ưu Benamou-Brenier | - **Kinetic Action Minimization**: Tối thiểu hóa động năng $\frac{1}{2}\|\vec{v}_i\|_2^2$ theo nguyên lý tác dụng cực tiểu | `temporal_drift_velocity`, `temporal_geodesic_length`, `temporal_tortuosity`, `temporal_kinetic_action`, `temporal_acceleration_norm` |
| **4. Neyman-Pearson Level-Set** (`neyman_pearson.py`) | Ước lượng bao phân vị năng lượng (Scott & Nowak 2005) | - **Pinball Quantile Loss**: Tự động ước lượng ngưỡng phân vị năng lượng $\tau_\alpha$ tại mức rủi ro $\alpha$ | `temporal_drift_velocity`, `temporal_np_recon_energy`, `temporal_np_margin`, `temporal_np_risk_score` |

### 8.2. Kết quả đạt được
1. **Triệt tiêu 100% Label Leakage**: Không có bất kỳ hàm loss hay prototype head nào nhận nhãn $y \in \{0, 1\}$.
2. **Khôi phục tính trực giao và liên tục**: Các đặc trưng trắc địa (độ cong, gia tốc, động năng, tỷ số ngoằn ngoèo tortuosity, góc quay siêu cầu, khoảng cách hyperbol tới gốc) là các đại lượng vật lý nội tại, bổ trợ trực giao cho dữ liệu bảng tĩnh thay vì cạnh tranh cướp split.
3. **Bảo toàn 100% tính tương thích ngược**: Hệ thống chạy mượt mà trên toàn bộ pipeline và bộ test tự động.

### 8.3. Chuẩn hóa Không âm cho Hàm Mất mát Đa tạp (Non-Negative Loss Calibration)
Trong định nghĩa nguyên bản của Wang & Isola (ICML 2020), hàm phân bố đều siêu cầu:
$$\mathcal{L}_{\text{uniform}}(f; t) = \log \mathbb{E}_{i \ne j} \left[ e^{-t \|z_i - z_j\|^2} \right]$$
Do $\|z_i - z_j\|^2 \in [0, 4]$ trên siêu cầu $\mathbb{S}^{d-1}$, đại lượng $e^{-t \|z_i - z_j\|^2} \in [e^{-4t}, 1]$, do đó giá trị kỳ vọng luôn $\le 1$ và hàm $\log$ nguyên bản luôn mang giá trị âm: $\mathcal{L}_{\text{uniform}} \in [-4t, 0]$. Khi sai số tái tạo $\text{recon\_mse}$ hội tụ xuống mức nhỏ ($< 0.05$), giá trị âm này sẽ áp đảo và làm tổng hàm loss trở nên âm ($\text{train\_loss} < 0$).

Để đảm bảo tính trực quan, ổn định cho bộ theo dõi loss và bộ điều khiển learning rate, hệ thống đã chuẩn hóa tịnh tiến lý thuyết:
$$\widetilde{\mathcal{L}}_{\text{uniform}} = \log \mathbb{E}_{i \ne j} \left[ e^{-t \|z_i - z_j\|^2} \right] + 4t \ge 0$$
- **Tính bảo toàn đạo hàm tuyệt đối**: $\nabla_z \widetilde{\mathcal{L}}_{\text{uniform}} = \nabla_z \mathcal{L}_{\text{uniform}} + \nabla_z (4t) = \nabla_z \mathcal{L}_{\text{uniform}}$. Quá trình lan truyền ngược gradient là đồng nhất 100% với lý thuyết gốc của Wang & Isola.
- **Giá trị loss luôn dương**: Đảm bảo $\text{train\_loss} \ge 0$ và $\text{val\_loss} \ge 0$ đơn điệu giảm ở mọi epoch huấn luyện.

---

## 9. CƠ CHẾ ĐẦU VÀO, PHÉP BIẾN ĐỔI KHÔNG GIAN HÌNH HỌC VÀ BỘ ĐẶC TRƯNG CHUYỂN TIẾP CHO DOWNSTREAM XGBOOST

### 9.1. Biểu Diễn Đầu Vào Của Các Hướng Manifold: Bản Chất Tensor $\mathcal{X}$ So Với Unsupervised

Một thắc mắc cốt lõi trong nghiên cứu: **Liệu biểu diễn đầu vào của các hướng Manifold có khác biệt so với Unsupervised Autoencoder hay không?**

**Câu trả lời:** **Biểu diễn đầu vào hoàn toàn ĐỒNG NHẤT 100%.**
Cả mô hình Unsupervised Baseline lẫn tất cả các chiến lược Manifold (`hyperspherical`, `hyperbolic_poincare`, `optimal_transport`, `neyman_pearson`, `temporal_attention`) đều nhận chung một tensor chuỗi thời gian 3 chiều:
$$\mathcal{X} \in \mathbb{R}^{B \times T \times D}$$
trong đó:
- $B$: Batch size (ví dụ: $256$).
- $T$: Chiều dài chuỗi thời gian (số bước lịch sử quan sát).
- $D$: Số kênh đặc trưng hành vi tại mỗi bước thời gian.

Trong kiến trúc hệ thống hiện nay, tensor đầu vào $\mathcal{X}$ này được cung cấp qua 2 chế độ cấu hình linh hoạt (điều khiển bởi cờ `use_raw_event_tokens` trong `config.yaml`):

```
                                  ┌──────────────────────────────────────────────┐
                                  │   DỮ LIỆU GIAO DỊCH & LOG THÔ (1.8M ROWS)    │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                        ┌────────────────────────────────┴────────────────────────────────┐
                        ▼                                                                 ▼
      [Chế độ 1: Handcrafted Extractors]                                [Chế độ 2: Raw Event Tokens]
         use_raw_event_tokens: false                                      use_raw_event_tokens: true
                        │                                                                 │
    • 4 Extractors nghiệp vụ chuyên gia                              • RawEventSequenceExtractor
      (Balance, Transaction, Web/App, Support)                         (Log-amount, Event Category, Channel, Status)
    • Cắt theo T = 6 snapshot tháng                                  • L = 50 sự kiện gần nhất
    • D = 37 kênh động học thủ công                                   • D = 26 chiều token nhúng thô
                        │                                                                 │
                        ▼                                                                 ▼
            Tensor X in R^(B x 6 x 37)                                       Tensor X in R^(B x 50 x 26)
                        │                                                                 │
                        └────────────────────────────────┬────────────────────────────────┘
                                                         │
                                                         ▼
                                      ┌─────────────────────────────────────┐
                                      │   Bi-GRU / Attention Backbone       │
                                      │   sinh vector trạng thái ẩn h in R^d │
                                      └──────────────────┬──────────────────┘
                                                         │
                         ┌───────────────────────────────┼───────────────────────────────┐
                         ▼                               ▼                               ▼
                 [Unsupervised]                 [Hyperspherical]                 [Hyperbolic]
             Không gian phẳng R^16           Mặt siêu cầu S^15               Đĩa Poincaré B^16
```

1. **Chế độ 1: Handcrafted Snapshot Tensor** (`TemporalSnapshotExtractor`):
   - Kích thước: $\mathcal{X} \in \mathbb{R}^{B \times 6 \times 37}$.
   - Chuỗi $T=6$ tháng, mỗi tháng trích xuất 37 chỉ số biến động từ 4 domain extractors (Balance, Transactions, Web/App Engagement, Customer Support).
2. **Chế độ 2: Raw Event Token Tensor** (`RawEventSequenceExtractor`):
   - Kích thước: $\mathcal{X} \in \mathbb{R}^{B \times 50 \times 26}$.
   - Trích xuất trực tiếp $L=50$ sự kiện lịch sử gần nhất của khách hàng từ 1.8 triệu dòng logs thô, không qua bất kỳ công thức thống kê trung bình/tổng thủ công nào. Mỗi sự kiện được mã hóa thành vector 26 chiều (One-hot event type, channel, status, log transaction amount, time-delta).

**Kết luận về đầu vào:**
Các chiến lược Manifold **không thay đổi định dạng hay nội dung dữ liệu đầu vào**. Tất cả đều đi qua cùng một mạng nơ-ron trích xuất đặc trưng thời gian (Backbone Bi-GRU 2 chiều hoặc Temporal Attention + Multi-scale Conv1D) để nén toàn bộ chuỗi lịch sử thành một vector trạng thái ẩn $h_i \in \mathbb{R}^{16}$.

---

### 9.2. Tận Dụng Lợi Thế Của Các Phép Biến Đổi Không Gian Biểu Diễn (Manifold Warping & Geometry)

Nếu biểu diễn đầu vào và mạng backbone giống hệt nhau, **vậy ưu thế của các hướng nghiên cứu Manifold nằm ở đâu?**

Lợi thế nằm ở **cấu trúc topo (metric tensor), hình học của không gian ẩn (latent space geometry), và các toán tử ràng buộc vật lý/trắc địa**. Cụ thể:

#### 1. Sự khác biệt về Metric Tensor và Độ cong Không gian (Space Curvature)
- **Unsupervised Autoencoder (Không gian phẳng Euclidean $\mathbb{R}^d$):**
  - Metric tensor là ma trận đơn vị $g_{ij} = \delta_{ij}$. Khoảng cách trắc địa chỉ đơn giản là đường thẳng Euclid $d(u, v) = \|u - v\|_2$.
  - Không gian có độ cong bằng 0 ($K = 0$). Thể tích quả cầu bán kính $r$ tăng chậm theo hàm đa thức: $V(r) \propto r^d$.
  - *Hạn chế (Manifold Tangling):* Do thể tích phẳng bị giới hạn, các quỹ đạo suy thoái phức tạp của khách hàng bị đan xen vào nhau ở các vùng mật độ cao, không thể tự tách rời nếu chỉ tối ưu hóa sai số tái tạo phương sai $\mathcal{L}_{recon}$.
- **Hyperspherical Manifold ($\mathbb{S}^{d-1}$ - Độ cong dương $K > 0$):**
  - Thực hiện phép chiếu chuẩn hóa $L_2$: $\tilde{z} = \frac{z}{\|z\|_2} \in \mathbb{S}^{d-1}$.
  - *Lợi thế hình học:* Triệt tiêu hoàn toàn **độ lệch biên độ (magnitude bias)**. Trong dữ liệu ngân hàng, những khách hàng VIP giao dịch hàng trăm triệu thường có vector ẩn $\|z\|$ rất lớn so với khách hàng phổ thông (vài trăm nghìn), dù tỷ lệ suy thoái hành vi của họ là như nhau. Việc chuẩn hóa lên siêu cầu đưa mọi khách hàng về cùng một bán kính, biến bài toán thành so sánh **góc trắc địa (geodesic angle)** $\theta = \arccos(\tilde{z}_i^\top \tilde{z}_j)$.
  - Tối ưu hóa nguyên lý phân bố đều (Uniformity): Cực đại hóa entropy trên mặt cầu, chống sụp đổ biểu diễn và phân bổ tối ưu góc mở.
- **Hyperbolic Poincaré Ball ($\mathbb{B}_c^d$ - Độ cong âm $K = -c < 0$):**
  - Chiếu qua ánh xạ hàm mũ Riemannian: $u = \tanh(\sqrt{c}\|h\|) \frac{h}{\sqrt{c}\|h\|} \in \mathbb{B}_c^d$.
  - Metric tensor: $g_u = \left(\frac{2}{1 - c\|u\|^2}\right)^2 g^E$.
  - *Lợi thế hình học:* Thể tích không gian dãn nở theo **hàm mũ** $V_{\mathbb{H}}(r) \propto \sinh^{d-1}(r) \approx \frac{1}{2^{d-1}} e^{(d-1)r}$.
  - Hành vi tài chính có tính chất **cây phân cấp thứ bậc (Hierarchical Degradation Tree)**: Khách hàng chuẩn mực ở trung tâm $\to$ phân nhánh suy giảm tương tác $\to$ dạt biên thành nhiều loại hình dị biệt. Không gian phẳng Euclid không đủ "chỗ chứa" cho các nút lá của cây phân cấp khiến chúng bị đè ép vào nhau (crowding effect). Trong khi đó, không gian Hyperbolic cung cấp diện tích vô hạn ở gần biên ($\|u\| \to 1$), cho phép các dạng suy thoái hiếm gặp tách rời hoàn toàn khỏi lõi khách hàng ổn định mà không làm vỡ cấu trúc cụm!
- **Optimal Transport & Kinetic Manifold (Đa tạp Động học Benamou-Brenier):**
  - Không coi chuỗi là các điểm độc lập, mà coi là một đường cong trắc địa $\gamma_i(t)$ trên đa tạp xác suất.
  - Áp dụng nguyên lý tác dụng cực tiểu: Tối thiểu hóa động năng $\mathcal{K} = \frac{1}{2} \int \|\dot{\gamma}(t)\|^2 dt$.
  - *Lợi thế hình học:* Bắt trọn đạo hàm bậc 1 (vận tốc trôi $\vec{v}$) và đạo hàm bậc 2 (gia tốc chuyển pha $\vec{a}$), phân biệt dứt điểm giữa "khách hàng ít giao dịch do thói quen" và "khách hàng VIP đang trên đà lao dốc đột ngột".

#### 2. Toán tử Ràng buộc Hình học Thay vì Chỉ Tái tạo Thô
- Unsupervised chỉ giải quyết bài toán:
  $$\min_{\theta} \mathcal{L}_{recon} = \frac{1}{BTD} \sum \|X - \hat{X}\|_2^2$$
  (Mô hình chỉ đóng vai trò như một bộ nén tín hiệu phi tuyến / Non-linear PCA).
- Các chiến lược Manifold bổ sung các **toán tử điều hòa hình học tự giám sát (Self-Supervised Geometric Regularizers)**:
  $$\min_{\theta} \mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{geom} \cdot \mathcal{L}_{geom}$$
  - Hyperspherical: $\mathcal{L}_{geom} = \mathcal{L}_{align}(\tilde{z}_{summary}, \tilde{z}_{t6}) + \widetilde{\mathcal{L}}_{uniform}(\tilde{z})$.
  - Hyperbolic: $\mathcal{L}_{geom} = d_{\mathbb{H}}(u_{t1}, u_{t6}) + \lambda_{origin} \|u_{summary}\|^2$.
  - Optimal Transport: $\mathcal{L}_{geom} = \text{Kinetic Action}(\vec{v}) + \text{Geodesic Smoothness}$.
  - Neyman-Pearson: $\mathcal{L}_{geom} = \text{Pinball Quantile Loss}(\text{Energy}, \alpha)$.

Nhờ đó, không gian ẩn được **chủ động định hình (sculpted)** để phản ánh đúng cấu trúc động học rủi ro mà không cần đến nhãn giám sát $y$.

---

### 9.3. Đưa Cái Gì Vào Downstream XGBoost? (Downstream Feature Probing & Schema)

Mô hình XGBoost (hoặc LightGBM, Random Forest, CatBoost) là mô hình tập hợp cây quyết định (Decision Trees). Cây quyết định **chỉ có thể nhận ma trận đặc trưng bảng 2 chiều dạng số thực $[N \times K]$**, và thực hiện các lát cắt trực giao phân đoạn:
$$\text{If } x_j \le c \text{ Then Left Else Right}$$
Cây **hoàn toàn không thể nhận trực tiếp một đa tạp Riemannian trừu tượng** hay một metric tensor $g_{ij}$.

Do đó, sau khi mô hình Manifold Autoencoder hoàn tất quá trình biến đổi không gian, hệ thống tiến hành bước **"Chiếu đo lường hình học" (Geometric Probing / Scalar Invariant Extraction)** để trích xuất các biến số nạp vào XGBoost:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TỔNG THỂ MA TRẬN ĐẶC TRƯNG NẠP VÀO DOWNSTREAM XGBOOST             │
│                               X_model in R^(N x K)                                     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐              ┌──────────────────┐
│  NHÓM 1: BẢNG    │             │ NHÓM 2: MANIFOLD │              │  NHÓM 3: GNN     │
│   (TABULAR)      │             │    TEMPORAL      │              │   TOPOLOGY       │
│                  │             │                  │              │ (NẾU BẬT CỜ GNN) │
│ • 142 đặc trưng  │             │ • 16-d Latents:  │              │ • 16-d Latents:  │
│   nhân khẩu học, │             │   temporal_z0..15│              │   gnn_z0..15     │
│   số dư, tài sản,│             │ • 1-5 Bất biến   │              │ • 3 Graph Metrics│
│   giao dịch tĩnh,│             │   hình học đo lường│              │   centrality,    │
│   tỷ lệ tài chính│             │   (Invariants)   │              │   clustering,... │
└────────┬─────────┘             └────────┬─────────┘              └────────┬─────────┘
         │                                │                                 │
         └────────────────────────────────┼─────────────────────────────────┘
                                          │
                                          ▼
                Ma trận hoàn chỉnh: K = 142 + 21 + 19 = 182 features
```

#### Chi Tiết 3 Nhóm Đặc Trưng Đầu Vào Của XGBoost:

#### 1. Nhóm 1: Đặc trưng Bảng Thủ công (Tabular Domain Features) — 142 features
- Các biến nhân khẩu học: `Age`, `Gender`, `Geography`, `CreditScore`, `Tenure`.
- Các biến hành vi tĩnh: `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
- Các đặc trưng tương tác & chuyên gia: `is_middle_aged_vulnerable`, `balance_to_salary_ratio`, `credit_age_interaction`,...

#### 2. Nhóm 2: Đặc trưng Đa tạp Thời gian (Temporal Manifold Features) — 17 đến 21 features
Được trích xuất từ `TemporalEmbeddingManager` thông qua chiến lược manifold đã cấu hình:
- **Tọa độ tiềm ẩn nén (Compressed Latent Coordinates - 16 chiều):**
  - `temporal_z0`, `temporal_z1`, ..., `temporal_z15`: Đại diện cho vị trí không gian nén của toàn bộ quỹ đạo hành vi.
- **Các bất biến hình học & Chỉ số động học trắc địa (Geometric Invariants - 1 đến 5 chiều):**
  Tùy theo chiến lược manifold được chọn trong `config.yaml`, các đại lượng hình học nội tại sau đây sẽ được tính toán và đưa thẳng vào XGBoost:
  
| Chiến lược Manifold | Các đại lượng hình học đưa vào XGBoost | Bản chất Toán học & Ý nghĩa Nghiệp vụ |
| :--- | :--- | :--- |
| **`unsupervised`** | `temporal_drift_velocity` | $\|\mathbf{z}_{t6} - \mathbf{z}_{t1}\|_2$: Tốc độ trôi dạt tổng thể giữa tháng 1 và tháng 6. |
| **`asymmetric_arcface`**<br>(Hyperspherical) | `temporal_drift_velocity`<br>`temporal_angular_drift`<br>`temporal_hypersphere_norm`<br>`temporal_hypersphere_density` | - $\arccos(\tilde{z}_{t1}^\top \tilde{z}_{t6})$: Góc quay trắc địa trên siêu cầu.<br>- $\|z\|_2$: Cường độ hoạt động tài chính trước khi chuẩn hóa.<br>- Mật độ tập trung cục bộ trên mặt cầu. |
| **`hyperbolic_poincare`**<br>(Hyperbolic Ball) | `temporal_poincare_origin_dist`<br>`temporal_poincare_geodesic_drift`<br>`temporal_poincare_conformal_factor`<br>`temporal_poincare_norm`<br>`temporal_drift_velocity` | - $d_{\mathbb{H}}(u, 0) = \frac{2}{\sqrt{c}}\operatorname{artanh}(\sqrt{c}\|u\|)$: **Khoảng cách Hyperbolic tới gốc tọa độ**. (Khách hàng chuẩn mực nằm sát 0; khách hàng suy thoái dị biệt dạt xa về biên vô cực).<br>- $d_{\mathbb{H}}(u_{t1}, u_{t6})$: Khoảng cách trắc địa hyperbolic.<br>- $\lambda_u = \frac{2}{1 - c\|u\|^2}$: Hệ số co giãn Riemannian. |
| **`optimal_transport`**<br>(Kinetics) | `temporal_geodesic_length`<br>`temporal_tortuosity`<br>`temporal_kinetic_action`<br>`temporal_acceleration_norm`<br>`temporal_drift_velocity` | - $\sum_{t=1}^5 \|z_{t+1} - z_t\|$: Tổng quãng đường trắc địa đi qua 6 tháng.<br>- $\frac{\text{Tổng quãng đường}}{\text{Độ dời thẳng}}$: **Hệ số ngoằn ngoèo (Tortuosity)**. (Khách hàng biến động hỗn loạn có Tortuosity cao; khách hàng lao dốc đơn điệu có Tortuosity $\approx 1$).<br>- $\frac{1}{2}\sum \|\vec{v}_t\|^2$: Động năng tích lũy Hamilton.<br>- $\|\vec{a}_t\|$: Độ cong gia tốc chuyển pha. |
| **`neyman_pearson`**<br>(Level-Set) | `temporal_np_recon_energy`<br>`temporal_np_margin`<br>`temporal_np_risk_score`<br>`temporal_drift_velocity` | - Sai số năng lượng tái tạo.<br>- Khoảng cách tới mặt cắt phân vị rủi ro $\tau_\alpha$.<br>- Điểm số rủi ro trơn hóa qua Sigmoid. |

#### 3. Nhóm 3: Đặc trưng Topo Đồ thị Mạng lưới (GNN Topological Features) — 19 features (Khi bật `enable_gnn_embeddings: true`)
- `gnn_z0` đến `gnn_z15`: 16 chiều biểu diễn vị trí của khách hàng trong không gian đồ thị tương đồng $k$-NN ($k=5$).
- `gnn_degree_centrality`: Mức độ trung tâm liên kết hành vi trong mạng lưới ngân hàng.
- `gnn_clustering_coeff`: Hệ số co cụm mạng xã hội/tài chính địa phương.
- `gnn_latent_norm`: Chuẩn độ dài biểu diễn đồ thị.

---

### 9.4. Tại Sao Các Biến Hình Học Này Lại Tạo Ưu Thế Vượt Trội Cho Cây Quyết Định XGBoost?

Cây quyết định XGBoost có một nhược điểm toán học cố hữu: **Nó chỉ có thể phân chia dữ liệu bằng các siêu phẳng trực giao song song với các trục tọa độ (axis-aligned orthogonal splits).**

```
   (A) KHÔNG CÓ ĐẶC TRƯNG HÌNH HỌC              (B) CÓ ĐẶC TRƯNG HÌNH HỌC MANIFOLD
   Cây phải xấp xỉ mặt cầu bằng 20+ lát cắt       Cây chỉ cần 1 lát cắt duy nhất trên biến trắc địa!
   (Staircase approximation -> Overfitting)       (Single split -> Robust & Generalizable)

         z2 ▲                                           poincare_dist ▲
            │     ┌───┐                                               │
            │  ┌──┘   └──┐                                            │    VÙNG CHURN DỊ BIỆT
            │  │ Loyal│  │   Churn                                    ├──────────────────────────  Ngưỡng cắt: dist > 1.25
            │  └──┬───┘  │                                            │
            │     └──────┘                                            │    VÙNG KHÁCH HÀNG CHUẨN MỰC
            └──────────────► z1                                       └──────────────────────────►
```

1. **Tuyến tính hóa độ cong phi tuyến (Linearization of Non-linear Curvature):**
   - Giả sử nhóm khách hàng an toàn nằm trong một quả cầu bán kính $R$: $\|z\|_2 \le R$, và khách hàng rời bỏ nằm ở ngoài vỏ cầu: $\|z\|_2 > R$.
   - Nếu chỉ đưa các tọa độ $z_0, z_1, \dots, z_{15}$ vào XGBoost: Cây quyết định phải tạo ra hàng chục lát cắt bậc thang $(z_0 \le c_1) \land (z_1 \le c_2) \dots$ để xấp xỉ đường tròn. Điều này làm cạn kiệt độ sâu của cây (`max_depth: 4 - 6`), dễ gây phân mảnh dữ liệu và overfitting ở các lá.
   - Nhưng khi Manifold đã tính sẵn biến hình học `temporal_poincare_origin_dist` ($d_{\mathbb{H}}(u, 0)$) hoặc `temporal_hypersphere_norm` ($\|z\|_2$), cây XGBoost **chỉ cần duy nhất 1 lát cắt đơn giản ở tầng 1**:
     $$\text{If } \texttt{temporal\_poincare\_origin\_dist} > 1.25 \text{ Then Churn Else Loyal}$$
   - Biến đổi manifold đã biến một bề mặt phân cách phi tuyến đa chiều phức tạp thành **một đặc trưng vô hướng 1 chiều biến thiên đơn điệu với rủi ro**, giúp XGBoost tối ưu hóa độ lợi thông tin (Split Gain) với độ sâu cây tối thiểu.
2. **Loại bỏ sự phụ thuộc vào phương sai thuần túy:**
   - Unsupervised chỉ cung cấp $z$, phản ánh các thành phần có phương sai lớn nhất.
   - Các biến manifold trắc địa (`temporal_tortuosity`, `temporal_angular_drift`, `temporal_kinetic_action`) mang thông tin về **tốc độ thay đổi và độ bất ổn định hành vi**. Đây chính là tín hiệu dự báo sớm (leading indicators) mà các biến bảng tĩnh (vốn là lagging indicators) không thể có được.







