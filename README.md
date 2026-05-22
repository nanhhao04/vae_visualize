# VAE Clustering & Latent Space Visualization Pipeline

Pipeline huấn luyện Beta-VAE 32D, trích xuất không gian ẩn (latent space), phân cụm K-Means và trực quan hóa phân phối dữ liệu (cho 3 dataset: **CIFAR-10**, **AG News**, và **Speech Commands**).

---

## 🛠️ Bước 0: Thiết lập Dataset hoạt động
Mở file [config.yml](config.yml) và điền tên dataset bạn muốn xử lý:
```yaml
data_name: cifar10  # Chọn một trong ba: cifar10 | agnew | speechcommand
```

---

## 🚀 Các bước chạy chương trình

### Bước 1: Huấn luyện VAE (`training.py`)
Lệnh này sẽ lưu tệp phân phối thô và tự động huấn luyện mô hình Beta-VAE 32D tương ứng trong 3 epochs (nếu chưa có sẵn file `.pth`):
```bash
python training.py
```
* **Đầu ra:** 
  * Lưu trữ cấu hình phân phối làm cơ sở cho bước sau: `X_loaded.npy`, `y_loaded.npy`.
  * Lưu mô hình VAE đã huấn luyện: `beta_vae_<dataset>_latent32.pth`.

---

### Bước 2: Chạy Inference (`inference.py`)
Lệnh này thực hiện rút trích mẫu, đưa qua VAE encoder để lấy đặc trưng không gian ẩn 32D, và tính toán độ đo phân cụm K-Means:
```bash
python inference.py
```
* **Đầu ra:** Tạo tệp lưu tọa độ và nhãn dự đoán K-Means: `results_inferred.pkl`.

---

### Bước 3: Vẽ biểu đồ Phân phối (`gen_plot.py`)
Sử dụng dữ liệu trung gian ở Bước 2 để vẽ biểu đồ so sánh giữa phân cụm K-Means và nhãn thực tế:
```bash
python gen_plot.py
```
* **Đầu ra:** File biểu đồ định dạng PDF chất lượng cao được lưu tại thư mục tương ứng:
  * **Đường dẫn:** `<dataset>/kmeans_latent_32d_single_8x6.pdf`
  * **Nội dung:** Biểu đồ phân tán 2D (PCA). Trong đó **Màu sắc** (Đỏ/Xanh lá/Vàng) biểu diễn cụm dự đoán bởi K-Means, còn **Hình dạng** (Tròn/Vuông/Tam giác) biểu diễn nhãn Ground Truth thực tế của phân phối.

---

### 🎨 Bước 4 (Tùy chọn): Vẽ biểu đồ không gian mẫu ảnh cá nhân (`visualize_per_image.py`)
Vẽ trực quan hóa đặc trưng không gian ẩn 32D của chính các mẫu dữ liệu đơn lẻ (lấy chính xác 50 ảnh/mẫu mỗi nhãn):
```bash
python visualize_per_image.py
```
* **Đầu ra:** File PDF cực kỳ tinh gọn, không tiêu đề rườm rà, chú thích chỉ ghi nhãn:
  * **Đường dẫn:** `<dataset>/visualize_latent32.pdf`
