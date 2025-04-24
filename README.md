Chắc chắn rồi, đây là một tệp `README.md` đầy đủ và chi tiết cho dự án DViT Face Landmark Detection của bạn.

```markdown
# Phát hiện Điểm mốc Khuôn mặt bằng DViT (Dual Vision Transformer)

Dự án này là một triển khai và minh họa mô hình **Dual Vision Transformer (DViT)** cho bài toán định vị điểm mốc trên khuôn mặt, được huấn luyện và đánh giá trên tập dữ liệu **300W**. Nó dựa trên kiến trúc được đề xuất trong bài báo "DViT: Dual-View Vision Transformer for Facial Landmark Detection".

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Tùy chọn: Thêm license -->

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc Dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
  - [Yêu cầu](#yêu-cầu)
  - [Các bước cài đặt](#các-bước-cài-đặt)
- [Hướng dẫn Sử dụng](#hướng-dẫn-sử-dụng)
  - [1. Chuẩn bị Dữ liệu (Tạo Danh sách Tệp)](#1-chuẩn-bị-dữ-liệu-tạo-danh-sách-tệp)
  - [2. Huấn luyện Mô hình](#2-huấn-luyện-mô-hình)
  - [3. Đánh giá Mô hình](#3-đánh-giá-mô-hình)
  - [4. Trực quan hóa Dự đoán (Script)](#4-trực-quan-hóa-dự-đoán-script)
  - [5. Chạy Demo Streamlit](#5-chạy-demo-streamlit)
- [Cấu hình](#cấu-hình)
- [Kết quả](#kết-quả)
- [Demo Streamlit](#demo-streamlit)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Giấy phép](#giấy-phép)
- [Đóng góp](#đóng-góp)

## Giới thiệu

Định vị điểm mốc khuôn mặt là một nhiệm vụ cơ bản trong thị giác máy tính nhằm xác định vị trí các điểm ngữ nghĩa chính trên khuôn mặt (ví dụ: khóe mắt, chóp mũi, khóe miệng). Mô hình DViT tận dụng sức mạnh của kiến trúc Transformer bằng cách sử dụng cơ chế chú ý kép (Dual-View Attention): **Spatial ViT** tập trung vào mối quan hệ không gian giữa các vùng ảnh và **Channel ViT** nắm bắt sự phụ thuộc giữa các kênh đặc trưng. Dự án này cung cấp một triển khai thực tế của kiến trúc tầng **Cascaded DViT** trên tập dữ liệu 300W phổ biến.

## Tính năng

*   Triển khai kiến trúc **Cascaded DViT** với mạng xương sống ResNet-18.
*   Sử dụng các thành phần **SpatialViT** và **ChannelViT**.
*   **Long-Short Context (LSC) Fusion** để kết hợp đặc trưng đa cấp.
*   **Giám sát Trung gian (Intermediate Supervision)** trong quá trình huấn luyện.
*   Hàm mất mát kết hợp **Smooth L1 Loss** (cho tọa độ) và **Adaptive Wing (Awing) Loss** (cho heatmap).
*   Xử lý và tăng cường dữ liệu cho tập **300W** bằng Albumentations.
*   Quy trình huấn luyện đầy đủ với Optimizer (Adam), Scheduler (StepLR).
*   Đánh giá hiệu năng bằng các chỉ số tiêu chuẩn: **NME** (Full, Common, Challenging), **FR@0.10**, **AUC@0.10**, và vẽ biểu đồ **CED**.
*   Công cụ trực quan hóa kết quả dự đoán và heatmap (qua script và ứng dụng Streamlit).
*   Mã nguồn được tổ chức theo module, dễ hiểu và mở rộng.
*   Ứng dụng **Streamlit** tương tác để demo và phân tích mô hình.

## Cấu trúc Dự án

```
DViT_Face_Landmarks/
├── data/                     # Xử lý tập dữ liệu và tiền xử lý
│   ├── __init__.py
│   ├── dataset_300w.py         # Lớp FaceLandmark300WDataset
│   └── generate_file_list.py   # Script tạo danh sách file train/test
├── models/                   # Định nghĩa kiến trúc mô hình
│   ├── __init__.py
│   ├── vit_components.py       # MLP, Attention, ViTBlock
│   ├── dvit_modules.py         # SpatialViT, ChannelViT, khối DViT
│   └── cascaded_dvit.py        # PredictionHead, CascadedDViT (mô hình chính)
├── loss/                     # Triển khai các hàm mất mát
│   ├── __init__.py
│   └── losses.py               # AwingLoss, TotalLoss
├── utils/                    # Các tiện ích trợ giúp (bản đồ nhiệt, đánh giá, trực quan hóa)
│   ├── __init__.py
│   ├── heatmap_utils.py        # generate_*, soft_argmax
│   ├── visualization.py        # draw_landmarks, visualize_predictions_heatmaps
│   └── evaluation_metrics.py   # calculate_nme, calculate_fr_auc, evaluate_model, plot_ced
├── config/                   # Tệp cấu hình (siêu tham số, đường dẫn, hằng số)
│   └── __init__.py
│   └── config.py               # Các siêu tham số, đường dẫn, hằng số
├── scripts/                  # Các script có thể thực thi để huấn luyện, đánh giá, v.v.
│   ├── train.py                # Script huấn luyện chính
│   ├── evaluate.py             # Script đánh giá định lượng
│   └── visualize_predictions.py # Script độc lập để trực quan hóa kết quả
├── results/                    # Thư mục lưu mô hình, log, biểu đồ (được tạo bởi script)
│   └── DViT_300W_Demo/         # Thư mục con thường đặt tên theo mô hình/thí nghiệm
├── streamlit_app.py          # Ứng dụng Streamlit để trực quan hóa
├── report.md                   # Báo cáo chi tiết dự án (tiếng Việt)
├── requirements.txt            # Các gói phụ thuộc Python
└── README.md                 # Tệp này (hướng dẫn tổng quan)
```

## Cài đặt

### Yêu cầu

*   Python 3.8+
*   PyTorch 1.10+
*   Torchvision
*   OpenCV (ví dụ: `opencv-python-headless`)
*   NumPy
*   Matplotlib
*   Albumentations
*   tqdm
*   scikit-learn (cho `auc`)
*   Streamlit
*   Pillow (cho Streamlit)

### Các bước cài đặt

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd DViT_Face_Landmarks
    ```

2.  **Tạo Môi trường Ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    # venv\Scripts\activate
    ```

3.  **Cài đặt các Gói phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải Tập dữ liệu 300W:**
    *   Tải về bộ dữ liệu `ibug_300W_large_face_landmark_dataset` (bạn có thể tìm kiếm trên mạng hoặc sử dụng các nguồn cung cấp bộ dữ liệu học thuật).
    *   Giải nén và đặt thư mục dữ liệu vào một vị trí bạn chọn (ví dụ: `/data/datasets/ibug_300W_large_face_landmark_dataset`).

5.  **Cấu hình Đường dẫn:**
    *   Mở tệp `config/config.py`.
    *   **Quan trọng:** Cập nhật biến `ROOT_DATA_DIR` thành đường dẫn tuyệt đối đến thư mục dữ liệu 300W bạn vừa tải về ở bước 4.
    *   Xem xét và điều chỉnh các cấu hình khác nếu cần (ví dụ: `GPU_IDS`, `BATCH_SIZE`, `SAVE_DIR`).

## Hướng dẫn Sử dụng

### 1. Chuẩn bị Dữ liệu (Tạo Danh sách Tệp)

Chạy script sau để tạo các tệp danh sách `train` và `test` cần thiết cho `DataLoader`. Script này sẽ quét thư mục dữ liệu 300W đã cấu hình.

```bash
python data/generate_file_list.py --root_dir /path/to/your/ibug_300W_large_face_landmark_dataset --output_dir .
```

*   Thay thế `/path/to/your/ibug_300W_large_face_landmark_dataset` bằng đường dẫn thực tế.
*   Lệnh này tạo ra `generated_300w_train_list.txt` và `generated_300w_test_list.txt` trong thư mục hiện tại. Đảm bảo các tên tệp này khớp với `TRAIN_LIST_FILE` và `TEST_LIST_FILE` trong `config/config.py`.

### 2. Huấn luyện Mô hình

Chạy script huấn luyện chính:

```bash
python scripts/train.py
```

*   Script sẽ đọc cấu hình, tải dữ liệu, khởi tạo mô hình và bắt đầu quá trình huấn luyện.
*   Tiến trình huấn luyện, mất mát sẽ được hiển thị trên console.
*   Các checkpoint (đặc biệt là `best_model.pth`), log (nếu có) và biểu đồ mất mát (`loss_curve.png`) sẽ được lưu vào thư mục `results/EXPERIMENT_NAME` (ví dụ: `results/DViT_300W_Demo/`).

### 3. Đánh giá Mô hình

Đánh giá hiệu năng của mô hình đã huấn luyện trên tập kiểm tra 300W:

```bash
python scripts/evaluate.py --checkpoint path/to/your/best_model.pth --dataset_name 300w
```

*   Thay thế `path/to/your/best_model.pth` bằng đường dẫn đến checkpoint bạn muốn đánh giá (thường là `results/EXPERIMENT_NAME/best_model.pth`).
*   Script sẽ tính toán và in ra các chỉ số NME (Full, Common, Challenging), FR@0.10, AUC@0.10.
*   Biểu đồ CED (`ced_curve_300w.png`) sẽ được vẽ và lưu vào thư mục `results/`.

### 4. Trực quan hóa Dự đoán (Script)

Xem xét định tính kết quả dự đoán và heatmap trên một số mẫu ngẫu nhiên từ tập kiểm tra:

```bash
python scripts/visualize_predictions.py --checkpoint path/to/your/best_model.pth --dataset_name 300w --num_samples 10
```

*   Thay thế đường dẫn checkpoint.
*   Điều chỉnh `--num_samples` để thay đổi số lượng ảnh hiển thị.
*   Script sẽ hiển thị một cửa sổ Matplotlib với các ảnh, landmarks (GT và dự đoán), và heatmaps. Hình ảnh này cũng sẽ được lưu vào thư mục `results/` (ví dụ: `prediction_visualization.png`).

### 5. Chạy Demo Streamlit

Khởi chạy ứng dụng web tương tác để trực quan hóa:

```bash
streamlit run streamlit_app.py
```

*   Ứng dụng sẽ mở trong trình duyệt của bạn.
*   Làm theo hướng dẫn trên giao diện để tải mô hình, chọn ảnh và xem kết quả.

## Cấu hình

Các siêu tham số chính, đường dẫn dữ liệu, cấu hình huấn luyện và đánh giá được quản lý tập trung trong tệp `config/config.py`. Vui lòng xem xét và chỉnh sửa tệp này trước khi chạy các script:

*   **Đường dẫn:** `ROOT_DATA_DIR`, `TRAIN_LIST_FILE`, `TEST_LIST_FILE`, `SAVE_DIR`.
*   **Thiết bị:** `GPU_IDS`.
*   **Kiến trúc Mô hình:** `NUM_BLOCKS`, `NUM_LANDMARKS`, `DVIT_INTERNAL_CHANNELS`, `VIT_EMBED_DIM`, v.v.
*   **Huấn luyện:** `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, `WEIGHT_DECAY`.
*   **Loss:** `LOSS_HEATMAP_BETA`, `LOSS_INTERMEDIATE_WEIGHT_W`, các tham số Awing.
*   **Đánh giá:** `EVAL_BATCH_SIZE`, `EVAL_FAILURE_THRESHOLD`.

## Kết quả

Sau khi chạy các script huấn luyện và đánh giá, các kết quả sau sẽ được tạo ra:

*   **Mô hình đã huấn luyện:** Các tệp `.pth` trong thư mục `results/EXPERIMENT_NAME`.
*   **Chỉ số Định lượng:** NME, FR, AUC được in ra console bởi `evaluate.py`.
*   **Biểu đồ:** Đường cong mất mát (`loss_curve.png`) và đường cong CED (`ced_curve_*.png`) được lưu trong thư mục `results/`.
*   **Trực quan hóa:** Ảnh kết quả dự đoán (`prediction_visualization.png`) được lưu trong thư mục `results/`.

Tham khảo tệp `report.md` để xem phân tích chi tiết về các kết quả thu được trong quá trình chạy thử nghiệm.

## Demo Streamlit

Ứng dụng `streamlit_app.py` cung cấp một giao diện đồ họa người dùng (GUI) để:

1.  Tải mô hình DViT đã huấn luyện.
2.  Chọn ảnh từ tập kiểm tra 300W hoặc tải lên ảnh tùy chỉnh.
3.  Hiển thị ảnh với điểm mốc dự đoán (đỏ) và ground truth (xanh lá).
4.  Hiển thị các bản đồ nhiệt (heatmaps) dự đoán cho các điểm mốc được chọn.

Đây là một công cụ hữu ích để phân tích định tính, gỡ lỗi và minh họa hiệu năng của mô hình.

## Tài liệu tham khảo

*   Zhang, J., et al. (2023). *DViT: Dual-View Vision Transformer for Facial Landmark Detection*. (Cần tìm trích dẫn/link đầy đủ)
*   Sagonas, C., et al. (2013). *300 faces in-the-wild challenge: The first facial landmark localization challenge*. IEEE ICCV Workshops.
*   Wang, X., et al. (2018). *Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression*. IEEE ICCV.
*   Thư viện PyTorch: Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
*   Thư viện Albumentations: Buslaev, A., et al. (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information.
*   Thư viện Streamlit: [https://streamlit.io/](https://streamlit.io/)

## Giấy phép

Dự án này được cấp phép theo Giấy phép MIT. Xem tệp `LICENSE` (nếu có) để biết chi tiết.

## Đóng góp

Chúng tôi hoan nghênh các đóng góp! Vui lòng tạo Pull Request hoặc Issue nếu bạn có đề xuất cải thiện hoặc sửa lỗi.
```

Hy vọng tệp README này đầy đủ và hữu ích cho dự án của bạn!