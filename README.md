# Phát hiện Điểm mốc Khuôn mặt bằng DViT

## Tổng quan

Định vị điểm mốc khuôn mặt là một nhiệm vụ cơ bản trong thị giác máy tính. Mô hình DViT tận dụng kiến trúc Transformer để nắm bắt cả mối quan hệ không gian giữa các vùng ảnh (Spatial ViT) và sự phụ thuộc giữa các kênh đặc trưng (Channel ViT). Dự án này cung cấp:

*   Triển khai kiến trúc Cascaded DViT bằng PyTorch.
*   Quy trình xử lý và chuẩn bị tập dữ liệu 300W.
*   Script huấn luyện với hàm mất mát kết hợp (Awing Loss + Smooth L1) và giám sát trung gian.
*   Script đánh giá hiệu năng sử dụng các chỉ số tiêu chuẩn (NME, FR@0.10, AUC@0.10).
*   Script trực quan hóa tĩnh kết quả dự đoán và bản đồ nhiệt.

## Tính năng chính

*   **Kiến trúc DViT:** Triển khai các thành phần cốt lõi: SpatialViT, ChannelViT, DViT Block, Cascaded DViT.
*   **Backbone ResNet:** Sử dụng ResNet-18 làm mạng xương sống trích xuất đặc trưng ban đầu.
*   **LSC Fusion:** Kết hợp ngữ cảnh dài-ngắn giữa đặc trưng backbone và đặc trưng từ các khối DViT trước đó.
*   **Giám sát Trung gian:** Huấn luyện mô hình dựa trên đầu ra của tất cả các khối DViT.
*   **Hàm Mất mát Nâng cao:** Sử dụng Awing Loss cho bản đồ nhiệt và Smooth L1 Loss cho tọa độ.
*   **Hỗ trợ 300W:** Cung cấp lớp `Dataset` và script chuẩn bị dữ liệu riêng cho tập 300W (68 điểm mốc).
*   **Đánh giá Tiêu chuẩn:** Tính toán NME (Full, Common, Challenging), FR, và AUC.

## Cài đặt

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd DViT_Face_Landmarks
    ```

2.  **Tạo và Kích hoạt Môi trường ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    # .\venv\Scripts\activate
    ```

3.  **Cài đặt các Gói phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: Đảm bảo bạn đã cài đặt PyTorch với phiên bản CUDA phù hợp với hệ thống nếu sử dụng GPU. Truy cập [pytorch.org](https://pytorch.org/) để xem hướng dẫn cài đặt cụ thể.)*

4.  **Tải Tập dữ liệu 300W:**
    *   Tải tập dữ liệu `ibug_300W_large_face_landmark_dataset` từ nguồn chính thức hoặc các nguồn đáng tin cậy khác.

5.  **Cấu hình Dự án:**
    *   Mở tệp `config/config.py`.
    *   **Quan trọng:** Cập nhật biến `ROOT_DATA_DIR` thành đường dẫn tuyệt đối đến thư mục `ibug_300W_large_face_landmark_dataset` bạn vừa tải về.
        ```python
        # Ví dụ:
        ROOT_DATA_DIR = '/path/to/datasets/ibug_300W_large_face_landmark_dataset'
        ```
    *   Kiểm tra và đảm bảo các đường dẫn `TRAIN_LIST_FILE` và `TEST_LIST_FILE` (ví dụ: `'./generated_300w_train_list.txt'`) là nơi bạn muốn lưu/đọc các danh sách tệp (thường là thư mục gốc dự án).
    *   Cấu hình `GPU_IDS`:
        *   Để sử dụng CPU: `GPU_IDS = []`
        *   Để sử dụng GPU đầu tiên (ID 0): `GPU_IDS = [0]`
        *   Để sử dụng GPU 0 và 1 với DataParallel: `GPU_IDS = [0, 1]`
    *   (Tùy chọn) Điều chỉnh các siêu tham số khác như `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE` nếu cần.

## Hướng dẫn Sử dụng (300W)

Thực hiện các lệnh sau từ thư mục gốc của dự án (`DViT_Face_Landmarks/`).

1.  **Chuẩn bị Danh sách Tệp Dữ liệu:**
    *   Chạy script để tạo tệp `generated_300w_train_list.txt` và `generated_300w_test_list.txt`.
    ```bash
    python data/generate_file_list.py --root_dir /path/to/ibug_300W_large_face_landmark_dataset --output_dir .
    ```
    *   *(Thay thế `/path/to/ibug_300W_large_face_landmark_dataset` bằng đường dẫn thực tế)*.
    *   *(Đảm bảo `--output_dir .` tạo tệp trong thư mục gốc, khớp với cấu hình `config.py`)*.

2.  **Huấn luyện Mô hình:**
    *   Bắt đầu quá trình huấn luyện sử dụng cấu hình trong `config/config.py`.
    ```bash
    python scripts/train.py
    ```
    *   Tiến trình huấn luyện sẽ được hiển thị trên console.
    *   Các checkpoint (đặc biệt là `best_model.pth`) và biểu đồ mất mát (`loss_curve.png`) sẽ được lưu vào thư mục `results/EXPERIMENT_NAME` (ví dụ: `results/DViT_300W_Demo`).

3.  **Đánh giá Mô hình:**
    *   Đánh giá hiệu năng của mô hình đã huấn luyện trên tập kiểm tra 300W.
    ```bash
    python scripts/evaluate.py --checkpoint results/DViT_300W_Demo/best_model.pth --dataset_name 300w
    ```
    *   *(Thay thế đường dẫn `--checkpoint` nếu cần)*.
    *   Các chỉ số NME, FR, AUC sẽ được in ra.
    *   Biểu đồ CED (`ced_curve_300w.png`) sẽ được lưu vào thư mục `results/`.