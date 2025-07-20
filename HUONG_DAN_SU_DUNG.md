# 🎱 HƯỚNG DẪN SỬ DỤNG YOLOv11 ADVANCED BILLIARDS DETECTION

## 📋 Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Cài đặt](#cài-đặt)
3. [Khởi động hệ thống](#khởi-động-hệ-thống)
4. [Giao diện cơ bản](#giao-diện-cơ-bản)
5. [Giao diện nâng cao](#giao-diện-nâng-cao)
6. [Troubleshooting](#troubleshooting)

## 🎯 Giới thiệu

Hệ thống YOLOv11 Advanced Billiards Detection là một ứng dụng web cho phép:
- **Phát hiện bóng billiards** trong ảnh và video
- **Phát hiện bàn billiards** với độ chính xác 99.5%
- **Phát hiện lỗ bàn** với 10 loại lỗ khác nhau
- **Xử lý video real-time** với camera
- **Điều chỉnh tham số** chi tiết cho detection

## ⚙️ Cài đặt

### Yêu cầu hệ thống:
- Windows 10/11
- Python 3.8+
- GPU NVIDIA (khuyến nghị)
- RAM: 8GB+ (16GB+ khuyến nghị)

### Cài đặt dependencies:
```bash
pip install ultralytics streamlit opencv-python pillow pandas numpy
```

## 🚀 Khởi động hệ thống

### 1. Giao diện cơ bản (Basic Interface)
```bash
python start_web_interface.py
```
- **URL**: http://localhost:8501
- **Tính năng**: Upload ảnh/video cơ bản

### 2. Giao diện nâng cao (Advanced Interface)
```bash
python start_advanced_interface.py
```
- **URL**: http://localhost:8503
- **Tính năng**: Đầy đủ controls cho video và real-time

## 📸 Giao diện cơ bản

### Cách sử dụng:
1. **Mở trình duyệt**: Truy cập http://localhost:8501
2. **Upload file**: Chọn ảnh (JPG, PNG) hoặc video (MP4, AVI, MOV, MKV)
3. **Điều chỉnh confidence**: Kéo thanh trượt (0.01 - 1.0)
4. **Xem kết quả**: Hệ thống hiển thị ảnh gốc và ảnh có detection
5. **Tải xuống**: Nhấn nút "Download Result"

### Tính năng:
- ✅ Upload ảnh/video
- ✅ Điều chỉnh confidence threshold
- ✅ Hiển thị số lượng detections
- ✅ Download kết quả

## 🎬 Giao diện nâng cao

### Cách sử dụng:

#### 1. **Image Detection (Phát hiện ảnh)**
- Chọn mode "📸 Image Detection"
- Upload ảnh
- Điều chỉnh tham số trong sidebar:
  - **Confidence Threshold**: 0.01 - 1.0
  - **IoU Threshold**: 0.1 - 1.0
  - **Max Detections**: 1 - 100
  - **Resize Factor**: 0.1 - 2.0

#### 2. **Video Processing (Xử lý video)**
- Chọn mode "🎬 Video Processing"
- Upload video
- Điều chỉnh tham số:
  - **Output FPS**: 1 - 60
  - **Frame Skip**: 1 - 10 (xử lý mỗi N frame)
  - **Max Frames**: 10 - 10000
  - **Save Processed**: Bật/tắt lưu video
  - **Show Progress**: Hiển thị tiến trình
  - **Show Stats**: Hiển thị thống kê

#### 3. **Real-time Detection (Phát hiện real-time)**
- Chọn mode "📹 Real-time Detection"
- Bật "Enable Real-time Mode" trong sidebar
- Điều chỉnh:
  - **Camera Source**: 0, 1, 2 (chọn camera)
  - **Real-time FPS**: 1 - 30
  - **Show FPS**: Hiển thị FPS
- Nhấn "🎥 Start Real-time Detection"

### Tham số nâng cao:

#### **Detection Settings:**
- **Confidence Threshold**: Ngưỡng tin cậy (0.01 - 1.0)
- **IoU Threshold**: Ngưỡng IoU cho NMS (0.1 - 1.0)
- **Max Detections**: Số lượng detection tối đa (1 - 100)

#### **Video Processing:**
- **Output FPS**: FPS của video output (1 - 60)
- **Frame Skip**: Bỏ qua N frame (1 - 10)
- **Max Frames**: Số frame tối đa xử lý (10 - 10000)
- **Resize Factor**: Tỷ lệ resize (0.1 - 2.0)

#### **Real-time Settings:**
- **Enable Real-time Mode**: Bật/tắt chế độ real-time
- **Real-time FPS**: FPS cho real-time (1 - 30)
- **Buffer Size**: Kích thước buffer (1 - 10)

#### **Model Selection:**
- **Ball Detector**: Phát hiện 23 loại bóng billiards
- **Table Detector**: Phát hiện bàn billiards (99.5% mAP50)
- **Pocket Detector**: Phát hiện 10 loại lỗ bàn

## 📊 Kết quả và thống kê

### Image Detection:
- **Original Image**: Ảnh gốc
- **Detection Result**: Ảnh có detection
- **Advanced Statistics**: Thống kê chi tiết
- **Class Distribution**: Phân bố các class
- **Detailed Detections**: Bảng chi tiết detections

### Video Processing:
- **Processing Summary**: Tóm tắt xử lý
- **Frame Statistics**: Thống kê theo frame
- **Performance Metrics**: Hiệu suất xử lý
- **Download Processed Video**: Tải video đã xử lý

### Real-time Detection:
- **Live Detection**: Phát hiện trực tiếp
- **FPS Counter**: Hiển thị FPS
- **Real-time Processing**: Xử lý real-time

## 🔧 Troubleshooting

### Lỗi thường gặp:

#### 1. **Port đã được sử dụng**
```
Port 8501/8503 is already in use
```
**Giải pháp:**
- Dừng các process đang chạy
- Hoặc thay đổi port trong file config

#### 2. **Model không tìm thấy**
```
No models found!
```
**Giải pháp:**
```bash
python check_model_status.py
```
- Kiểm tra models có sẵn
- Train lại models nếu cần

#### 3. **Camera không hoạt động**
```
Cannot access camera
```
**Giải pháp:**
- Kiểm tra camera được kết nối
- Thay đổi Camera Source (0, 1, 2)
- Cấp quyền truy cập camera

#### 4. **Video không mở được**
```
Cannot open video file
```
**Giải pháp:**
- Kiểm tra định dạng video (MP4, AVI, MOV, MKV)
- Kiểm tra file không bị hỏng
- Thử video khác

#### 5. **Performance chậm**
**Giải pháp:**
- Giảm Max Frames
- Tăng Frame Skip
- Giảm Resize Factor
- Sử dụng GPU nếu có

### Tối ưu hiệu suất:

#### **Cho Image Detection:**
- Giảm Max Detections
- Tăng Confidence Threshold
- Giảm Resize Factor

#### **Cho Video Processing:**
- Tăng Frame Skip
- Giảm Max Frames
- Giảm Output FPS
- Tắt Show Progress/Stats

#### **Cho Real-time Detection:**
- Giảm Real-time FPS
- Giảm Resize Factor
- Tắt Show FPS

## 📞 Hỗ trợ

### Kiểm tra status hệ thống:
```bash
python check_model_status.py
```

### Test models:
```bash
python test_all_models.py
```

### Monitor training:
```bash
python monitor_training.py
```

### Logs và debugging:
- Kiểm tra console output
- Xem logs trong terminal
- Kiểm tra file models tồn tại

## 🎯 Tips sử dụng

### **Cho kết quả tốt nhất:**
1. **Image Detection**: Confidence 0.3-0.7, IoU 0.5
2. **Video Processing**: Frame Skip 2-3, Max Frames 1000
3. **Real-time**: FPS 15-20, Buffer Size 3

### **Cho performance tốt nhất:**
1. Sử dụng GPU NVIDIA
2. Giảm resolution nếu cần
3. Tăng frame skip cho video dài
4. Tắt các tính năng không cần thiết

### **Cho accuracy tốt nhất:**
1. Sử dụng Ball Detector cho bóng
2. Sử dụng Table Detector cho bàn
3. Sử dụng Pocket Detector cho lỗ
4. Điều chỉnh confidence phù hợp

---

## 🎉 Chúc bạn sử dụng hiệu quả!

Hệ thống YOLOv11 Advanced Billiards Detection đã sẵn sàng phục vụ bạn với đầy đủ tính năng phát hiện bóng, bàn và lỗ billiards chuyên nghiệp! 