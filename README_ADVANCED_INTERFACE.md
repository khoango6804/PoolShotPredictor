# 🎱 YOLOv11 Advanced Web Interface

## 📋 Mô tả

Giao diện web nâng cao với khả năng chỉnh sửa chi tiết cho video và real-time detection. Hỗ trợ nhiều thông số tùy chỉnh và phân tích chi tiết.

## 🚀 Cách sử dụng

### 1. Khởi động giao diện nâng cao

```bash
# Cách 1: Script tự động
python start_advanced_interface.py

# Cách 2: Chạy trực tiếp
streamlit run web_interface_advanced.py --server.port 8502
```

### 2. Truy cập giao diện

- **Local URL**: http://localhost:8502
- **Network URL**: http://your-ip:8502

## 🎯 Tính năng nâng cao

### 📸 Image Detection
- **Advanced Parameters**: Confidence, IoU, Max Detections
- **Resize Control**: Tùy chỉnh kích thước ảnh
- **Detailed Analysis**: Thống kê chi tiết từng detection
- **Class Distribution**: Biểu đồ phân bố classes
- **Confidence Analysis**: Phân tích độ tin cậy

### 🎬 Video Processing
- **Frame Control**: 
  - Frame Skip: Xử lý mỗi N frame
  - Max Frames: Giới hạn số frame xử lý
  - Output FPS: Tùy chỉnh FPS đầu ra
- **Resize Factor**: Thay đổi kích thước video
- **Progress Tracking**: Theo dõi tiến trình real-time
- **Frame Statistics**: Thống kê từng frame
- **Performance Metrics**: Tốc độ xử lý

### 📹 Real-time Detection
- **Camera Control**: Chọn camera source
- **FPS Control**: Điều chỉnh FPS real-time
- **Live Processing**: Xử lý trực tiếp từ camera
- **Performance Display**: Hiển thị FPS thực tế

## ⚙️ Advanced Settings

### 🎯 Detection Settings
- **Confidence Threshold**: 0.01 - 1.0 (độ tin cậy)
- **IoU Threshold**: 0.1 - 1.0 (Intersection over Union)
- **Max Detections**: 1 - 100 (số detection tối đa)

### 🎬 Video Processing Settings
- **Output FPS**: 1 - 60 FPS
- **Frame Skip**: 1 - 10 (xử lý mỗi N frame)
- **Max Frames**: 10 - 10000 frames
- **Resize Factor**: 0.1 - 2.0 (thay đổi kích thước)

### 📹 Real-time Settings
- **Enable Real-time**: Bật/tắt chế độ real-time
- **Real-time FPS**: 1 - 30 FPS
- **Buffer Size**: 1 - 10 (kích thước buffer)

### 🤖 Model Selection
- **Ball Detector**: 23 classes billiards balls
- **Table Detector**: 1 class table
- **Pocket Detector**: 10 classes pockets

## 📊 Kết quả chi tiết

### Image Detection Results
- **Original vs Detected**: So sánh ảnh gốc và kết quả
- **Detection Statistics**: Tổng số, max/min/avg confidence
- **Class Distribution**: Biểu đồ phân bố classes
- **Detailed Table**: Bảng chi tiết từng detection
- **Confidence Status**: Phân loại độ tin cậy (High/Medium/Low)

### Video Processing Results
- **Processing Summary**: Tóm tắt quá trình xử lý
- **Performance Metrics**: Tốc độ xử lý, thời gian
- **Frame Statistics**: Thống kê từng frame
- **Download Options**: Tải video đã xử lý

### Real-time Results
- **Live Display**: Hiển thị trực tiếp
- **FPS Counter**: Đếm FPS thực tế
- **Detection Overlay**: Overlay detection boxes
- **Performance Monitoring**: Theo dõi hiệu suất

## 🔧 Yêu cầu hệ thống

### Dependencies
```bash
pip install streamlit ultralytics opencv-python numpy pillow pandas
```

### Hardware
- **GPU**: NVIDIA GPU (recommended)
- **RAM**: 8GB+ (recommended)
- **Camera**: Webcam cho real-time mode
- **Storage**: 5GB+ free space

### Software
- **CUDA**: GPU drivers và CUDA
- **Webcam**: Camera drivers
- **Browser**: Modern web browser

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **Camera not accessible**
   ```bash
   # Kiểm tra camera drivers
   # Thử camera source khác (0, 1, 2)
   ```

2. **Low FPS in real-time**
   ```bash
   # Giảm resize factor
   # Giảm real-time FPS
   # Tăng confidence threshold
   ```

3. **Video processing slow**
   ```bash
   # Tăng frame skip
   # Giảm max frames
   # Giảm resize factor
   ```

4. **Memory issues**
   ```bash
   # Giảm batch size
   # Giảm buffer size
   # Đóng ứng dụng khác
   ```

### Performance Tips

1. **Real-time Optimization**:
   - Giảm resize factor (0.5 - 0.8)
   - Tăng confidence threshold (> 0.5)
   - Giảm real-time FPS (10-15)

2. **Video Processing**:
   - Sử dụng frame skip (2-5)
   - Giới hạn max frames (500-1000)
   - Giảm output FPS (15-30)

3. **Memory Management**:
   - Giảm buffer size (1-3)
   - Giảm max detections (20-30)
   - Đóng browser tabs khác

## 📁 File Structure

```
poolShotPredictor/
├── web_interface_advanced.py      # Advanced web interface
├── start_advanced_interface.py    # Startup script
├── web_interface.py               # Basic web interface
├── start_web_interface.py         # Basic startup script
├── requirements_web.txt           # Dependencies
├── runs/detect/yolo11_billiards_gpu/weights/best.pt  # Ball model
├── table_detector/yolo11_table_detector/weights/best.pt  # Table model
├── pocket_detector/yolo11_pocket_detector/weights/best.pt  # Pocket model
└── README_ADVANCED_INTERFACE.md   # This file
```

## 🎉 Kết quả mong đợi

### Performance
- **Real-time FPS**: 15-30 FPS với GPU
- **Video Processing**: 2-5x real-time speed
- **Detection Accuracy**: 90%+ với confidence 0.3
- **Memory Usage**: < 4GB RAM

### User Experience
- **Intuitive Controls**: Dễ sử dụng
- **Real-time Feedback**: Phản hồi tức thì
- **Detailed Analytics**: Phân tích chi tiết
- **Flexible Settings**: Tùy chỉnh linh hoạt

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra dependencies đã cài đặt
2. Đảm bảo GPU drivers và CUDA
3. Kiểm tra camera permissions
4. Monitor system resources

---

**🎱 YOLOv11 Advanced Billiards Detection System**  
*Powered by Ultralytics YOLOv11 | GPU Optimized | Advanced Controls* 