# 🎱 YOLOv11 Billiards Detection Web Interface

## 📋 Mô tả

Giao diện web local để upload ảnh và video, sau đó chạy detection với model YOLOv11 đã được train. Hỗ trợ real-time detection với GPU optimization.

## 🚀 Cách sử dụng

### 1. Khởi động giao diện web

```bash
# Cách 1: Sử dụng script tự động
python start_web_interface.py

# Cách 2: Chạy trực tiếp
streamlit run web_interface.py
```

### 2. Truy cập giao diện

- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501

### 3. Upload và xử lý

1. **Upload file**: Chọn ảnh (JPG, PNG) hoặc video (MP4, AVI, MOV, MKV)
2. **Điều chỉnh confidence**: Sử dụng slider trong sidebar
3. **Xem kết quả**: Detection boxes và thống kê chi tiết
4. **Download**: Tải về ảnh/video đã được xử lý

## 🎯 Tính năng

### 📸 Xử lý ảnh
- Upload ảnh JPG, PNG
- Real-time detection với YOLOv11
- Hiển thị detection boxes
- Thống kê chi tiết (số lượng detections, confidence)
- Phân bố classes
- Download ảnh kết quả

### 🎬 Xử lý video
- Upload video MP4, AVI, MOV, MKV
- Xử lý từng frame
- Progress tracking
- Thống kê tổng quan
- Download video đã xử lý

### ⚙️ Cài đặt
- **Confidence Threshold**: Điều chỉnh độ tin cậy (0.01 - 1.0)
- **Model Info**: Thông tin model và performance
- **Real-time Processing**: GPU optimized

## 📊 Kết quả

### Thống kê detection
- **Total Detections**: Tổng số bóng được detect
- **Max/Min/Avg Confidence**: Độ tin cậy cao nhất/thấp nhất/trung bình
- **Class Distribution**: Phân bố các loại bóng
- **Detailed Detections**: Chi tiết từng detection

### Visualization
- **Original vs Detected**: So sánh ảnh gốc và kết quả
- **Detection Boxes**: Bounding boxes với labels
- **Confidence Scores**: Hiển thị độ tin cậy
- **Class Labels**: Nhãn loại bóng

## 🔧 Yêu cầu hệ thống

### Dependencies
```bash
pip install -r requirements_web.txt
```

### Model
- YOLOv11 model đã được train: `runs/detect/yolo11_billiards_gpu/weights/best.pt`
- 23 classes billiards balls
- GPU optimized

### Hardware
- **GPU**: NVIDIA GPU (recommended)
- **RAM**: 8GB+ (recommended)
- **Storage**: 2GB+ free space

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **Model not found**
   ```bash
   # Train model trước
   python train_yolo11.py
   ```

2. **Dependencies missing**
   ```bash
   # Cài đặt dependencies
   pip install -r requirements_web.txt
   ```

3. **Port 8501 in use**
   ```bash
   # Sử dụng port khác
   streamlit run web_interface.py --server.port 8502
   ```

4. **GPU not available**
   - Model sẽ tự động chuyển sang CPU
   - Performance có thể chậm hơn

### Performance tips

1. **GPU Usage**: Đảm bảo CUDA được cài đặt
2. **Video Processing**: Giới hạn 100 frames cho demo
3. **Memory**: Đóng các ứng dụng khác khi xử lý video lớn
4. **Confidence**: Tăng confidence để giảm false positives

## 📁 File structure

```
poolShotPredictor/
├── web_interface.py          # Main web interface
├── start_web_interface.py    # Startup script
├── requirements_web.txt      # Dependencies
├── runs/detect/yolo11_billiards_gpu/weights/best.pt  # Model
└── README_WEB_INTERFACE.md   # This file
```

## 🎉 Kết quả mong đợi

- **Detection Accuracy**: 90%+ với confidence 0.3
- **Processing Speed**: Real-time với GPU
- **User Experience**: Intuitive web interface
- **Output Quality**: High-quality detection boxes

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Dependencies đã được cài đặt đầy đủ
2. Model YOLOv11 đã được train
3. GPU drivers và CUDA
4. Port 8501 không bị chiếm

---

**🎱 YOLOv11 Billiards Detection System**  
*Powered by Ultralytics YOLOv11 | GPU Optimized* 