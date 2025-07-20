# 🎱 YOLOv11 BILLIARDS DETECTION - QUICK START

## 🚀 Khởi động nhanh

### 1. Giao diện cơ bản (Dễ sử dụng)
```bash
python start_web_interface.py
```
**Truy cập**: http://localhost:8501

### 2. Giao diện nâng cao (Đầy đủ tính năng)
```bash
python start_advanced_interface.py
```
**Truy cập**: http://localhost:8503

## 📸 Cách sử dụng cơ bản

### Upload ảnh:
1. Mở trình duyệt → http://localhost:8501
2. Chọn "Choose an image file"
3. Upload ảnh JPG/PNG
4. Điều chỉnh confidence (0.01-1.0)
5. Xem kết quả detection

### Upload video:
1. Chọn "Choose a video file"
2. Upload video MP4/AVI/MOV/MKV
3. Điều chỉnh tham số
4. Xem video đã xử lý

## 🎬 Tính năng nâng cao

### Video Processing:
- **Output FPS**: 1-60 (tốc độ video output)
- **Frame Skip**: 1-10 (bỏ qua N frame)
- **Max Frames**: 10-10000 (số frame tối đa)
- **Save Processed**: Lưu video đã xử lý

### Real-time Detection:
- **Camera Source**: 0,1,2 (chọn camera)
- **Real-time FPS**: 1-30 (tốc độ real-time)
- **Show FPS**: Hiển thị FPS

### Model Selection:
- **Ball Detector**: 23 loại bóng billiards
- **Table Detector**: Bàn billiards (99.5% accuracy)
- **Pocket Detector**: 10 loại lỗ bàn

## ⚙️ Tham số quan trọng

### Detection Settings:
- **Confidence**: 0.3-0.7 (khuyến nghị)
- **IoU**: 0.5 (mặc định)
- **Max Detections**: 50 (mặc định)

### Performance:
- **Frame Skip**: 2-3 (cho video dài)
- **Max Frames**: 1000 (cho video ngắn)
- **Resize Factor**: 1.0 (gốc), 0.5 (nhanh hơn)

## 🔧 Troubleshooting

### Lỗi thường gặp:
```bash
# Kiểm tra models
python check_model_status.py

# Test tất cả models
python test_all_models.py
```

### Port bị chiếm:
- Dừng process cũ
- Hoặc đổi port trong config

### Camera không hoạt động:
- Kiểm tra camera được kết nối
- Thay đổi Camera Source (0,1,2)

## 📊 Kết quả

### Image Detection:
- Ảnh gốc + Ảnh có detection
- Số lượng detections
- Download kết quả

### Video Processing:
- Video đã xử lý
- Thống kê frame-by-frame
- Performance metrics

### Real-time:
- Live detection
- FPS counter
- Real-time processing

## 🎯 Tips

### Cho kết quả tốt:
- Confidence: 0.3-0.7
- IoU: 0.5
- Sử dụng đúng model cho task

### Cho performance tốt:
- Frame Skip: 2-3
- Max Frames: 1000
- Resize Factor: 0.5-1.0

### Cho real-time:
- FPS: 15-20
- Buffer Size: 3
- Tắt features không cần

---

## 📖 Hướng dẫn chi tiết

Xem file `HUONG_DAN_SU_DUNG.md` để biết thêm chi tiết!

## 🎉 Sẵn sàng sử dụng!

Hệ thống đã được cấu hình sẵn với 3 models:
- ✅ Ball Detector (5.2 MB)
- ✅ Table Detector (5.9 MB) 
- ✅ Pocket Detector (6.0 MB)

**Truy cập ngay**: http://localhost:8501 hoặc http://localhost:8503 