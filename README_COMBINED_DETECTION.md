# 🎱 Billiards Combined Detection System

Hệ thống nhận dạng billiards kết hợp với khả năng phát hiện bóng, bàn và các loại lỗ khác nhau.

## 📊 **Dataset Structure**

Hệ thống sử dụng 3 dataset riêng biệt:

### **1. Ball Detection Dataset (`billiards-2/`)**
- **Class:** ball (class_id: 0)
- **Format:** YOLO format
- **Mục đích:** Nhận dạng các quả bóng billiards

### **2. Table Detection Dataset (`table detector/`)**
- **Class:** table (class_id: 1)
- **Format:** YOLO format
- **Mục đích:** Nhận dạng toàn bộ bàn billiards

### **3. Pocket Detection Dataset (`pocket detection/`)**
- **Classes:** 10 loại lỗ khác nhau (class_id: 2-11)
  - BottomLeft, BottomRight
  - IntersectionLeft, IntersectionRight
  - MediumLeft, MediumRight
  - SemicircleLeft, SemicircleRight
  - TopLeft, TopRight
- **Format:** YOLO format
- **Mục đích:** Nhận dạng các loại lỗ khác nhau trên bàn

## 🚀 **Quick Start**

### **1. Cài đặt Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Tạo Dataset Kết Hợp**
```bash
python src/train_combined_model.py --skip-dataset-creation
```

### **3. Train Model**
```bash
# Train với model size m (medium)
python src/train_combined_model.py --model-size m --epochs 100

# Train với model size l (large) - chính xác hơn nhưng chậm hơn
python src/train_combined_model.py --model-size l --epochs 100

# Train không dùng wandb
python src/train_combined_model.py --no-wandb --epochs 50
```

### **4. Test Detection**
```bash
# Test với ảnh tổng hợp
python src/demo_combined_detection.py --synthetic

# Test với ảnh từ dataset
python src/demo_combined_detection.py --test-dataset

# Test với model đã train
python src/demo_combined_detection.py --model runs/train/billiards_model/combined_m/weights/best.pt --synthetic
```

### **5. Chạy Detection trên Video**
```bash
# Chạy detection trên video
python src/video_detection.py input_video.mp4 --output output_video.mp4

# Chạy với model đã train
python src/video_detection.py input_video.mp4 --model runs/train/billiards_model/combined_m/weights/best.pt --output output_video.mp4

# Lưu pocket events
python src/video_detection.py input_video.mp4 --save-events pocket_events.json
```

## 📁 **Project Structure**

```
poolShotPredictor/
├── src/
│   ├── config/
│   │   └── config.py              # Cấu hình hệ thống
│   ├── models/
│   │   ├── ball_detector.py       # Multi-object detector
│   │   ├── pocket_detector.py     # Pocket event detector
│   │   └── ball_tracker.py        # Ball tracking
│   ├── utils/
│   │   ├── dataset_processor.py   # Dataset processing
│   │   ├── annotation_tool.py     # Manual annotation tool
│   │   └── video_utils.py         # Video utilities
│   ├── train_combined_model.py    # Training script
│   ├── demo_combined_detection.py # Demo script
│   └── video_detection.py         # Video processing
├── data/
│   └── combined_dataset/          # Combined dataset
├── models/
│   └── billiards_model.pt         # Trained model
├── pocket detection/              # Pocket dataset
├── table detector/                # Table dataset
├── billiards-2/                   # Ball dataset
└── runs/                          # Training results
```

## 🔧 **Configuration**

### **Classes và Colors**
```python
CLASSES = {
    0: "ball",           # Green
    1: "table",          # Blue
    2: "BottomLeft",     # Cyan
    3: "BottomRight",    # Cyan
    4: "IntersectionLeft", # Cyan
    5: "IntersectionRight", # Cyan
    6: "MediumLeft",     # Cyan
    7: "MediumRight",    # Cyan
    8: "SemicircleLeft", # Cyan
    9: "SemicircleRight", # Cyan
    10: "TopLeft",       # Cyan
    11: "TopRight"       # Cyan
}
```

### **Detection Parameters**
- **Confidence Threshold:** 0.3
- **IOU Threshold:** 0.3
- **Max Detections:** 30
- **Pocket Detection Radius:** 50 pixels

## 📈 **Training Options**

### **Model Sizes**
- `n` (nano): Nhỏ nhất, nhanh nhất
- `s` (small): Nhỏ, nhanh
- `m` (medium): Cân bằng (khuyến nghị)
- `l` (large): Lớn, chính xác
- `x` (xlarge): Lớn nhất, chính xác nhất

### **Training Parameters**
```bash
python src/train_combined_model.py \
    --model-size m \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640
```

## 🎯 **Features**

### **1. Multi-Class Detection**
- Phát hiện bóng, bàn và 10 loại lỗ khác nhau
- Filter theo kích thước cho từng class
- Color-coded bounding boxes

### **2. Pocket Event Detection**
- Theo dõi vị trí bóng theo thời gian
- Phát hiện khi bóng rơi vào lỗ
- Thống kê theo loại lỗ
- Lưu events vào file JSON

### **3. Video Processing**
- Xử lý video real-time
- Hiển thị FPS và statistics
- Lưu video output
- Pause/Resume controls

### **4. Dataset Management**
- Tự động kết hợp 3 dataset
- Convert labels sang format thống nhất
- Tạo YAML config tự động

## 📊 **Performance Metrics**

Sau khi train, model sẽ hiển thị:
- **mAP@0.5:** Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95:** Mean Average Precision across IoU thresholds
- **Precision:** Độ chính xác
- **Recall:** Độ bao phủ

## 🔍 **Troubleshooting**

### **1. Import Errors**
```bash
# Đảm bảo đang ở thư mục gốc
cd poolShotPredictor
python src/demo_combined_detection.py --synthetic
```

### **2. CUDA Issues**
```bash
# Kiểm tra CUDA
python src/check_cuda.py

# Train trên CPU nếu cần
python src/train_combined_model.py --device cpu
```

### **3. Memory Issues**
```bash
# Giảm batch size
python src/train_combined_model.py --batch-size 8

# Giảm image size
python src/train_combined_model.py --img-size 416
```

### **4. Display Issues**
```bash
# Chạy không hiển thị
python src/demo_combined_detection.py --synthetic --no-display
```

## 📝 **Usage Examples**

### **Training từ đầu**
```bash
# 1. Tạo dataset kết hợp
python src/train_combined_model.py --skip-dataset-creation

# 2. Train model
python src/train_combined_model.py --model-size m --epochs 100

# 3. Test model
python src/demo_combined_detection.py --model runs/train/billiards_model/combined_m/weights/best.pt --synthetic
```

### **Chạy trên video thực tế**
```bash
# 1. Chạy detection
python src/video_detection.py khoai.mkv --output khoai_detected.mp4

# 2. Lưu pocket events
python src/video_detection.py khoai.mkv --save-events pocket_events.json --no-display
```

### **Test với ảnh dataset**
```bash
# Test với ảnh từ pocket dataset
python src/demo_combined_detection.py --image "pocket detection/test/images/example.jpg"

# Test với ảnh từ table dataset
python src/demo_combined_detection.py --image "table detector/test/images/example.jpg"
```

## 🎯 **Next Steps**

1. **Train model** với dataset kết hợp
2. **Test detection** trên ảnh và video
3. **Fine-tune parameters** nếu cần
4. **Deploy** cho ứng dụng thực tế

## 📞 **Support**

Nếu gặp vấn đề, hãy kiểm tra:
1. Dataset structure đúng format
2. Dependencies đã cài đầy đủ
3. CUDA/GPU compatibility
4. Memory requirements

---

 