# Hệ thống Nhận dạng Đa lớp cho Billiards

Hệ thống này mở rộng khả năng nhận dạng từ chỉ bi (ball) lên 4 lớp đối tượng:
- **Ball** (Bi): Nhận dạng các quả bi trên bàn
- **Table Edge** (Cạnh bàn): Nhận dạng các cạnh của bàn billiards
- **Cue Stick** (Gậy): Nhận dạng gậy đánh bi
- **Pocket** (Lỗ): Nhận dạng các lỗ trên bàn và phát hiện sự kiện bi rơi lỗ

## Cấu trúc Dự án

```
src/
├── config/
│   └── config.py              # Cấu hình cho đa lớp detection
├── models/
│   ├── ball_detector.py       # MultiObjectDetector (đã cập nhật)
│   ├── pocket_detector.py     # Phát hiện sự kiện bi rơi lỗ
│   └── ball_tracker.py        # Tracking bi
├── utils/
│   ├── dataset_processor.py   # Xử lý dataset đa lớp
│   ├── create_multi_class_dataset.py  # Tạo dataset mẫu
│   └── annotation_tool.py     # Tool annotation thủ công
├── multi_object_detection.py  # Script chính chạy detection
└── train_multi_class.py       # Train model đa lớp
```

## Cài đặt và Sử dụng

### 1. Chuẩn bị Dataset

#### Tạo Dataset Tổng hợp (Synthetic)
```bash
# Tạo 100 ảnh tổng hợp với annotation đầy đủ
python src/utils/create_multi_class_dataset.py --output data/multi_class_dataset --synthetic 100
```

#### Chuyển đổi Dataset Hiện có
```bash
# Chuyển đổi dataset chỉ có ball thành đa lớp
python src/utils/create_multi_class_dataset.py --output data/multi_class_dataset --convert billiards-2
```

#### Annotation Thủ công
```bash
# Sử dụng tool annotation
python src/utils/annotation_tool.py --images data/images --labels data/labels
```

**Hướng dẫn sử dụng Annotation Tool:**
- **1-4**: Chuyển đổi class (1=ball, 2=table_edge, 3=cue_stick, 4=pocket)
- **Mouse**: Vẽ bounding box
- **S**: Lưu annotation
- **N**: Ảnh tiếp theo
- **P**: Ảnh trước
- **D**: Xóa annotation cuối
- **C**: Xóa tất cả annotation
- **Q**: Thoát

### 2. Xử lý Dataset

```bash
# Chia dataset thành train/val/test
python -c "
from src.utils.dataset_processor import DatasetProcessor
processor = DatasetProcessor('data/multi_class_dataset', 'data/processed_dataset')
processor.process_dataset(split_ratio=(0.7, 0.2, 0.1))
processor.create_yaml_config()
"
```

### 3. Train Model

```bash
# Train model đa lớp
python src/train_multi_class.py --dataset data/processed_dataset --epochs 100 --batch-size 16
```

**Các tùy chọn:**
- `--model-size`: n, s, m, l, x (kích thước model)
- `--epochs`: Số epoch training
- `--batch-size`: Batch size
- `--img-size`: Kích thước ảnh input
- `--no-wandb`: Tắt logging wandb

### 4. Chạy Detection

#### Xử lý Video
```bash
# Xử lý video file
python src/multi_object_detection.py --input video.mp4 --output output.mp4
```

#### Xử lý Camera
```bash
# Xử lý camera live
python src/multi_object_detection.py --input 0
```

#### Sử dụng Model Custom
```bash
# Sử dụng model đã train
python src/multi_object_detection.py --input video.mp4 --model runs/train/billiards_model/multi_class_m/weights/best.pt
```

## Tính năng Chính

### 1. Multi-Object Detection
- Nhận dạng 4 lớp đối tượng với độ chính xác cao
- Filter overlapping detections
- Size validation cho từng class

### 2. Pocket Event Detection
- Phát hiện sự kiện bi rơi lỗ
- Tracking bi để xác định pocketing
- Thống kê pocket rate và history

### 3. Ball Tracking
- Tracking bi qua các frame
- Vẽ trajectory của bi
- Phân tích chuyển động

### 4. Real-time Statistics
- FPS counter
- Pocket statistics
- Detection counts
- Performance metrics

## Cấu hình

### Class Definitions (config.py)
```python
CLASSES = {
    0: "ball",
    1: "table_edge", 
    2: "cue_stick",
    3: "pocket"
}

CLASS_COLORS = {
    0: (0, 255, 0),    # Green for balls
    1: (255, 0, 0),    # Blue for table edges
    2: (0, 0, 255),    # Red for cue sticks
    3: (255, 255, 0)   # Cyan for pockets
}
```

### Detection Parameters
```python
# Size constraints for each class
MIN_BALL_SIZE = 20
MAX_BALL_SIZE = 100
MIN_CUE_SIZE = 50
MAX_CUE_SIZE = 300
MIN_POCKET_SIZE = 30
MAX_POCKET_SIZE = 80

# Pocket detection
POCKET_DETECTION_RADIUS = 50
POCKET_CONFIDENCE_THRESHOLD = 0.5
```

## Output Format

### Detection Results (JSON)
```json
{
  "video_info": {
    "total_frames": 1000,
    "processing_time": 45.2,
    "fps": 22.1
  },
  "detection_history": [...],
  "pocket_statistics": {
    "total_pockets": 15,
    "recent_pockets": 3,
    "pocket_rate": 0.05
  },
  "classes_detected": ["ball", "table_edge", "cue_stick", "pocket"]
}
```

### Pocket Events
```json
{
  "ball_position": [500, 300],
  "pocket_position": [100, 100],
  "ball_confidence": 0.95,
  "pocket_confidence": 0.88,
  "distance": 25.5,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Tips và Best Practices

### 1. Dataset Preparation
- Thu thập ảnh từ nhiều góc độ khác nhau
- Đảm bảo đủ samples cho mỗi class
- Annotation chính xác và nhất quán

### 2. Training
- Bắt đầu với model size nhỏ (n, s) để test
- Sử dụng data augmentation
- Monitor validation metrics

### 3. Detection
- Điều chỉnh confidence threshold theo nhu cầu
- Sử dụng GPU để tăng tốc
- Monitor pocket detection radius

### 4. Performance Optimization
- Giảm input image size nếu cần tốc độ
- Sử dụng model quantization
- Batch processing cho video

## Troubleshooting

### Common Issues

1. **Low Detection Accuracy**
   - Kiểm tra dataset quality
   - Tăng số lượng training data
   - Điều chỉnh model size

2. **False Pocket Detections**
   - Giảm POCKET_DETECTION_RADIUS
   - Tăng POCKET_CONFIDENCE_THRESHOLD
   - Cải thiện pocket annotations

3. **Slow Performance**
   - Sử dụng model size nhỏ hơn
   - Giảm input resolution
   - Enable GPU acceleration

4. **Memory Issues**
   - Giảm batch size
   - Sử dụng model size nhỏ
   - Process video in chunks

## Next Steps

1. **Collect Real Data**: Thu thập ảnh thực tế từ bàn billiards
2. **Manual Annotation**: Sử dụng annotation tool để label dữ liệu
3. **Train Custom Model**: Train model trên dataset thực tế
4. **Deploy**: Triển khai hệ thống trong môi trường thực tế
5. **Optimize**: Tối ưu hóa performance và accuracy

## Support

Nếu gặp vấn đề, hãy kiểm tra:
- Log files trong thư mục runs/
- Dataset format và structure
- Model compatibility
- System requirements (CUDA, OpenCV, etc.) 