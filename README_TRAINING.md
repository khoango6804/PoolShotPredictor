# 🎱 YOLOv11 Multi-Model Training Guide

## 📋 Tổng quan

Hệ thống training cho 3 loại model YOLOv11:
1. **Ball Detector** - Phát hiện các loại bóng bi-a (23 classes)
2. **Table Detector** - Phát hiện bàn bi-a (1 class)
3. **Pocket Detector** - Phát hiện các loại túi (10 classes)

## 🚀 Cách sử dụng

### 1. Kiểm tra status hiện tại

```bash
python check_model_status.py
```

### 2. Training từng model

#### **Table Detector:**
```bash
python train_table_detector.py
```

#### **Pocket Detector:**
```bash
python train_pocket_detector.py
```

### 3. Training tất cả (Interactive)

```bash
python train_all_models.py
```

### 4. Monitor training progress

```bash
python monitor_training.py
```

## 📊 Dataset Information

### **Ball Dataset**
- **Path**: `data/combined_dataset/dataset.yaml`
- **Classes**: 23 different ball types
- **Status**: ✅ Available (model trained)

### **Table Dataset**
- **Path**: `table detector/data.yaml`
- **Classes**: 1 (table)
- **Status**: ✅ Available (needs training)

### **Pocket Dataset**
- **Path**: `pocket detection/data.yaml`
- **Classes**: 10 pocket types
  - BottomLeft, BottomRight
  - IntersectionLeft, IntersectionRight
  - MediumLeft, MediumRight
  - SemicircleLeft, SemicircleRight
  - TopLeft, TopRight
- **Status**: ✅ Available (needs training)

## ⚙️ Training Configuration

### **Default Settings:**
- **Model**: YOLOv11n (nano)
- **Epochs**: 100 (fresh), 50 (continued)
- **Image Size**: 640x640
- **Batch Size**: 16
- **Device**: GPU (CUDA)
- **Patience**: 20 epochs
- **Save Period**: 10 epochs

### **Checkpoint Training:**
- Tự động tìm checkpoint đã có
- Tiếp tục training từ checkpoint
- Nếu không có checkpoint → train từ đầu

## 📁 Output Structure

```
project/
├── table_detector/
│   ├── yolo11_table_detector/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   ├── last.pt
│   │   │   └── epoch*.pt
│   │   ├── results.csv
│   │   └── args.yaml
│   └── yolo11_table_detector_continued/
│       └── ...
├── pocket_detector/
│   ├── yolo11_pocket_detector/
│   │   ├── weights/
│   │   │   ├── best.pt
│   │   │   ├── last.pt
│   │   │   └── epoch*.pt
│   │   ├── results.csv
│   │   └── args.yaml
│   └── yolo11_pocket_detector_continued/
│       └── ...
└── runs/detect/yolo11_billiards_gpu/
    └── weights/best.pt (Ball Detector)
```

## 🧪 Testing Models

### **Test tất cả models:**
```bash
python test_all_models.py
```

### **Test từng model riêng:**
```bash
# Ball detector
python test_yolo11_images.py

# Table detector
python test_table_detector.py

# Pocket detector  
python test_pocket_detector.py
```

## 🔧 Troubleshooting

### **Lỗi thường gặp:**

1. **CUDA out of memory**
   ```bash
   # Giảm batch size
   batch_size = 8  # thay vì 16
   ```

2. **Dataset not found**
   ```bash
   # Kiểm tra đường dẫn dataset
   python check_model_status.py
   ```

3. **Training stuck**
   ```bash
   # Kiểm tra progress
   python monitor_training.py
   ```

### **Performance Tips:**

1. **GPU Memory**: Đóng các ứng dụng khác khi training
2. **Batch Size**: Điều chỉnh theo GPU memory
3. **Image Size**: Giảm xuống 416 nếu cần
4. **Epochs**: Bắt đầu với 50 epochs

## 📈 Training Progress

### **Monitoring:**
- **Real-time**: Check `results.csv` trong output directory
- **Visualization**: TensorBoard logs
- **Checkpoints**: Saved every 10 epochs

### **Metrics:**
- **mAP**: Mean Average Precision
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## 🎯 Expected Results

### **Table Detector:**
- **mAP50**: > 0.90
- **Precision**: > 0.85
- **Recall**: > 0.80

### **Pocket Detector:**
- **mAP50**: > 0.85
- **Precision**: > 0.80
- **Recall**: > 0.75

## 🚀 Quick Start

```bash
# 1. Check current status
python check_model_status.py

# 2. Train missing models
python train_all_models.py

# 3. Monitor progress
python monitor_training.py

# 4. Test all models
python test_all_models.py
```

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra CUDA và GPU drivers
2. Đảm bảo đủ disk space (> 5GB)
3. Kiểm tra dataset paths
4. Monitor GPU memory usage

---

**🎱 YOLOv11 Multi-Model Training System**  
*Powered by Ultralytics YOLOv11 | GPU Optimized* 