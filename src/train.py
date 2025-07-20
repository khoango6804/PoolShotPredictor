from ultralytics import YOLO
import os
from pathlib import Path
import torch

def train_model():
    # Kiểm tra GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Khởi tạo model YOLOv8l (large version)
    model = YOLO('yolov8l.pt')
    
    # Đường dẫn đến file data.yaml
    data_yaml = str(Path('billiards-2/data.yaml').absolute())
    
    # Các tham số training
    training_args = {
        'data': data_yaml,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'patience': 50,
        'device': '0',
        'workers': 8,
        'project': 'runs/train',
        'name': 'billiards_model',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'amp': True,
        'cache': 'disk',
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'single_cls': False,
        'multi_scale': False,
        'rect': False,
        'nbs': 32,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # Bắt đầu training
    print("Starting training...")
    results = model.train(**training_args)
    
    # Lưu model
    model.save('models/billiards_model.pt')
    print("Training completed. Model saved to models/billiards_model.pt")
    
    # Ghi log ra file log.txt
    if results and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        with open('log.txt', 'w', encoding='utf-8') as f:
            f.write("Training Results\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print("Training metrics have been saved to log.txt")
    return results

if __name__ == "__main__":
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Train model
    results = train_model() 