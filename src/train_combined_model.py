import os
import yaml
from pathlib import Path
import argparse
from ultralytics import YOLO
import wandb
import shutil

def create_combined_dataset():
    """
    Create a combined dataset from separate datasets
    """
    print("Creating combined dataset...")
    
    # Create combined dataset directory
    combined_dir = Path("data/combined_dataset")
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for split in ['train', 'valid', 'test']:
        (combined_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (combined_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy ball dataset (billiards-2)
    print("Processing ball dataset...")
    ball_dataset = Path("billiards-2")
    if ball_dataset.exists():
        for split in ['train', 'valid', 'test']:
            if (ball_dataset / split).exists():
                # Copy images
                src_images = ball_dataset / split / 'images'
                dst_images = combined_dir / split / 'images'
                if src_images.exists():
                    for img_file in src_images.glob('*.jpg'):
                        shutil.copy2(img_file, dst_images)
                
                # Copy and convert labels (ball class = 0)
                src_labels = ball_dataset / split / 'labels'
                dst_labels = combined_dir / split / 'labels'
                if src_labels.exists():
                    for label_file in src_labels.glob('*.txt'):
                        # Convert ball labels to class 0
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Convert to class 0 (ball)
                                new_line = f"0 {' '.join(parts[1:])}\n"
                                new_lines.append(new_line)
                        
                        # Save converted label
                        new_label_file = dst_labels / label_file.name
                        with open(new_label_file, 'w') as f:
                            f.writelines(new_lines)
    
    # Copy table dataset
    print("Processing table dataset...")
    table_dataset = Path("table detector")
    if table_dataset.exists():
        for split in ['train', 'valid', 'test']:
            if (table_dataset / split).exists():
                # Copy images
                src_images = table_dataset / split / 'images'
                dst_images = combined_dir / split / 'images'
                if src_images.exists():
                    for img_file in src_images.glob('*.jpg'):
                        shutil.copy2(img_file, dst_images)
                
                # Copy and convert labels (table class = 1)
                src_labels = table_dataset / split / 'labels'
                dst_labels = combined_dir / split / 'labels'
                if src_labels.exists():
                    for label_file in src_labels.glob('*.txt'):
                        # Convert table labels to class 1
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Convert to class 1 (table)
                                new_line = f"1 {' '.join(parts[1:])}\n"
                                new_lines.append(new_line)
                        
                        # Save converted label
                        new_label_file = dst_labels / label_file.name
                        with open(new_label_file, 'w') as f:
                            f.writelines(new_lines)
    
    # Copy pocket dataset
    print("Processing pocket dataset...")
    pocket_dataset = Path("pocket detection")
    if pocket_dataset.exists():
        for split in ['train', 'valid', 'test']:
            if (pocket_dataset / split).exists():
                # Copy images
                src_images = pocket_dataset / split / 'images'
                dst_images = combined_dir / split / 'images'
                if src_images.exists():
                    for img_file in src_images.glob('*.jpg'):
                        shutil.copy2(img_file, dst_images)
                
                # Copy and convert labels (pocket classes = 2-11)
                src_labels = pocket_dataset / split / 'labels'
                dst_labels = combined_dir / split / 'labels'
                if src_labels.exists():
                    for label_file in src_labels.glob('*.txt'):
                        # Convert pocket labels (add 2 to class_id)
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Convert pocket classes (add 2 to shift from 0-9 to 2-11)
                                old_class = int(parts[0])
                                new_class = old_class + 2  # Shift to 2-11
                                new_line = f"{new_class} {' '.join(parts[1:])}\n"
                                new_lines.append(new_line)
                        
                        # Save converted label
                        new_label_file = dst_labels / label_file.name
                        with open(new_label_file, 'w') as f:
                            f.writelines(new_lines)
    
    print(f"Combined dataset created at: {combined_dir}")
    return combined_dir

def create_combined_yaml(dataset_path):
    """
    Create YAML configuration file for combined dataset
    """
    yaml_content = f"""# YOLO dataset configuration for combined billiards detection
path: {Path(dataset_path).absolute()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: 12  # number of classes
names: ['ball', 'table', 'BottomLeft', 'BottomRight', 'IntersectionLeft', 'IntersectionRight', 'MediumLeft', 'MediumRight', 'SemicircleLeft', 'SemicircleRight', 'TopLeft', 'TopRight']  # class names
"""
    
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Combined dataset YAML created: {yaml_path}")
    return yaml_path

def train_combined_model(dataset_yaml, model_size='m', epochs=100, batch_size=16, imgsz=640, device='auto'):
    """
    Train YOLO model for combined detection
    
    Args:
        dataset_yaml: Path to dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
        device: Device to use for training (e.g., 0, cpu, auto)
    """
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project="billiards-combined-detection",
        config={
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "classes": ['ball', 'table', 'BottomLeft', 'BottomRight', 'IntersectionLeft', 'IntersectionRight', 'MediumLeft', 'MediumRight', 'SemicircleLeft', 'SemicircleRight', 'TopLeft', 'TopRight']
        }
    )
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=20,
        save=True,
        save_period=10,
        device=device,
        workers=8,
        project='billiards_model',
        name=f'combined_{model_size}',
        exist_ok=True
    )
    
    # Validate the model
    metrics = model.val()
    
    # Log final metrics
    wandb.log({
        "final_map50": metrics.box.map50,
        "final_map50_95": metrics.box.map,
        "final_precision": metrics.box.mp,
        "final_recall": metrics.box.mr
    })
    
    wandb.finish()
    
    print("Training completed!")
    print(f"Best model saved to: {model.ckpt_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Combined YOLO Model for Billiards')
    parser.add_argument('--model-size', '-m', type=str, default='m', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', '-i', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (e.g., 0, cpu, auto)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--skip-dataset-creation', action='store_true',
                       help='Skip dataset creation if already exists')
    
    args = parser.parse_args()
    
    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Create combined dataset
    if not args.skip_dataset_creation:
        dataset_path = create_combined_dataset()
    else:
        dataset_path = Path("data/combined_dataset")
    
    # Create dataset YAML
    yaml_path = dataset_path / 'dataset.yaml'
    if not yaml_path.exists():
        create_combined_yaml(dataset_path)
    
    # Verify dataset structure
    required_dirs = ['train/images', 'train/labels', 
                    'valid/images', 'valid/labels']
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"Error: Required directory not found: {full_path}")
            return
    
    # Check if dataset has images
    train_images = list((dataset_path / 'train/images').glob('*.jpg'))
    val_images = list((dataset_path / 'valid/images').glob('*.jpg'))
    
    if not train_images:
        print("Error: No training images found!")
        return
    
    if not val_images:
        print("Error: No validation images found!")
        return
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Train model
    try:
        model = train_combined_model(
            dataset_yaml=yaml_path,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            device=args.device
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Best model: {model.ckpt_path}")
        print(f"Model can be used with: model = YOLO('{model.ckpt_path}')")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main() 