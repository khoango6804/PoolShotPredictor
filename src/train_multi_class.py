import os
import yaml
from pathlib import Path
import argparse
from ultralytics import YOLO
import wandb

def create_dataset_yaml(dataset_path, output_path):
    """
    Create YAML configuration file for YOLO training
    
    Args:
        dataset_path: Path to processed dataset
        output_path: Path to save YAML file
    """
    yaml_content = f"""# YOLO dataset configuration for billiards multi-class detection
path: {Path(dataset_path).absolute()}  # dataset root dir
train: training/images  # train images (relative to 'path')
val: validation/images  # val images (relative to 'path')
test: testing/images  # test images (relative to 'path')

# Classes
nc: 4  # number of classes
names: ['ball', 'table_edge', 'cue_stick', 'pocket']  # class names
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created: {output_path}")
    return output_path

def train_model(dataset_yaml, model_size='m', epochs=100, batch_size=16, imgsz=640):
    """
    Train YOLO model for multi-class detection
    
    Args:
        dataset_yaml: Path to dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        imgsz: Input image size
    """
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project="billiards-multi-class-detection",
        config={
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "classes": ['ball', 'table_edge', 'cue_stick', 'pocket']
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
        device='auto',
        workers=8,
        project='billiards_model',
        name=f'multi_class_{model_size}',
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
    parser = argparse.ArgumentParser(description='Train Multi-Class YOLO Model for Billiards')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                       help='Path to processed dataset directory')
    parser.add_argument('--model-size', '-m', type=str, default='m', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', '-i', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Create dataset YAML
    dataset_path = Path(args.dataset)
    yaml_path = dataset_path / 'dataset.yaml'
    
    if not yaml_path.exists():
        create_dataset_yaml(dataset_path, yaml_path)
    
    # Verify dataset structure
    required_dirs = ['training/images', 'training/labels', 
                    'validation/images', 'validation/labels']
    
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            print(f"Error: Required directory not found: {full_path}")
            return
    
    # Check if dataset has images
    train_images = list((dataset_path / 'training/images').glob('*.jpg'))
    val_images = list((dataset_path / 'validation/images').glob('*.jpg'))
    
    if not train_images:
        print("Error: No training images found!")
        return
    
    if not val_images:
        print("Error: No validation images found!")
        return
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Train model
    try:
        model = train_model(
            dataset_yaml=yaml_path,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.img_size
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Best model: {model.ckpt_path}")
        print(f"Model can be used with: model = YOLO('{model.ckpt_path}')")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return

if __name__ == "__main__":
    main() 