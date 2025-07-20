#!/usr/bin/env python3
"""
Test all YOLOv11 models: Ball Detector, Table Detector, and Pocket Detector
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import os

def load_models():
    """Load all available models"""
    models = {}
    
    # Ball detector
    ball_model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    if Path(ball_model_path).exists():
        try:
            models['ball'] = YOLO(ball_model_path)
            print(f"✅ Ball Detector loaded: {ball_model_path}")
        except Exception as e:
            print(f"❌ Failed to load Ball Detector: {e}")
    
    # Table detector
    table_model_paths = [
        "table_detector/yolo11_table_detector/weights/best.pt",
        "table_detector/yolo11_table_detector_continued/weights/best.pt"
    ]
    for path in table_model_paths:
        if Path(path).exists():
            try:
                models['table'] = YOLO(path)
                print(f"✅ Table Detector loaded: {path}")
                break
            except Exception as e:
                print(f"❌ Failed to load Table Detector: {e}")
    
    # Pocket detector
    pocket_model_paths = [
        "pocket_detector/yolo11_pocket_detector/weights/best.pt",
        "pocket_detector/yolo11_pocket_detector_continued/weights/best.pt"
    ]
    for path in pocket_model_paths:
        if Path(path).exists():
            try:
                models['pocket'] = YOLO(path)
                print(f"✅ Pocket Detector loaded: {path}")
                break
            except Exception as e:
                print(f"❌ Failed to load Pocket Detector: {e}")
    
    return models

def test_image_with_all_models(image_path, models, confidence=0.3):
    """Test image with all available models"""
    print(f"\n🎯 Testing image: {image_path}")
    print("=" * 60)
    
    # Load image
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create subplots
    num_models = len(models)
    if num_models == 0:
        print("❌ No models available for testing")
        return
    
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Test each model
    for i, (model_name, model) in enumerate(models.items(), 1):
        try:
            # Run detection
            results = model(image, conf=confidence, verbose=False)
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    detections = len(boxes)
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy()
                    
                    print(f"🎯 {model_name.upper()} DETECTOR:")
                    print(f"   Detections: {detections}")
                    print(f"   Confidences: {confidences}")
                    print(f"   Classes: {class_ids.astype(int)}")
                    
                    # Display result
                    result_image = results[0].plot()
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(result_image_rgb)
                    axes[i].set_title(f"{model_name.title()} Detector\n{detections} detections")
                else:
                    print(f"🎯 {model_name.upper()} DETECTOR: No detections")
                    axes[i].imshow(image_rgb)
                    axes[i].set_title(f"{model_name.title()} Detector\nNo detections")
            else:
                print(f"🎯 {model_name.upper()} DETECTOR: No results")
                axes[i].imshow(image_rgb)
                axes[i].set_title(f"{model_name.title()} Detector\nNo results")
            
            axes[i].axis('off')
            
        except Exception as e:
            print(f"❌ Error testing {model_name} detector: {e}")
            axes[i].imshow(image_rgb)
            axes[i].set_title(f"{model_name.title()} Detector\nError")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save result
    output_path = f"test_all_models_{Path(image_path).stem}.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Result saved: {output_path}")

def find_test_images():
    """Find available test images"""
    test_images = []
    
    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    # Search in current directory
    for ext in extensions:
        test_images.extend(Path('.').glob(ext))
    
    # Search in specific directories
    search_dirs = ['test', 'valid', 'data']
    for search_dir in search_dirs:
        if Path(search_dir).exists():
            for ext in extensions:
                test_images.extend(Path(search_dir).rglob(ext))
    
    return test_images

def main():
    """Main function"""
    print("🎱 YOLOv11 Multi-Model Testing System")
    print("=" * 50)
    
    # Load models
    models = load_models()
    
    if not models:
        print("❌ No models available for testing!")
        print("Please train models first using:")
        print("  python train_all_models.py")
        return
    
    print(f"\n✅ Loaded {len(models)} model(s): {list(models.keys())}")
    
    # Find test images
    test_images = find_test_images()
    
    if not test_images:
        print("❌ No test images found!")
        print("Please add some images to test with.")
        return
    
    print(f"\n📸 Found {len(test_images)} test image(s):")
    for i, img_path in enumerate(test_images[:10], 1):  # Show first 10
        print(f"  {i}. {img_path}")
    
    if len(test_images) > 10:
        print(f"  ... and {len(test_images) - 10} more")
    
    # Test with confidence levels
    confidence_levels = [0.1, 0.3, 0.5]
    
    for confidence in confidence_levels:
        print(f"\n🎯 Testing with confidence: {confidence}")
        print("=" * 40)
        
        # Test first few images
        for img_path in test_images[:3]:  # Test first 3 images
            test_image_with_all_models(str(img_path), models, confidence)
            
            # Ask user if they want to continue
            response = input(f"\nContinue testing with confidence {confidence}? (y/n): ").strip().lower()
            if response != 'y':
                break

if __name__ == "__main__":
    main() 