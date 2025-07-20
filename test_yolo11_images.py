#!/usr/bin/env python3
"""
Test YOLOv11 model on images
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import glob
import os

def test_on_images(model_path, image_folder=".", conf=0.3, max_images=10):
    """Test YOLOv11 model on images"""
    
    print(f"🖼️ Testing YOLOv11 on images...")
    print(f"Model: {model_path}")
    print(f"Confidence: {conf}")
    print("=" * 50)
    
    # Load model
    model = YOLO(model_path)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    # Filter out detection result images
    image_files = [f for f in image_files if not any(x in f.lower() for x in ['detection', 'result', 'output'])]
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    # Limit number of images to test
    image_files = image_files[:max_images]
    
    total_detections = 0
    processed_images = 0
    
    for i, image_path in enumerate(image_files):
        print(f"\n🖼️ Testing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ❌ Cannot load image: {image_path}")
            continue
        
        # Get image info
        height, width = image.shape[:2]
        print(f"  📐 Size: {width}x{height}")
        
        # Run detection
        results = model(image, conf=float(conf), verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                print(f"  🎯 Detections: {detections}")
                print(f"  📊 Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
                print(f"  🏷️ Classes detected: {np.unique(class_ids.astype(int))}")
                
                # Show top detections
                if detections > 0:
                    top_indices = np.argsort(confidences)[-5:]  # Top 5
                    print(f"  🏆 Top detections:")
                    for idx in reversed(top_indices):
                        class_id = int(class_ids[idx])
                        conf = confidences[idx]
                        print(f"    Class {class_id}: {conf:.3f}")
                
                total_detections += detections
            else:
                print(f"  ❌ No detections")
        else:
            print(f"  ❌ No results")
        
        processed_images += 1
        
        # Save result image
        if results and len(results) > 0:
            output_path = f"test_result_{i+1}_{os.path.basename(image_path)}"
            result_image = results[0].plot()
            cv2.imwrite(output_path, result_image)
            print(f"  💾 Result saved: {output_path}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  🖼️ Images processed: {processed_images}")
    print(f"  🎯 Total detections: {total_detections}")
    print(f"  📈 Average detections per image: {total_detections/processed_images:.1f}")
    
    return {
        'processed_images': processed_images,
        'total_detections': total_detections,
        'avg_detections': total_detections/processed_images if processed_images > 0 else 0
    }

def test_specific_images(model_path, image_paths, conf=0.3):
    """Test on specific images"""
    
    print(f"🎯 Testing specific images...")
    print(f"Model: {model_path}")
    print(f"Confidence: {conf}")
    print("=" * 50)
    
    # Load model
    model = YOLO(model_path)
    
    total_detections = 0
    processed_images = 0
    
    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n🖼️ Testing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ❌ Cannot load image: {image_path}")
            continue
        
        # Get image info
        height, width = image.shape[:2]
        print(f"  📐 Size: {width}x{height}")
        
        # Run detection
        results = model(image, conf=float(conf), verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                detections = len(boxes)
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                
                print(f"  🎯 Detections: {detections}")
                print(f"  📊 Confidence range: {confidences.min():.3f} - {confidences.max():.3f}")
                print(f"  🏷️ Classes detected: {np.unique(class_ids.astype(int))}")
                
                # Show all detections
                if detections > 0:
                    print(f"  📋 All detections:")
                    for idx in range(min(detections, 10)):  # Show first 10
                        class_id = int(class_ids[idx])
                        conf = confidences[idx]
                        print(f"    Class {class_id}: {conf:.3f}")
                    
                    if detections > 10:
                        print(f"    ... and {detections - 10} more")
                
                total_detections += detections
            else:
                print(f"  ❌ No detections")
        else:
            print(f"  ❌ No results")
        
        processed_images += 1
        
        # Save result image
        if results and len(results) > 0:
            output_path = f"test_specific_{i+1}_{os.path.basename(image_path)}"
            result_image = results[0].plot()
            cv2.imwrite(output_path, result_image)
            print(f"  💾 Result saved: {output_path}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  🖼️ Images processed: {processed_images}")
    print(f"  🎯 Total detections: {total_detections}")
    print(f"  📈 Average detections per image: {total_detections/processed_images:.1f}")
    
    return {
        'processed_images': processed_images,
        'total_detections': total_detections,
        'avg_detections': total_detections/processed_images if processed_images > 0 else 0
    }

def main():
    print("🖼️ YOLOv11 Image Testing")
    print("=" * 40)
    
    model_path = "runs/detect/yolo11_billiards_gpu/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Test on available images
    print("1️⃣ Testing on available images...")
    test_on_images(model_path, conf=0.3, max_images=5)
    
    # Test on specific images if they exist
    specific_images = [
        "check_labels_sample_1.jpg",
        "check_labels_sample_2.jpg", 
        "check_labels_sample_3.jpg",
        "check_labels_sample_4.jpg",
        "check_labels_sample_5.jpg",
        "demo_result.jpg"
    ]
    
    existing_images = [img for img in specific_images if os.path.exists(img)]
    
    if existing_images:
        print(f"\n2️⃣ Testing on specific images...")
        test_specific_images(model_path, existing_images, conf=0.3)
    
    print(f"\n🎉 Image testing completed!")

if __name__ == "__main__":
    main() 