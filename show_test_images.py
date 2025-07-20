#!/usr/bin/env python3
"""
Show original and test images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_images_comparison():
    """Show original and test images side by side"""
    
    print("🖼️ Image Comparison")
    print("=" * 40)
    
    # Image paths
    original_path = "demo_result.jpg"
    test_path = "low_conf_test_demo_result.jpg"
    
    # Check if images exist
    if not Path(original_path).exists():
        print(f"❌ Original image not found: {original_path}")
        return
    
    if not Path(test_path).exists():
        print(f"❌ Test image not found: {test_path}")
        return
    
    # Load images
    original = cv2.imread(original_path)
    test = cv2.imread(test_path)
    
    if original is None:
        print(f"❌ Cannot load original image: {original_path}")
        return
    
    if test is None:
        print(f"❌ Cannot load test image: {test_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    
    # Get image info
    orig_height, orig_width = original.shape[:2]
    test_height, test_width = test.shape[:2]
    
    print(f"📐 Original image: {orig_width}x{orig_height}")
    print(f"📐 Test image: {test_width}x{test_height}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    axes[0].imshow(original_rgb)
    axes[0].set_title('🖼️ Original Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Test image with detections
    axes[1].imshow(test_rgb)
    axes[1].set_title('🎯 YOLOv11 Detection Result (conf=0.01)', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Comparison saved: image_comparison.png")
    
    # Show individual images
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(test_rgb)
    plt.title('Detection Result', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('side_by_side.png', dpi=300, bbox_inches='tight')
    print(f"💾 Side-by-side saved: side_by_side.png")
    
    # Show image details
    print(f"\n📊 Image Details:")
    print(f"  🖼️ Original: {original_path} ({orig_width}x{orig_height})")
    print(f"  🎯 Test: {test_path} ({test_width}x{test_height})")
    print(f"  📈 Size increase: {test.shape[0]*test.shape[1]/(orig_height*orig_width):.1f}x")
    
    return {
        'original_size': (orig_width, orig_height),
        'test_size': (test_width, test_height),
        'original_path': original_path,
        'test_path': test_path
    }

def show_all_test_images():
    """Show all test images"""
    
    print("\n🖼️ All Test Images")
    print("=" * 40)
    
    test_images = [
        "demo_result.jpg",
        "low_conf_test_demo_result.jpg", 
        "test_specific_1_demo_result.jpg"
    ]
    
    existing_images = []
    for img_path in test_images:
        if Path(img_path).exists():
            existing_images.append(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                size_kb = Path(img_path).stat().st_size / 1024
                print(f"  📸 {img_path}: {width}x{height}, {size_kb:.1f}KB")
    
    if len(existing_images) >= 2:
        # Create grid of images
        fig, axes = plt.subplots(1, len(existing_images), figsize=(6*len(existing_images), 6))
        
        if len(existing_images) == 1:
            axes = [axes]
        
        for i, img_path in enumerate(existing_images):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(f'{img_path}', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('all_test_images.png', dpi=300, bbox_inches='tight')
        print(f"💾 All images saved: all_test_images.png")

def main():
    print("🖼️ Image Display and Comparison")
    print("=" * 50)
    
    # Show comparison
    result = show_images_comparison()
    
    # Show all test images
    show_all_test_images()
    
    print(f"\n🎉 Image display completed!")
    print(f"📁 Check the generated PNG files for image comparisons")

if __name__ == "__main__":
    main() 