#!/usr/bin/env python3
"""
Create test image from description
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_billiards_test_image():
    """Create a test image based on the description"""
    
    # Create a blue pool table background
    width, height = 1280, 720
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue pool table color
    blue_color = (139, 69, 19)  # Dark blue-green
    image[:] = blue_color
    
    # Create a simple billiards setup
    # This is a simplified version based on the description
    
    # Draw 15 balls in triangle formation
    ball_radius = 20
    center_x, center_y = width // 2, height // 2
    
    # Triangle formation positions (simplified)
    positions = []
    
    # Top row (5 balls)
    for i in range(5):
        x = center_x - 80 + i * 40
        y = center_y - 60
        positions.append((x, y))
    
    # Second row (4 balls)
    for i in range(4):
        x = center_x - 60 + i * 40
        y = center_y - 20
        positions.append((x, y))
    
    # Third row (3 balls)
    for i in range(3):
        x = center_x - 40 + i * 40
        y = center_y + 20
        positions.append((x, y))
    
    # Fourth row (2 balls)
    for i in range(2):
        x = center_x - 20 + i * 40
        y = center_y + 60
        positions.append((x, y))
    
    # Fifth row (1 ball)
    positions.append((center_x, center_y + 100))
    
    # Ball colors (simplified)
    ball_colors = [
        (255, 255, 0),   # Yellow (1)
        (0, 0, 255),     # Blue (2)
        (255, 0, 0),     # Red (3)
        (255, 192, 203), # Pink (4)
        (128, 0, 128),   # Purple (5)
        (0, 255, 0),     # Green (6)
        (139, 69, 19),   # Brown (7)
        (0, 0, 0),       # Black (8)
        (255, 255, 0),   # Yellow (9)
        (0, 0, 255),     # Blue (10)
        (255, 0, 0),     # Red (11)
        (255, 192, 203), # Pink (12)
        (128, 0, 128),   # Purple (13)
        (0, 255, 0),     # Green (14)
        (139, 69, 19),   # Brown (15)
    ]
    
    # Draw balls
    for i, (x, y) in enumerate(positions):
        if i < len(ball_colors):
            color = ball_colors[i]
            cv2.circle(image, (x, y), ball_radius, color, -1)
            cv2.circle(image, (x, y), ball_radius, (0, 0, 0), 2)
            
            # Add number
            cv2.putText(image, str(i+1), (x-5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add cue ball (white)
    cue_x, cue_y = center_x, height - 100
    cv2.circle(image, (cue_x, cue_y), ball_radius, (255, 255, 255), -1)
    cv2.circle(image, (cue_x, cue_y), ball_radius, (0, 0, 0), 2)
    
    # Save image
    output_path = "new_test_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"💾 Test image created: {output_path}")
    print(f"📐 Size: {width}x{height}")
    print(f"🎱 Balls: 16 (15 numbered + 1 cue ball)")
    
    return output_path

def main():
    print("🎱 Creating Billiards Test Image")
    print("=" * 40)
    
    # Create test image
    image_path = create_billiards_test_image()
    
    if image_path and os.path.exists(image_path):
        print(f"\n✅ Test image ready for YOLOv11 testing!")
        print(f"📁 File: {image_path}")
    else:
        print(f"\n❌ Failed to create test image")

if __name__ == "__main__":
    main() 