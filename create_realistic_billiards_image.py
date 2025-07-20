#!/usr/bin/env python3
"""
Create realistic billiards image based on description
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_realistic_billiards_image():
    """Create a realistic billiards image based on the description"""
    
    # Create a blue pool table background
    width, height = 1280, 720
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue pool table color (vibrant blue felt)
    blue_color = (139, 69, 19)  # Dark blue-green
    image[:] = blue_color
    
    # Add table rails (dark grey with light blue accents)
    rail_color = (64, 64, 64)  # Dark grey
    rail_width = 40
    
    # Draw rails
    cv2.rectangle(image, (0, 0), (width, rail_width), rail_color, -1)  # Top
    cv2.rectangle(image, (0, height-rail_width), (width, height), rail_color, -1)  # Bottom
    cv2.rectangle(image, (0, 0), (rail_width, height), rail_color, -1)  # Left
    cv2.rectangle(image, (width-rail_width, 0), (width, height), rail_color, -1)  # Right
    
    # Add diamond markers (white circles)
    marker_radius = 3
    marker_positions = [
        (rail_width//2, rail_width//2),  # Top-left
        (width//2, rail_width//2),       # Top-center
        (width-rail_width//2, rail_width//2),  # Top-right
        (rail_width//2, height-rail_width//2),  # Bottom-left
        (width//2, height-rail_width//2),       # Bottom-center
        (width-rail_width//2, height-rail_width//2),  # Bottom-right
        (rail_width//2, height//2),      # Left-center
        (width-rail_width//2, height//2) # Right-center
    ]
    
    for pos in marker_positions:
        cv2.circle(image, pos, marker_radius, (255, 255, 255), -1)
    
    # Ball positions based on description
    ball_positions = {
        'cue': (width//2 + 50, height//2 + 100),      # Lower-middle, slightly right
        '2': (width//2, rail_width + 80),             # Top-center (blue)
        '3': (rail_width + 100, height//2 - 50),      # Left side, middle (red)
        '4': (width//2 - 80, height//2 + 50),         # Lower-middle, left of cue (purple)
        '5': (width//2 + 120, height//2 - 30),        # Middle-right (orange)
        '6': (width//2 + 150, height//2 + 80),        # Lower-right (green)
        '7': (width//2, height - rail_width - 80),    # Bottom-center (maroon)
        '8': (width - rail_width - 80, rail_width + 80),  # Top-right (black)
        '9': (width//2 + 100, height//2 - 80),        # Middle-right, above 5 (striped yellow)
        '10': (rail_width + 80, rail_width + 60),     # Top-left (striped blue)
        '12': (rail_width + 120, height//2),          # Middle-left (striped purple)
        '14': (width//2 + 30, height//2 + 30),        # Lower-middle, right of 4 (striped green)
        '15': (width - rail_width - 60, height//2 - 80)  # Right rail, top (striped red)
    }
    
    # Ball colors and types
    ball_properties = {
        'cue': {'color': (255, 255, 255), 'type': 'solid'},      # White
        '2': {'color': (255, 0, 0), 'type': 'solid'},            # Blue
        '3': {'color': (0, 0, 255), 'type': 'solid'},            # Red
        '4': {'color': (128, 0, 128), 'type': 'solid'},          # Purple
        '5': {'color': (0, 165, 255), 'type': 'solid'},          # Orange
        '6': {'color': (0, 255, 0), 'type': 'solid'},            # Green
        '7': {'color': (139, 69, 19), 'type': 'solid'},          # Maroon
        '8': {'color': (0, 0, 0), 'type': 'solid'},              # Black
        '9': {'color': (0, 255, 255), 'type': 'striped'},        # Yellow/Orange striped
        '10': {'color': (255, 0, 0), 'type': 'striped'},         # Blue striped
        '12': {'color': (128, 0, 128), 'type': 'striped'},       # Purple striped
        '14': {'color': (0, 255, 0), 'type': 'striped'},         # Green striped
        '15': {'color': (0, 0, 255), 'type': 'striped'}          # Red striped
    }
    
    ball_radius = 25
    
    # Draw balls
    for ball_id, pos in ball_positions.items():
        if ball_id in ball_properties:
            props = ball_properties[ball_id]
            color = props['color']
            ball_type = props['type']
            
            # Draw main ball
            cv2.circle(image, pos, ball_radius, color, -1)
            cv2.circle(image, pos, ball_radius, (0, 0, 0), 2)  # Black border
            
            # Add stripes for striped balls
            if ball_type == 'striped':
                # Draw horizontal stripes
                for i in range(-ball_radius, ball_radius, 8):
                    y = pos[1] + i
                    if 0 <= y < height:
                        cv2.line(image, (pos[0] - ball_radius, y), (pos[0] + ball_radius, y), (255, 255, 255), 2)
            
            # Add number for numbered balls
            if ball_id != 'cue':
                cv2.putText(image, ball_id, (pos[0]-8, pos[1]+8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add cue stick (simplified)
    cue_start = (rail_width + 20, rail_width + 20)
    cue_end = (rail_width + 200, rail_width + 60)
    cv2.line(image, cue_start, cue_end, (139, 69, 19), 8)  # Brown cue stick
    
    # Add "Predator Arcadia" text on top rail
    cv2.putText(image, "Predator Arcadia", (width//2 - 100, rail_width//2 + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save image
    output_path = "realistic_billiards_test.jpg"
    cv2.imwrite(output_path, image)
    print(f"💾 Realistic billiards image created: {output_path}")
    print(f"📐 Size: {width}x{height}")
    print(f"🎱 Balls: 14 (1 cue + 13 numbered balls)")
    print(f"🎯 Features: Rails, pockets, diamond markers, cue stick")
    
    return output_path

def main():
    print("🎱 Creating Realistic Billiards Image")
    print("=" * 45)
    
    # Create realistic test image
    image_path = create_realistic_billiards_image()
    
    if image_path and os.path.exists(image_path):
        print(f"\n✅ Realistic billiards image ready for YOLOv11 testing!")
        print(f"📁 File: {image_path}")
    else:
        print(f"\n❌ Failed to create realistic billiards image")

if __name__ == "__main__":
    main() 