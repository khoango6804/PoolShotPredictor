#!/usr/bin/env python3
"""
Create billiards image based on user's detailed description
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_user_billiards_image():
    """Create a billiards image based on the user's detailed description"""
    
    # Create a blue pool table background
    width, height = 1280, 720
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Bright blue pool table color
    blue_color = (139, 69, 19)  # Bright blue
    image[:] = blue_color
    
    # Add table rails (dark grey)
    rail_color = (64, 64, 64)  # Dark grey
    rail_width = 40
    
    # Draw rails
    cv2.rectangle(image, (0, 0), (width, rail_width), rail_color, -1)  # Top
    cv2.rectangle(image, (0, height-rail_width), (width, height), rail_color, -1)  # Bottom
    cv2.rectangle(image, (0, 0), (rail_width, height), rail_color, -1)  # Left
    cv2.rectangle(image, (width-rail_width, 0), (width, height), rail_color, -1)  # Right
    
    # Add diamond markers (white dots)
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
    
    # Add pockets (black circles)
    pocket_radius = 15
    pocket_positions = [
        (rail_width//2, rail_width//2),  # Top-left
        (width//2, rail_width//2),       # Top-center
        (width-rail_width//2, rail_width//2),  # Top-right
        (rail_width//2, height-rail_width//2),  # Bottom-left
        (width//2, height-rail_width//2),       # Bottom-center
        (width-rail_width//2, height-rail_width//2)  # Bottom-right
    ]
    
    for pos in pocket_positions:
        cv2.circle(image, pos, pocket_radius, (0, 0, 0), -1)
    
    # Ball positions based on user's detailed description
    ball_positions = {
        'cue': (rail_width + 80, rail_width + 80),              # Top-left, near corner pocket
        '1': (rail_width + 120, height - rail_width - 120),     # Bottom-left, near corner
        '2': (width//2 + 50, rail_width + 100),                 # Top-center-right (blue)
        '3': (width - rail_width - 150, rail_width + 100),      # Top-right, near center pocket
        '3_2': (width - rail_width - 80, rail_width + 80),      # Top-right, near corner
        '3_3': (width - rail_width - 100, height - rail_width - 100),  # Bottom-right
        '5': (width//2 + 100, rail_width + 80),                 # Top-left area (orange)
        '5_2': (width - rail_width - 120, rail_width + 80),     # Top-right area (orange)
        '6': (rail_width + 100, rail_width + 120),              # Top-left area (green)
        '6_2': (rail_width + 120, height - rail_width - 80),    # Bottom-left (green)
        '7': (rail_width + 140, rail_width + 100),              # Top-left area (brown/red)
        '8': (width//2, height - rail_width - 80),              # Bottom-center (black)
        '9': (rail_width + 60, rail_width + 100),               # Top-left, near cue (striped yellow)
        '12': (width - rail_width - 100, height - rail_width - 80)  # Bottom-right (striped purple)
    }
    
    # Ball colors and types based on description
    ball_properties = {
        'cue': {'color': (255, 255, 255), 'type': 'solid'},      # White
        '1': {'color': (255, 255, 0), 'type': 'solid'},          # Yellow
        '2': {'color': (255, 0, 0), 'type': 'solid'},            # Blue
        '3': {'color': (0, 0, 255), 'type': 'solid'},            # Red
        '3_2': {'color': (0, 0, 255), 'type': 'solid'},          # Red
        '3_3': {'color': (0, 0, 255), 'type': 'solid'},          # Red
        '5': {'color': (0, 165, 255), 'type': 'solid'},          # Orange
        '5_2': {'color': (0, 165, 255), 'type': 'solid'},        # Orange
        '6': {'color': (0, 255, 0), 'type': 'solid'},            # Green
        '6_2': {'color': (0, 255, 0), 'type': 'solid'},          # Green
        '7': {'color': (139, 69, 19), 'type': 'solid'},          # Brown/Red
        '8': {'color': (0, 0, 0), 'type': 'solid'},              # Black
        '9': {'color': (0, 255, 255), 'type': 'striped'},        # Yellow striped
        '12': {'color': (128, 0, 128), 'type': 'striped'}        # Purple striped
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
                # Extract number from ball_id (remove _2, _3 suffixes)
                number = ball_id.split('_')[0]
                cv2.putText(image, number, (pos[0]-8, pos[1]+8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add cue stick (black) and player's arm
    cue_start = (rail_width + 20, rail_width + 20)
    cue_end = (rail_width + 180, rail_width + 60)
    cv2.line(image, cue_start, cue_end, (0, 0, 0), 8)  # Black cue stick
    
    # Add player's arm (simplified)
    arm_start = (rail_width + 10, rail_width + 10)
    arm_end = (rail_width + 40, rail_width + 40)
    cv2.line(image, arm_start, arm_end, (139, 69, 19), 6)  # Brown arm
    
    # Save image
    output_path = "user_billiards_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"💾 User billiards image created: {output_path}")
    print(f"📐 Size: {width}x{height}")
    print(f"🎱 Balls: 15 (1 cue + 14 numbered balls)")
    print(f"🎯 Features: Rails, 6 pockets, diamond markers, cue stick, player arm")
    print(f"📋 Ball distribution: Scattered across table (not in triangle formation)")
    
    return output_path

def main():
    print("🎱 Creating User Billiards Image")
    print("=" * 40)
    
    # Create user test image
    image_path = create_user_billiards_image()
    
    if image_path and os.path.exists(image_path):
        print(f"\n✅ User billiards image ready for YOLOv11 testing!")
        print(f"📁 File: {image_path}")
    else:
        print(f"\n❌ Failed to create user billiards image")

if __name__ == "__main__":
    main() 