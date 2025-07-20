"""
Multi-Class Ball Detectors Package
Các detector để detect bóng sử dụng tất cả các class 0-15
"""

from .multi_class_ball_detector import detect_all_balls, draw_multi_class_balls
from .multi_class_video_detector import process_video

__all__ = [
    'detect_all_balls',
    'draw_multi_class_balls', 
    'process_video'
] 