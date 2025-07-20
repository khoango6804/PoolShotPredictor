import cv2
import numpy as np
from ..config.config import CAMERA_WIDTH, CAMERA_HEIGHT, FPS

def initialize_camera(camera_id=0):
    """
    Initialize camera with specified settings
    
    Args:
        camera_id: ID of the camera to use
        
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    return cap

def read_frame(cap):
    """
    Read a frame from the camera
    
    Args:
        cap: VideoCapture object
        
    Returns:
        frame: numpy array of the frame
        success: boolean indicating if frame was read successfully
    """
    success, frame = cap.read()
    if not success:
        return None, False
        
    return frame, True

def save_frame(frame, filename):
    """
    Save a frame to a file
    
    Args:
        frame: numpy array of the frame
        filename: name of the file to save
    """
    cv2.imwrite(filename, frame)

def draw_fps(frame, fps):
    """
    Draw FPS on the frame
    
    Args:
        frame: numpy array of the frame
        fps: current FPS value
        
    Returns:
        frame with FPS drawn
    """
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    return frame 