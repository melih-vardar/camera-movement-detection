import cv2
import numpy as np
from typing import List

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Detect frames where significant camera movement occurs.
    Args:
        frames: List of image frames (as numpy arrays).
        threshold: Sensitivity threshold for detecting movement.
    Returns:
        List of indices where significant movement is detected.
    """

    if len(frames) < 2:
        return []
    
    movement_indices = []
    
    for idx in range(1, len(frames)):
        if is_camera_movement(frames[idx-1], frames[idx]):
            movement_indices.append(idx)
    
    return movement_indices

def analyze_edge_changes(binary_diff: np.ndarray) -> float:

    height, width = binary_diff.shape
    
    # determine edge locations (%10)
    edge_height = max(1, height // 10)
    edge_width = max(1, width // 10)
    
    top_edge = binary_diff[:edge_height, :]
    bottom_edge = binary_diff[-edge_height:, :]
    left_edge = binary_diff[:, :edge_width]
    right_edge = binary_diff[:, -edge_width:]
    
    # count edge changes
    edge_changes = (np.sum(top_edge > 0) 
                    + np.sum(bottom_edge > 0) 
                    + np.sum(left_edge > 0) 
                    + np.sum(right_edge > 0))
    
    # total edge pixels
    total_edge_pixels = (top_edge.size
                        + bottom_edge.size
                        + left_edge.size
                        + right_edge.size)
    
    return (edge_changes / total_edge_pixels) * 100 if total_edge_pixels > 0 else 0.0

def is_camera_movement(frame1: np.ndarray, frame2: np.ndarray) -> bool:

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
    
    diff = cv2.absdiff(gray1, gray2)
    _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    changed_pixels = np.sum(binary_diff > 0)
    total_pixels = binary_diff.shape[0] * binary_diff.shape[1]
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # we divide the frame into 9 regions (3x3) and count the number of regions with change
    height, width = binary_diff.shape
    regions_with_change = 0
    
    for i in range(3):
        for j in range(3):
            # coordinates of each region
            region_y1 = i * height // 3
            region_y2 = (i + 1) * height // 3
            region_x1 = j * width // 3
            region_x2 = (j + 1) * width // 3
            
            # is there any change in this region?
            region = binary_diff[region_y1:region_y2, region_x1:region_x2]
            if np.sum(region > 0) > 0:
                regions_with_change += 1
    
    # the percentage of edge changes
    edge_change_percentage = analyze_edge_changes(binary_diff)
    
    # camera movement rules:
    if change_percentage >= 5.0 and regions_with_change >= 6 and edge_change_percentage >= 3.0:
        return True
    
    elif change_percentage >= 8.0 and regions_with_change >= 7:
        return True
    
    elif change_percentage >= 15.0 and edge_change_percentage >= 5.0:
        return True
    
    return False
