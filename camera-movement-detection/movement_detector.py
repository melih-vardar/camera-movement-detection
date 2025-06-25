import cv2
import numpy as np
from typing import List

def detect_significant_movement(frames: List[np.ndarray]) -> List[int]:
    """
    Detect frames where significant camera movement occurs.
    Args:
        frames: List of image frames (as numpy arrays).
    Returns:
        List of indices where significant movement is detected.
    """

    if len(frames) < 2:
        return []
    
    movement_indices = []
    
    for idx in range(1, len(frames)):
        if is_camera_movement_features(frames[idx-1], frames[idx]):
            movement_indices.append(idx)
    
    return movement_indices

def is_camera_movement_features(frame1: np.ndarray, frame2: np.ndarray) -> bool:

    # feature-based camera movement detection --> goodFeaturesToTrack + findHomography
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if len(frame2.shape) == 3 else frame2
    
    # strong corners' detection
    feature_params = dict(
        maxCorners=100,    
        qualityLevel=0.01,  
        minDistance=10,     
        blockSize=3          
    )
    
    features1 = cv2.goodFeaturesToTrack(gray1, **feature_params)
    
    if features1 is None or len(features1) < 10:
        return False
    
    # optical flow - feature tracking
    lk_params = dict(
        winSize=(15, 15),   
        maxLevel=2,          
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # features in frame2 (Lucas-Kanade optical flow)
    features2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, features1, None, **lk_params
    )
    
    # successful tracking features
    good_features1 = features1[status == 1]
    good_features2 = features2[status == 1]
    
    if len(good_features1) < 8:
        return False
    
    # homography - geometric transformation
    try:
        homography, mask = cv2.findHomography(
            good_features1, good_features2, 
            cv2.RANSAC,  # outlier removal
            5.0  # threshold
        )
        
        if homography is None:
            return False
        
        # homography analysis - camera movement detection
        return analyze_homography_for_camera_movement(homography, mask, good_features1, good_features2)
        
    except cv2.error:
        return False

def analyze_homography_for_camera_movement(homography: np.ndarray, mask: np.ndarray, 
                                         features1: np.ndarray, features2: np.ndarray) -> bool:

    if homography is None or homography.shape != (3, 3):
        return False
    
    # inlier ratio check (how many features are successfully matched)
    inlier_ratio = np.sum(mask) / len(mask) if len(mask) > 0 else 0
    if inlier_ratio < 0.7:
        return False
    
    # feature movement consistency analysis
    movements = features2 - features1
    movement_magnitudes = np.sqrt(movements[:, 0]**2 + movements[:, 1]**2)
    avg_movement = np.mean(movement_magnitudes)
    std_movement = np.std(movement_magnitudes)
    
    # movement consistency: camera movement = all features move in the same direction
    movement_consistency = 1 - (std_movement / (avg_movement + 0.001))
    
    # 1. translation (translation) - PAN/TILT movement
    tx = homography[0, 2]  
    ty = homography[1, 2] 
    translation_magnitude = np.sqrt(tx*tx + ty*ty)
    
    # 2. scale change - zoom in/out
    transform_matrix = homography[:2, :2]
    scale = np.sqrt(np.linalg.det(transform_matrix))
    
    # 3. perspective distortion - camera angle change
    perspective_change = abs(homography[2, 0]) + abs(homography[2, 1])
    
    # 4. rotation - camera rotation
    rotation_angle = np.arctan2(homography[1, 0], homography[0, 0])
        
    if translation_magnitude > 1.0 and movement_consistency > 0.3:
        return True
    
    if abs(scale - 1.0) > 0.02 and movement_consistency > 0.3:
        return True
    
    if abs(rotation_angle) > 0.01 and movement_consistency > 0.3:
        return True
    
    if avg_movement > 2.0 and movement_consistency > 0.5:
        return True
    
    if translation_magnitude > 5.0 or abs(scale - 1.0) > 0.1 or abs(rotation_angle) > 0.05:
        return True
    
    return False
