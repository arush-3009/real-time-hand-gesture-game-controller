import cv2
import mediapipe as mp
import os
import time
import numpy as np

GESTURES = {
    '1': "open_hand",
    '2': "fist",
    '3': "v_sign",
    '4': "index_pointing",
    '5': "no_gesture"
}

IMG_SIZE = 224
CAPTURE_INTERVAL = 0.2
DATASET_PATH = "dataset/raw"
PADDING = 30 #Padding for bouding box aroung the hand in the image

for gesture_name in GESTURES.values():
    folder_path = os.path.join(DATASET_PATH, gesture_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created Folder: {folder_path}")


#Get bouding box around hands
def get_hand_bounding_box(landmarks, frame_width, frame_height, padding):
    """
    Calculate bounding box around detected hand.
    Args:
        landmarks: MediaPipe hand landmarks (21 points)
        frame_width: Width of video frame in pixels
        frame_height: Height of video frame in pixels
        padding: Padding to around hand in the image to ensure nothing gets cut off, in pixels.
    
    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates
        or None if no valid box
    """
    if not landmarks:
        return None
    

    x_coordinates = [lmk.x for lmk in landmarks]
    y_coordinates = [lmk.y for lmk in landmarks]

    #Get extreme coordinates (in normalized form right now)
    x_min_norm = min(x_coordinates)
    y_min_norm = min(y_coordinates)
    x_max_norm = max(x_coordinates)
    y_max_norm = max(y_coordinates)

    #Convert Normalized coordinates to pixels and add Padding
    x_min = int(x_min_norm * frame_width) - padding
    x_max = int(x_max_norm * frame_width) + padding
    y_min = int(y_min_norm * frame_height) - padding
    y_max = int(y_max_norm * frame_height) + padding

    #Limit the extremities to always be inside the entire image's frame
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)

    return (x_min, y_min, x_max, y_max)