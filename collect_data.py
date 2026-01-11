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


def save_image(frame, bbox, saved_img_size, gesture_name, counter):
    """
    Crop the hand out of the image, resize to an appropriate size for Model input, and count the number.

    Args:
    frame: The original image as captured by webcam.
    bbox: Coordinates of the appropriate path of the images that captures the hand.
    saved_img_size: The size the final, saved images should be (in pixels).
    gesture_name: Name of the gesture which the images shows.
    counter: Number of images saved for the gesture.

    Returns:
    True if image saved in folder, False otherwise.
    """

    if bbox in None:
        return False

    x_min, y_min, x_max, y_max = bbox

    #Crop the path of the image with the hand
    hand_cropped = frame[y_min:y_max, x_min:x_max]
    if hand_cropped.size == 0:
        return False
    
    #Resize hand to specified size
    hand_resized = cv2.resize(hand_cropped, (saved_img_size, saved_img_size))

    file_name = f"{gesture_name}_{counter:04d}.jpg"

    filepath = os.path.join(DATASET_PATH, gesture_name, file_name)

    #Save to disk
    ret = cv2.imwrite(filepath, hand_resized)

    return ret


class DataCollector:
    """
    Hand Gesuture Images data collection using the webcam.
    """
    def __init__(self, webcam_resolution):

        #Set up webcam
        camera_w, camera_h = webcam_resolution
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Issue accessing Webcam!")

        #MediaPipe Hand Detection
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

        self.mp_draw = mp.solutions.drawing_utils

        #Initialize state machine for tracking
        self.current_gesture = None
        self.is_capturing = False
        self.last_capture_time = 0

        self.counter = {}
        for name in GESTURES.values():
            self.counter[name] = 0

        print("="*60)
        print("Data Collection Ready")
        print("="*60)

        print("\nPress the following buttons to capture the respective gestures:\n")
        for key, value in GESTURES.items():
            print(f"Gesture: {value}  ->  Button: {key}")
        print(f"\nAdditionally, press:\nSPACE -> Toggle auto-capture\nQ -> Quit")
        print('-'*45)
