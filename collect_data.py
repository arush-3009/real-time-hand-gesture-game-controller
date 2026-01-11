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
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "raw")
PADDING = 30 #Padding for bouding box aroung the hand in the image

for gesture_name in GESTURES.values():
    folder_path = os.path.join(DATASET_PATH, gesture_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created Folder: {folder_path}")


#Get bouding box around hands
def get_hand_bounding_box(landmarks, frame_width, frame_height):
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
    x_min = int(x_min_norm * frame_width) - PADDING
    x_max = int(x_max_norm * frame_width) + PADDING
    y_min = int(y_min_norm * frame_height) - PADDING
    y_max = int(y_max_norm * frame_height) + PADDING

    #Limit the extremities to always be inside the entire image's frame
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)

    return (x_min, y_min, x_max, y_max)


def save_image(frame, bbox, gesture_name, counter, saved_img_size=224):
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

    if bbox is None:
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

        self.counters = {}
        for name in GESTURES.values():
            self.counters[name] = 0

        print("="*60)
        print("Data Collection Ready")
        print("="*60)

        print("\nPress the following buttons to capture the respective gestures:\n")
        for key, value in GESTURES.items():
            print(f"Gesture: {value}  ->  Button: {key}")
        print(f"\nAdditionally, press:\nSPACE -> Toggle auto-capture\nQ -> Quit")
        print('-'*45)

    def run(self):

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print(f"Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            #Get landmarks info and the bouding box
            landmarks = None
            bbox = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark

                #Draw landmarks on frame
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                #Get the bouding box
                h, w, c = frame.shape
                bbox = get_hand_bounding_box(landmarks, w, h)

                # Draw bounding box
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
                
            if self.is_capturing and self.current_gesture and landmarks:
                current_time = time.time()
                if current_time - self.last_capture_time >= CAPTURE_INTERVAL:
                    gesture_name = GESTURES[self.current_gesture]
                    success = save_image(frame, bbox, gesture_name, self.counters[gesture_name])
                    if success:
                        self.counters[gesture_name] += 1
                        self.last_capture_time = current_time
            
            # Display info on frame
            self.draw_info(frame)
            
            # Show frame
            cv2.imshow('Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key(key):
                break  # User pressed 'q'
        
        self.cleanup()

    
    def draw_info(self, frame):
        """Draw UI information on frame"""
        
        # Status text
        y_pos = 30
        
        # Current mode
        if self.current_gesture:
            mode_text = f"Mode: {GESTURES[self.current_gesture].upper()}"
            color = (0, 255, 0)
        else:
            mode_text = "Mode: NONE (Press 1-5)"
            color = (0, 0, 255)
        
        cv2.putText(frame, mode_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_pos += 35
        
        # Capture status
        if self.is_capturing:
            status_text = "Capturing: ON"
            status_color = (0, 255, 0)
        else:
            status_text = "Capturing: OFF"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        y_pos += 35
        
        # Current gesture count
        if self.current_gesture:
            gesture_name = GESTURES[self.current_gesture]
            count = self.counters[gesture_name]
            count_text = f"Count: {count}"
            cv2.putText(frame, count_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 35
        
        # All gesture counts (bottom right)
        y_pos = frame.shape[0] - 150
        for name in GESTURES.values():
            count_text = f"{name}: {self.counters[name]}"
            cv2.putText(frame, count_text, (frame.shape[1] - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25


    def handle_key(self, key):
            """
            Handle keyboard input
            Args:
                key: Key code from cv2.waitKey()
            Returns:
                True to continue, False to quit
            """
            
            # Gesture mode selection (1-5)
            if chr(key) in GESTURES:
                self.current_gesture = chr(key)
                self.is_capturing = False  # Stop capturing when switching modes
                print(f"\n>>> Switched to {GESTURES[chr(key)].upper()} mode")
                return True
            
            # Toggle capturing (SPACE)
            elif key == 32:  # SPACE key
                if self.current_gesture:
                    self.is_capturing = not self.is_capturing
                    status = "ON" if self.is_capturing else "OFF"
                    print(f">>> Capturing {status}")
                else:
                    print(">>> Select a gesture mode first (1-5)")
                return True
            
            # Quit
            elif key == ord('q'):
                print("\n>>> Quitting...")
                return False
            
            return True

    def cleanup(self):
        """Release resources and print statistics"""
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print("COLLECTION COMPLETE")
        print("="*50)
        print("\nImages collected:")
        total = 0
        for name, count in self.counters.items():
            print(f"  {name}: {count}")
            total += count
        print(f"\nTotal: {total} images")
        print("="*50)


if __name__ == '__main__':
    try:
        collector = DataCollector((1280, 720))
        collector.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise