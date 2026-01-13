import cv2
import math

from ml.inference import GesturePredictor
import ml.config as cnn_config


class Gestures:

    def __init__(self, config, prediction_confidence=0.8):
        self.gestures_data = {
            'open_hand': False,
            'fist': False,
            'v': False,
            'index': False,
            'steering_direction': 'center',
            'hand_x': 0.0
        }
        self.config = config

        #Initialize a GesturePredictor object in order to predict hand gesture using the CNN
        self.predictor = GesturePredictor(
            path_to_trained_model=(cnn_config.MODELS_DIR / cnn_config.MODEL_SAVE_NAME),
            mean_norm=cnn_config.NORMALIZE_MEAN,
            std_norm=cnn_config.NORMALIZE_STD,
            device=cnn_config.DEVICE,
            img_size=cnn_config.IMG_SIZE,
            class_names=cnn_config.CLASS_NAMES
            )

        self.PRED_CONF = prediction_confidence

    def get_hand_position(self, landmarks):
        wrist = landmarks[0]
        self.gestures_data['hand_x'] = wrist.x
        return wrist.x

    def get_steering_direction(self, landmarks, display_output=False, img=None):
        if len(landmarks) == 0: return 'center'
        x = landmarks[0].x
        
       
        LEFT_THRESHOLD = self.config.get('gestures', 'left_threshold')
        RIGHT_THRESHOLD = self.config.get('gestures', 'right_threshold')

        direction = None

        if x < LEFT_THRESHOLD:
            direction = 'left'
        elif x > RIGHT_THRESHOLD:
            direction = 'right'
        else:
            direction = 'center'
        
        if display_output:
            if img is None:
                raise ValueError("img required when display_output=True")
            cv2.putText(img, f'Direction: {direction}', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        
        self.gestures_data['steering_direction'] = direction

        return direction

    def classify_gesture_cnn(self, frame, bbox):

        # Reset all gesture states
        self.gestures_data['open_hand'] = False
        self.gestures_data['fist'] = False
        self.gestures_data['v'] = False
        self.gestures_data['index'] = False
        
        # Get prediction
        predicted_gesture = self.predictor.predict_with_threshold(frame, bbox, self.PRED_CONF)
        
        # Map CNN output to gestures_data keys
        gesture_mapping = {
            'open_hand': 'open_hand',
            'fist': 'fist',
            'v_sign': 'v',
            'index_pointing': 'index',
            'no_gesture': None
        }
        
        # Set the predicted gesture
        if predicted_gesture in gesture_mapping:
            key = gesture_mapping[predicted_gesture]
            if key is not None:
                self.gestures_data[key] = True
