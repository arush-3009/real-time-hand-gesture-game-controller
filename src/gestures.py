import cv2
import math


class Gestures:

    def __init__(self, config):
        self.gestures_data = {
            'open_hand': False,
            'fist': False,
            'v': False,
            'index': False,
            'steering_direction': 'center',
            'hand_x': 0.0
        }
        self.config = config

    def calc_distance(self, p1, p2):
        return math.sqrt(((p1.x - p2.x)**2) + ((p1.y - p2.y)**2))
    
    def get_hand_size(self, landmarks):
        wrist = landmarks[0]
        middle_base = landmarks[9]
        return self.calc_distance(wrist, middle_base)

    def get_hand_position(self, landmarks):
        wrist = landmarks[0]
        self.gestures_data['hand_x'] = wrist.x
        return wrist.x

    def is_fist(self, landmarks, display_output=False, img=None):
        if len(landmarks) == 0: return None
        wrist = landmarks[0]
        middle_finger_base = landmarks[9]
        hand_size = self.get_hand_size(landmarks)
        ret = True
        tips_and_thresholds = {8: 1, 12: 1, 16: 0.85, 20: 0.85}
        for fingertip in tips_and_thresholds:
            dist = self.calc_distance(landmarks[fingertip], wrist)
            ratio = dist/hand_size
            if ratio > tips_and_thresholds[fingertip]:
                ret = False
                break
        
        thumb_tip = landmarks[4]
        ring_knuckle = landmarks[14]
        thumb_to_ring = self.calc_distance(thumb_tip, ring_knuckle)
        thumb_ring_hand_size_ratio = thumb_to_ring/hand_size
        if thumb_ring_hand_size_ratio > 0.35: ret = False

        if display_output:
            if img is None:
                raise ValueError("img required when display_output=True")
            cv2.putText(img, f'Fist: {ret}', (50, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        self.gestures_data['fist'] = ret
        return ret
    
    def is_open(self, landmarks, display_output=False, img=None):
        if len(landmarks) == 0: return None
        wrist = landmarks[0]
        middle_finger_base = landmarks[9]
        hand_size = self.get_hand_size(landmarks)
        ret = True
        tips_and_thresholds = {8: 1.6, 12: 1.8, 16: 1.6, 20: 1.4}
        for fingertip in tips_and_thresholds:
            dist = self.calc_distance(landmarks[fingertip], wrist)
            ratio = dist/hand_size
            if ratio < tips_and_thresholds[fingertip]:
                ret = False
                break
        
        thumb_tip = landmarks[4]
        ring_knuckle = landmarks[14]
        thumb_to_ring = self.calc_distance(thumb_tip, ring_knuckle)
        thumb_ring_hand_size_ratio = thumb_to_ring/hand_size
        if thumb_ring_hand_size_ratio < 0.35: ret = False

        if display_output:
            if img is None:
                raise ValueError("img required when display_output=True")
            cv2.putText(img, f'Open Hand: {ret}', (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        self.gestures_data['open_hand'] = ret
        return ret
    
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
    
    def is_index_pointing(self, landmarks, display_output=False, img=None):

        if len(landmarks) == 0: return None
        hand_size = self.get_hand_size(landmarks)
        wrist = landmarks[0]
        tips_and_thresholds = {8: 1.65, 12: 0.8, 16: 0.8, 20: 0.8}
        ret = True
        for fingertip in tips_and_thresholds:
            dist = self.calc_distance(wrist, landmarks[fingertip])
            ratio = dist/hand_size
            if fingertip == 8 and ratio < tips_and_thresholds[fingertip]:
                ret = False
                break
            elif fingertip != 8 and ratio > tips_and_thresholds[fingertip]:
                ret = False
                break
            
        if display_output:
            if img is None:
                raise ValueError("img required when display_output=True")
            cv2.putText(img, f'Index Pointing: {ret}', (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        
        self.gestures_data['index'] = ret
        
        return ret
    
    def is_v(self, landmarks, display_output=False, img=None):

        if len(landmarks) == 0: return None

        hand_size = self.get_hand_size(landmarks)
        wrist = landmarks[0]
        tips_and_thresholds = {8: 1.7, 12: 1.7, 16: 1, 20: 1}
        ret = True

        extended_fingers = {8, 12}

        for fingertip in tips_and_thresholds:
            dist = self.calc_distance(wrist, landmarks[fingertip])
            ratio = dist / hand_size

            if fingertip in extended_fingers and ratio < tips_and_thresholds[fingertip]:
                ret = False
                break
            elif fingertip not in extended_fingers and ratio > tips_and_thresholds[fingertip]:
                ret = False
                break

        if display_output:
            if img is None:
                raise ValueError("img required when display_output=True")
            cv2.putText(img, f'V showing: {ret}', (50, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        
        self.gestures_data['v'] = ret
            
        return ret