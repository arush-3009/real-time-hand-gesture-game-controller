from src import camera, config, display, gestures, keyboard_input, tracking
import time
import cv2

class GameController:

    def __init__(self):
        self.configuration = config.Config()
        self.camera = camera.Camera(self.configuration)
        self.display = display.DisplayManager(self.configuration)
        self.keyboard_control = keyboard_input.KeyboardController(self.configuration)
        self.hand_tracker = tracking.HandDetector()
        self.gesture_tracker = gestures.Gestures(self.configuration)

    def press_game_keys(self, landmarks):

        if landmarks:
            self.gesture_tracker.is_open(landmarks)
            self.gesture_tracker.is_fist(landmarks)
            self.gesture_tracker.is_v(landmarks)
            self.gesture_tracker.is_index_pointing(landmarks)
            self.gesture_tracker.get_steering_direction(landmarks)
            self.gesture_tracker.get_hand_position(landmarks)
        
            # Hand detected
            if self.gesture_tracker.gestures_data['v']:
                self.keyboard_control.handle_drift(True, self.gesture_tracker.gestures_data['steering_direction'])
            else:
                self.keyboard_control.handle_drift(False, self.gesture_tracker.gestures_data['steering_direction'])
                if self.gesture_tracker.gestures_data['fist']:
                    self.keyboard_control.handle_braking(True)
                else:
                    self.keyboard_control.handle_braking(False)
            
            self.keyboard_control.handle_acceleration(self.gesture_tracker.gestures_data['open_hand'])
            self.keyboard_control.handle_steering(self.gesture_tracker.gestures_data['steering_direction'])
            self.keyboard_control.handle_nitro(self.gesture_tracker.gestures_data['index'])
        else:
            # No hand detected
            self.keyboard_control.release_all_keys()

        
    
    def play_game(self):
        curr_time = 0
        prev_time = 0

        while self.camera.cap.isOpened():

            ret, frame = self.camera.get_frame_return_val()
            
            if not ret:
                print('Wecam feed ended or abrupted.')
                break

            frame = self.hand_tracker.find_hands(frame, draw=False)
            landmarks = self.hand_tracker.find_pos(frame)

            self.press_game_keys(landmarks)

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time > 0 else 0
            prev_time = curr_time

            display_frame = self.display.render(frame, fps, self.gesture_tracker.gestures_data, self.keyboard_control.pressed_keys)
            self.display.show(display_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nExiting...')
                break
    
    def cleanup(self):
        self.keyboard_control.release_all_keys()
        self.camera.cleanup()
        self.display.cleanup()
        print('Cleanup Complete!')



                


            






        

