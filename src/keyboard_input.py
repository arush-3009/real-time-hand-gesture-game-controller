from pynput.keyboard import Key, Controller
import time



class KeyboardController:

    def __init__(self, config):
        self.config = config

        self.keyboard = Controller()
        self.pressed_keys = set()
        self.brake_state = None  # None, 'just_braking', or 'reversing'
        self.brake_start_time = 0
        self.smooth_counters = {
            'acceleration_gesture': 0,
            'braking_reversing_gesture': 0,
            'drifting_gesture': 0,
            'nitro_gesture': 0
        }
        
        self.SMOOTH_THRESHOLD = self.config.get('gestures', 'smoothing_threshold')
        self.brake_to_reverse_delay = self.config.get('gestures', 'brake_to_reverse_delay')
    
    def press_key(self, k):
        if k not in self.pressed_keys:
            self.keyboard.press(k)
            self.pressed_keys.add(k)
    
    def release_key(self, k):
        if k in self.pressed_keys:
            self.keyboard.release(k)
            self.pressed_keys.remove(k)
    
    def release_all_keys(self):
        while self.pressed_keys:
            key = self.pressed_keys.pop()
            self.keyboard.release(key)
        self.brake_state = None

        for key in self.smooth_counters:
            self.smooth_counters[key] = 0

    #steering
    def handle_steering(self, steering_direction):
        if self.brake_state == 'just_braking': 
            self.release_key('a')
            self.release_key('d')
        else:
            if steering_direction == 'left':
                self.release_key('d')
                self.press_key('a')
            elif steering_direction == "right":
                self.release_key('a')
                self.press_key('d')
            else:
                self.release_key('a')
                self.release_key('d')


    #accelarating
    def handle_acceleration(self, gesture_active):
        if gesture_active:

            self.release_key('s')
            self.smooth_counters['braking_reversing_gesture'] = 0
            self.smooth_counters['drifting_gesture'] = 0

            self.smooth_counters['acceleration_gesture'] = self.SMOOTH_THRESHOLD
            self.press_key('w')
        else:
            self.smooth_counters['acceleration_gesture'] -= 1
            if self.smooth_counters['acceleration_gesture'] <= 0:
                self.release_key('w')
                self.smooth_counters['acceleration_gesture'] = 0


    def handle_braking(self, gesture_activate):
        if gesture_activate:

            self.release_key('w')
            self.smooth_counters['acceleration_gesture'] = 0

            self.smooth_counters['braking_reversing_gesture'] = self.SMOOTH_THRESHOLD
            # check if first time braking
            if self.brake_state == None:
                self.brake_state = 'just_braking'
                self.brake_start_time = time.time()
                # Release steering when starting to brake
                self.release_key('a')
                self.release_key('d')
            
            # Check if should transition to reversing
            elif self.brake_state == 'just_braking':
                curr_time = time.time()
                elapsed_time = curr_time - self.brake_start_time

                if elapsed_time >= self.brake_to_reverse_delay:
                    self.brake_state = 'reversing'
            
            self.press_key('s')

        else:
            self.smooth_counters['braking_reversing_gesture'] -= 1
            if self.smooth_counters['braking_reversing_gesture'] <= 0:
                #Release brake
                self.release_key('s')
                #Reset state to None
                self.brake_state = None
                self.smooth_counters['braking_reversing_gesture'] = 0



    #nitro
    def handle_nitro(self, gesture_active):
        if gesture_active:
            self.smooth_counters['nitro_gesture'] = self.SMOOTH_THRESHOLD
            self.press_key('n')
        else:
            self.smooth_counters['nitro_gesture'] -= 1
            if self.smooth_counters['nitro_gesture'] <= 0:
                self.release_key('n')
                self.smooth_counters['nitro_gesture'] = 0

    
    #drifiting
    def handle_drift(self, gesture_active, steering_direction):
        if gesture_active:

            self.release_key('w')
            self.smooth_counters['acceleration_gesture'] = 0

            self.smooth_counters['drifting_gesture'] = self.SMOOTH_THRESHOLD

            if steering_direction == 'left' or steering_direction == 'right':
                self.press_key('s')
        else:
            self.smooth_counters['drifting_gesture'] -= 1
            if self.smooth_counters['drifting_gesture'] <= 0:
                self.release_key('s')
                self.smooth_counters['drifting_gesture'] = 0


    


        
if __name__ == '__main__':

    kc = KeyboardController()
    kc.press_key(Key.shift)
    print(kc.pressed_keys) 

    kc.press_key('a')
    print(kc.pressed_keys)

    print(f'Does the state set have key "shift" ? : {Key.shift in kc.pressed_keys}\n')

    kc.release_key('a')
    print(kc.pressed_keys)
    kc.release_key(Key.shift)
    
    
    kc.press_key(Key.space)
    kc.release_key(Key.space)
    time.sleep(0.01)
    kc.press_key(Key.space)
    kc.release_key(Key.space)
    
