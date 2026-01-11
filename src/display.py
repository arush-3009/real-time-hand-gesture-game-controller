"""
Display Manager
Handles visual feedback in two modes: info_only or webcam
"""

import cv2
import numpy as np


class DisplayManager:
    """
    Manages display in two modes:
    1. info_only: Black background with info overlay (fast)
    2. webcam: Camera feed with info overlay (visual feedback)
    """
    
    def __init__(self, config):
        """
        Initialize display manager
        
        Args:
            config: Config object (from config file) with display settings
        """
        self.config = config
        
        # Get display settings
        self.mode = config.get('display', 'mode') or 'info_only'
        self.show_fps = config.get('display', 'show_fps')
        self.show_table = config.get('display', 'show_gesture_table')
        self.show_bars = config.get('display', 'show_steering_bars')
        self.show_keys = config.get('display', 'show_keys')
        self.window_name = config.get('display', 'window_name')
        
        # Window dimensions (matches camera resolution)
        self.width = config.get('camera', 'width')
        self.height = config.get('camera', 'height')
        
        # Create window with always-on-top property
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        
        print(f"Display mode: {self.mode}")
    
    def create_canvas(self, webcam_frame=None):
        """
        Create the base canvas (black or webcam)
        
        Args:
            webcam_frame: Camera frame (only used in webcam mode)
            
        Returns:
            Canvas to draw on (numpy array)
        """
        if self.mode == 'webcam' and webcam_frame is not None:
            canvas = webcam_frame.copy()
        else:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        return canvas
    
    def draw_fps(self, canvas, fps):
        """
        Draw FPS counter in top-left corner
        
        Args:
            canvas: Image to draw on
            fps: Frames per second value
        """
        if not self.show_fps:
            return
        
        cv2.putText(
            canvas,
            f'FPS: {fps}',
            (10, 30),  
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    
    def draw_gesture_table(self, canvas, gestures):
        """
        Draw gesture detection table in top-right
        
        Args:
            canvas: Image to draw on
            gestures: Dict with gesture states
                     {'open_hand': True, 'fist': False, ...}
        """
        if not self.show_table:
            return
        
        
        start_x = self.width - 200
        start_y = 30
        line_height = 30
        
        
        cv2.putText(
            canvas,
            'GESTURES:',
            (start_x, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        
        y = start_y + line_height
        
        
        gesture_list = [
            ('open_hand', 'Open'),
            ('fist', 'Fist'),
            ('v', 'V'),
            ('index', 'Index'),
            ('steering_direction', 'Dir')
        ]
        
        
        for key, label in gesture_list:
            value = gestures.get(key)
            if key == 'steering_direction':
                display_text = f"{label}: {value}"
                color = (255, 255, 255)
            else:
                if value:
                    display_text = f"{label}: TRUE"
                    color = (0, 255, 0)
                else:
                    display_text = f"{label}: FALSE"
                    color = (100, 100, 100)
            
            cv2.putText(
                canvas,
                display_text,
                (start_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            
            y += line_height
    
    def draw_steering_bars(self, canvas, hand_x_position, steering_direction):
        """
        Draw steering visualization with boundaries
        
        Args:
            canvas: Image to draw on
            hand_x_position: Hand's x position (0.0 to 1.0), None if no hand
            steering_direction: 'left', 'center', or 'right'
        """
        if not self.show_bars:
            return
        
        
        bar_width = 300
        bar_height = 80
        bar_x = (self.width - bar_width) // 2
        bar_y = self.height - bar_height - 20
        
        
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
    
        left_threshold = self.config.get('gestures', 'left_threshold')
        right_threshold = self.config.get('gestures', 'right_threshold')
        
        
        left_line_x = bar_x + int(bar_width * left_threshold)
        right_line_x = bar_x + int(bar_width * right_threshold)
        
        
        cv2.line(
            canvas,
            (left_line_x, bar_y),
            (left_line_x, bar_y + bar_height),
            (0, 255, 255),
            2
        )
        cv2.putText(
            canvas,
            'L',
            (left_line_x - 15, bar_y + bar_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
        
        
        cv2.line(
            canvas,
            (right_line_x, bar_y),
            (right_line_x, bar_y + bar_height),
            (0, 255, 255),
            2
        )
        cv2.putText(
            canvas,
            'R',
            (right_line_x - 5, bar_y + bar_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
        
    
        cv2.putText(
            canvas,
            'LEFT',
            (bar_x + 20, bar_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        cv2.putText(
            canvas,
            'CENTER',
            (bar_x + bar_width//2 - 30, bar_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        cv2.putText(
            canvas,
            'RIGHT',
            (bar_x + bar_width - 50, bar_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        

        if hand_x_position is not None:
            indicator_x = bar_x + int(bar_width * hand_x_position)
            indicator_y = bar_y + bar_height // 2
            
            # Color based on direction
            if steering_direction == 'left':
                color = (255, 0, 0)  # Blue
            elif steering_direction == 'right':
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw indicator circle
            cv2.circle(
                canvas,
                (indicator_x, indicator_y),
                12,
                color,
                -1
            )
            
            # Draw hand emoji/text
            cv2.putText(
                canvas,
                'H',
                (indicator_x - 10, indicator_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    def draw_keys(self, canvas, pressed_keys):
        """
        Draw currently pressed keys at bottom-left
        
        Args:
            canvas: Image to draw on
            pressed_keys: Set of pressed keys
        """
        if not self.show_keys:
            return
        
    
        if pressed_keys:
            keys_text = ', '.join(sorted(pressed_keys))
        else:
            keys_text = 'None'
        
        cv2.putText(
            canvas,
            f'Keys: {keys_text}',
            (10, self.height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
    
    def render(self, webcam_frame, fps, gestures, pressed_keys):
        """
        Render complete display with all elements
        
        Args:
            webcam_frame: Camera frame (can be None in info_only mode)
            fps: Current FPS
            gestures: Dict with all gesture states
            pressed_keys: Set of pressed keys
            
        Returns:
            Complete rendered frame
        """
        # Create base canvas
        canvas = self.create_canvas(webcam_frame)
        
        # Draw all elements
        self.draw_fps(canvas, fps)
        self.draw_gesture_table(canvas, gestures)
        self.draw_steering_bars(
            canvas,
            gestures.get('hand_x'),
            gestures.get('steering_direction', 'center')
        )
        self.draw_keys(canvas, pressed_keys)
        
        return canvas
    
    def show(self, frame):
        """
        Display the frame in window
        
        Args:
            frame: Image to display
        """
        cv2.imshow(self.window_name, frame)
    
    def cleanup(self):
        """Close all windows"""
        cv2.destroyAllWindows()