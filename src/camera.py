import cv2

class Camera:

    def __init__(self, config):
        self.config = config
        self.cap = cv2.VideoCapture(self.config.get('camera', 'device_id'))
        if not self.cap.isOpened():
            raise OSError("Cannot access webcam")
        
        self.cap.set(3, self.config.get('camera', 'width'))
        self.cap.set(4, self.config.get('camera', 'height'))

    def get_frame_return_val(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        return (ret, frame)
    
    def cleanup(self):
        self.cap.release()