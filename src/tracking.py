# """
# HandTracking using Mediapipe and OpenCV. Gives an easy and short way to use MediaPipe's hand tracking
# """
#

import cv2
import mediapipe as mp
import time

class HandDetector:

    def __init__(self, static_mode = False, max_hands = 1, model_complexity = 1, det_con = 0.5, track_con = 0.3):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.det_con = det_con
        self.track_con = track_con
        self.model_complexity = model_complexity

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_mode, self.max_hands, self.model_complexity,self.det_con, self.track_con)

        self.mp_draw = mp.solutions.drawing_utils


    def find_hands(self, img, draw = True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lmks, self.mp_hands.HAND_CONNECTIONS)

        return img


    def find_pos(self, img, hand_no=0, normalized_coordinates=True, draw=False, draw_pos_no=False):
        lm_list = []
        normalized_lm_list = []
        if self.results.multi_hand_landmarks:
            for i, lmk in enumerate(self.results.multi_hand_landmarks[hand_no].landmark):
                h, w, c = img.shape
                cx = int(w * lmk.x)
                cy = int(h*lmk.y)
                lm_list.append([i, cx, cy])
                normalized_lm_list.append(lmk)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), -1)
                if draw_pos_no:
                    cv2.putText(img, str(i), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if normalized_coordinates:
            return normalized_lm_list
        else:
            return lm_list




if __name__ == '__main__':
    detector = HandDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Issue opening webcam')
        exit()

    ctime = 0
    ptime = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            print('Video ended successfully or abrupted')
            break
        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        lst = detector.find_pos(frame, draw=True)
        if len(lst) != 0:
            for i in lst:
                print(i)
        ctime = time.time()
        fps = int(1/(ctime - ptime))
        ptime = ctime
        cv2.putText(frame, str(fps), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 4)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()