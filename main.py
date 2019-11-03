import cv2
import numpy as np
import dlib
from landmark_recognition import landmarks_for_face

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

img = cv2.imread('data/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
landmarks = landmarks_for_face(detector, predictor, gray)

if landmarks is not None:
    print('Face found.')
    cnt = 0
    for (x, y) in landmarks:
        cv2.putText(img, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cnt += 1

    left_eye = landmarks[36]
    right_eye = landmarks[45]
    nose = landmarks[30]

    cv2.imshow("Frame", img)
    cv2.imwrite('data/tmp.png', img)
    cv2.waitKey()
else:
    print('Face not found.')
