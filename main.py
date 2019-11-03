import argparse

import cv2
import numpy as np
import dlib
from landmark_recognition import landmarks_for_face


parser = argparse.ArgumentParser(description='Head pose estimation (yaw, pitch, roll)')
parser.add_argument('-i',
                    action='store',
                    dest='input_type',
                    help='type of input: photo, video, camera',
                    choices=['photo', 'video', 'camera'],
                    default='camera')
parser.add_argument('-p',
                    action='store',
                    dest='path',
                    help='path to input file',
                    required=False,
                    type=str)
parser.add_argument('-m',
                    action='store',
                    dest='method',
                    help='method used to estimate head pose: 0 - 3D model, 1 - planes, 2 - estimated 000 location',
                    choices=[0, 1, 2],
                    type=int,
                    default=0)
args = parser.parse_args()

if args.input_type != 'camera' and args.path is None:
    print('You need to provide file input.')
    exit(0)

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
