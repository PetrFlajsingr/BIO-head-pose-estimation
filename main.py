import argparse

import cv2
import numpy as np
import dlib

from geom_utils import euclidean_distance, per_elem_diff
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker
from Rotation2 import face_orientation2, Markers

from landmark_recognition import landmarks_for_face
from landmark_constants import *

unknown = 'unknown'
landmark_model_path = "models/shape_predictor_68_face_landmarks.dat"


def draw_and_show_landmarks_and_head_pose(landmarks, image, yaw, pitch, roll, info_text = ''):
    for pos in landmarks:
        cv2.circle(image, pos, 5, (0, 0, 255), -1)
    cv2.putText(image, 'Yaw: {}'.format(yaw), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Pitch: {}'.format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Roll: {}'.format(roll), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, info_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('Out', image)



def method2(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = landmarks_for_face(detector, predictor, gray)

    if landmarks is not None and len(landmarks) != 0:
        roll = np.arctan((landmarks[right_eye_right_corner][y_coord] - landmarks[left_eye_left_corner][y_coord])
                         / (landmarks[right_eye_right_corner][x_coord] - landmarks[left_eye_left_corner][x_coord]))

        k = 0.703
        k1 = (landmarks[right_ear][y_coord] - landmarks[left_ear][y_coord]) \
             / (landmarks[right_ear][x_coord] - landmarks[left_ear][x_coord])
        q1 = landmarks[left_eye_left_corner][y_coord] - k1 * landmarks[left_eye_left_corner][x_coord]

        if k1 == 0:
            k1 += 0.0000000001
        k2 = -1 / k1
        q2 = landmarks[nostrils_center][y_coord] - k2 * landmarks[nostrils_center][x_coord]

        x_p = (q2 - q1) / (k1 - k2)
        y_p = k1 * x_p + q1

        l = euclidean_distance(landmarks[right_ear], landmarks[left_ear]) * k

        x_s = (landmarks[right_ear][x_coord] + landmarks[left_ear][x_coord]) / 2
        y_s = k1 * x_s + q1

        yaw = np.arcsin(euclidean_distance((x_p, y_p), (x_s, y_s)) / l)
        if euclidean_distance(landmarks[left_ear], landmarks[nostrils_center]) \
                > euclidean_distance(landmarks[right_ear], landmarks[nostrils_center]):
            yaw = -yaw

        ###############################################################
        x_eye_corner_dist, y_eye_corner_dist = \
            per_elem_diff(landmarks[right_eye_right_corner], landmarks[left_eye_left_corner])
        x_eye_corner_dist /= 2
        y_eye_corner_dist /= 2

        x_mouth_corner_dist, y_mouth_corner_dist = \
            per_elem_diff(landmarks[mouth_right_corner], landmarks[mouth_left_corner])
        x_mouth_corner_dist /= 2
        y_mouth_corner_dist /= 2

        k = (y_eye_corner_dist - y_mouth_corner_dist) / (x_eye_corner_dist - x_mouth_corner_dist)
        if k == 0:
            k += 0.00000001
        q = y_mouth_corner_dist - k * x_mouth_corner_dist

        k2 = -1 / k
        q2 = landmarks[nose_bridge_tip][y_coord] - k2 * landmarks[nose_bridge_tip][x_coord]

        x_p = (q2 - q) / (k - k2)
        y_p = k * x_p + q

        if y_eye_corner_dist - y_mouth_corner_dist == 0:
            y_eye_corner_dist += 0.00000001
        pitch = np.arctan(
            ((y_p - y_mouth_corner_dist) / (y_eye_corner_dist - y_mouth_corner_dist) - (3.312 / 7.2)) / (3.75 / 7.2)
        )

        yaw = np.rad2deg(yaw)
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        print("Roll: " + str(roll) + "\nPitch: " + str(pitch) + "\nYaw: " + str(yaw))
        draw_and_show_landmarks_and_head_pose(landmarks, img, yaw, pitch, roll)
    else:
        draw_and_show_landmarks_and_head_pose([], img, unknown, unknown, unknown, 'face not detected')

    '''gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_landmarks = img.copy()
    landmarks = landmarks_for_face(detector, predictor, gray)

    if landmarks is not None and len(landmarks) != 0:
        print('Face found.')
        cnt = 0
        for (x, y) in landmarks:
            cv2.putText(img_landmarks, str(cnt), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cnt += 1

        left_eye_left_corner = landmarks[36]
        right_eye_right_corner = landmarks[45]
        left_eye_right_corner = landmarks[39]
        right_eye_left_corner = landmarks[42]

        selected_landmarks = Markers(
            left_eye_left_corner, left_eye_right_corner, right_eye_left_corner, right_eye_right_corner)

        observed_bridge_length = np.sqrt(
            np.power(landmarks[27][0] - landmarks[30][0], 2) + np.power(landmarks[27][1] - landmarks[30][1], 2))
        yaw, pitch, roll = face_orientation2(img.shape, selected_landmarks, observed_bridge_length)
        print("Roll: " + str(roll) + "\nPitch: " + str(pitch) + "\nYaw: " + str(yaw))'''


parser = argparse.ArgumentParser(description='Head pose estimation (yaw, pitch, roll)')
parser.add_argument('-i',
                    action='store',
                    dest='input_type',
                    help='type of input: image, video, camera',
                    choices=['image', 'video', 'camera'],
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


def video_estimation(method, file_path=0):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(file_path)

    if method == 0:
        head_pose_estimator = HeadPoseModel(detector, predictor)
    elif method == 1:
        head_pose_estimator = HeadPoseTracker(detector, predictor)
    success = False
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if method == 0 or method == 1:
                success, yaw, pitch, roll = head_pose_estimator.pose_for_image(frame)
                if success:
                    draw_and_show_landmarks_and_head_pose([], frame, yaw, pitch, roll)
                else:
                    draw_and_show_landmarks_and_head_pose([], frame, unknown, unknown, unknown, 'Face not found.')
            elif method == 2:
                method2(frame, detector, predictor)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def pose_estimation(method, file_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)
    img = cv2.imread(file_path)
    if method == 0:
        head_pose_estimator = HeadPoseModel(detector, predictor)
        success, yaw, pitch, roll = head_pose_estimator.pose_for_image(img)
        if success:
            draw_and_show_landmarks_and_head_pose([], img, yaw, pitch, roll)
        else:
            draw_and_show_landmarks_and_head_pose([], img, unknown, unknown, unknown, 'Face not found.')
    elif method == 1:
        print('Method 1 is unusable for images.')
        return
    elif method == 2:
        method2(img, detector, predictor)
    cv2.waitKey()


if args.input_type == 'camera':
    video_estimation(args.method)
elif args.input_type == 'video':
    video_estimation(args.method, args.path)
elif args.input_type == 'image':
    pose_estimation(args.method, args.path)

