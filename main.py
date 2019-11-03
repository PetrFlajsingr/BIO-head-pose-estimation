import argparse

import cv2
import numpy as np
import dlib
from rotation import face_orientation

from landmark_recognition import landmarks_for_face


def method0(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        nose = landmarks[30]
        chin = landmarks[8]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]

        selected_landmarks = np.array(
            [nose, chin, left_eye_left_corner, right_eye_right_corner, left_mouth, right_mouth], dtype="double")

        axis_points, rotate_degree = face_orientation(img.shape, selected_landmarks)

        cv2.line(img, nose, tuple(axis_points[1].ravel()), (0, 255, 0), 3)  # GREEN
        cv2.line(img, nose, tuple(axis_points[0].ravel()), (255, 0,), 3)  # BLUE
        cv2.line(img, nose, tuple(axis_points[2].ravel()), (0, 0, 255), 3)  # RED

        print("Roll: " + rotate_degree[0] + "\nPitch: " + rotate_degree[1] + "\nYaw: " + rotate_degree[2])

        img = cv2.resize(img, (384, 512))
        img_landmarks = cv2.resize(img_landmarks, (384, 512))

        cv2.imshow("Frame", img)
        cv2.imshow("Frame landmarks", img_landmarks)
    else:
        print('Face not found.')


def method1_init(img, detector, predictor):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = landmarks_for_face(detector, predictor, gray)
    result = {}

    eye_distance_tohead_depth_ratio = 1.6
    if landmarks is not None and len(landmarks) != 0:
        result['left_eye'] = landmarks[36]
        result['right_eye'] = landmarks[45]
        result['nose'] = landmarks[30]
        eye_distance = np.sqrt(
            np.power(landmarks[36][0] - landmarks[45][0], 2)
            + np.power(landmarks[36][1] - landmarks[45][1], 2))
        result['sphere_radius'] = eye_distance * eye_distance_tohead_depth_ratio / 2
        result['sphere_circumference'] = np.pi * 2 * result['sphere_radius']
        return result
    else:
        print('Face not found')
        return None


def method1(img, detector, predictor, last, yaw, pitch):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_landmarks = img.copy()
    landmarks = landmarks_for_face(detector, predictor, gray)

    if landmarks is not None and len(landmarks) != 0:
        left_eye_left_corner = landmarks[36]
        right_eye_right_corner = landmarks[45]
        nose = landmarks[30]
        roll = 0
        if left_eye_left_corner[0] != right_eye_right_corner[0] and left_eye_left_corner[1] != right_eye_right_corner[1]:
            roll = np.rad2deg(np.arctan(
                (left_eye_left_corner[0] - right_eye_right_corner[0])
                / (left_eye_left_corner[1] - right_eye_right_corner[1])))
        yaw += (last['nose'][0] - nose[0]) / last['sphere_circumference'] * 360
        pitch += (nose[1] - last['nose'][1]) / last['sphere_circumference'] * 360
        last['left_eye'] = left_eye_left_corner
        last['right_eye'] = right_eye_right_corner
        last['nose'] = nose

        x1 = abs(int(500 * (np.cos(yaw) * np.cos(roll))))
        y1 = abs(int(500 * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))))
        x2 = abs(int(500 * (-np.cos(yaw) * np.sin(roll))))
        y2 = abs(int(500 * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))))
        x3 = abs(int(500 * (np.sin(yaw))))
        y3 = abs(int(500 * (-np.cos(yaw) * np.sin(pitch))))
        cv2.line(img, (500, 500), (x1, y1), (0, 255, 0), 3)  # GREEN
        cv2.line(img, (500, 500), (x2, y2), (255, 0,), 3)  # BLUE
        cv2.line(img, (500, 500), (x3, y3), (0, 0, 255), 3)  # RED
        print("1: Roll:", roll, "\nPitch", pitch, "\nYaw:", yaw)

        img = cv2.resize(img, (384, 512))
        img_landmarks = cv2.resize(img_landmarks, (384, 512))

        cv2.imshow("Frame", img)
        cv2.imshow("Frame landmarks", img_landmarks)

    return last, yaw, pitch



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


def video_estimation(method, file_path=None):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    if file_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(file_path)

    method1_last = None
    yaw = 0.0
    pitch = 0.0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if method == 0:
                method0(frame, detector, predictor)
            if method == 1:
                if method1_last is None:
                    method1_last = method1_init(frame, detector, predictor)
                else:
                    method1_last, yaw, pitch = method1(frame, detector, predictor, method1_last, yaw, pitch)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()


def pose_estimation(method, file_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    img = cv2.imread(file_path)
    if method == 0:
        method0(img, detector, predictor)
    elif method == 1:
        print('Method 1 is unusable for images.')
        exit(0)
    cv2.waitKey()





if args.input_type == 'camera':
    video_estimation(args.method)
elif args.input_type == 'video':
    video_estimation(args.method, args.path)
elif args.input_type == 'image':
    pose_estimation(args.method, args.path)

