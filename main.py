import argparse

import cv2
import dlib
import numpy as np
from scipy.spatial.transform import Rotation

from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker
from landmark_constants import *

unknwn = 'unknown'
landmark_model_path = "models/shape_predictor_68_face_landmarks.dat"


class MultiHeadPoseEstimator:
    def __init__(self, detector, predictor):
        self.__geom = HeadPoseGeometry(detector, predictor)
        self.__track = HeadPoseTracker(detector, predictor)
        self.__model = HeadPoseModel(detector, predictor)
        self.landmarks = []

    def get_name(self):
        return "multi"

    def pose_for_image(self, image):
        geom_res = self.__geom.pose_for_image(image)
        track_res = self.__track.pose_for_image(image)
        model_res = self.__model.pose_for_image(image)
        self.landmarks = self.__geom.landmarks
        return geom_res[0], self.__format_group(geom_res[1], track_res[1], model_res[1]), \
               self.__format_group(geom_res[2], track_res[2], model_res[2]), \
               self.__format_group(geom_res[3], track_res[3], model_res[3])

    def __format_group(self, arg1, arg2, arg3):
        return 'geom: {}, track: {}, model: {}'.format(arg1, arg2, arg3)


def draw_and_show_landmarks_and_head_pose(landmarks, image, yaw, pitch, roll, info_text=''):
    for pos in landmarks:
        cv2.circle(image, (int(pos[0]), int(pos[1])), 5, (0, 0, 255), -1)
    cv2.putText(image, 'Yaw: {}'.format(yaw), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Pitch: {}'.format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Roll: {}'.format(roll), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, info_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if not isinstance(yaw, str) and not isinstance(pitch, str) or not isinstance(roll, str):

        rotation = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=True)
        rotation_matrix = rotation.as_dcm()
        axis_points = np.float32([[50, 0, 0],
                                  [0, -50, 0],
                                  [0, 0, 50],
                                  ])

        position = np.array([60, image.shape[0] - 60, 0])

        axis = np.zeros((3, 3), dtype=float)
        axis[0] = np.dot(rotation_matrix, axis_points[0]) + position
        axis[1] = np.dot(rotation_matrix, axis_points[1]) + position
        axis[2] = np.dot(rotation_matrix, axis_points[2]) + position

        cv2.line(image, (int(position[0]), int(position[1])), (int(axis[1][0]), int(axis[1][1])), (0, 255, 0), 3)
        cv2.line(image, (int(position[0]), int(position[1])), (int(axis[0][0]), int(axis[0][1])), (255, 0, 0), 3)
        cv2.line(image, (int(position[0]), int(position[1])), (int(axis[2][0]), int(axis[2][1])), (0, 0, 255), 3)

    cv2.imshow('Out', image)


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
                    help='method used to estimate head pose: 0 - 3D model, 1 - tracking, 2 - geometry',
                    choices=[0, 1, 2],
                    type=int,
                    default=0)
args = parser.parse_args()

if args.input_type != 'camera' and args.path is None:
    print('You need to provide file input. Missing -p.')
    exit(0)


def get_estimator(method, detector, predictor):
    if method == 0:
        head_pose_estimator = HeadPoseModel(detector, predictor)
    elif method == 1:
        head_pose_estimator = HeadPoseTracker(detector, predictor)
    elif method == 2:
        head_pose_estimator = HeadPoseGeometry(detector, predictor)
    elif method == 3:
        head_pose_estimator = MultiHeadPoseEstimator(detector, predictor)
    else:
        raise Exception("Invalid method:{}".format(method))
    return head_pose_estimator


def video_estimation(method, file_path=0):
    cap = cv2.VideoCapture(file_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)

    head_pose_estimator = get_estimator(method, detector, predictor)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            success, yaw, pitch, roll = head_pose_estimator.pose_for_image(frame)
            if success:
                draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, frame, yaw, pitch, roll,
                                                      "Using {}.".format(head_pose_estimator.get_name()))
            else:
                draw_and_show_landmarks_and_head_pose([], frame, unknwn, unknwn, unknwn,
                                                      "Using {}. Face not found.".format(
                                                          head_pose_estimator.get_name()))
            input = cv2.waitKey(20) & 0xFF
            if input == ord('q'):
                break
            elif input == ord('n'):
                method = (method + 1) % 4
                head_pose_estimator = get_estimator(method, detector, predictor)
        else:
            break
    cap.release()


def pose_estimation(method, file_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)
    img = cv2.imread(file_path)
    if method == 0:
        head_pose_estimator = HeadPoseModel(detector, predictor)
    elif method == 1:
        print('Method 1 is unusable for images.')
        return
    elif method == 2:
        head_pose_estimator = HeadPoseGeometry(detector, predictor)
    else:
        raise Exception("Invalid method:{}".format(method))

    success, yaw, pitch, roll = head_pose_estimator.pose_for_image(img)
    if success:
        draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, img, yaw, pitch, roll)
    else:
        draw_and_show_landmarks_and_head_pose([], img, unknwn, unknwn, unknwn, 'Face not found.')
    cv2.waitKey()


if args.input_type == 'camera':
    video_estimation(args.method)
elif args.input_type == 'video':
    video_estimation(args.method, args.path)
elif args.input_type == 'image':
    pose_estimation(args.method, args.path)
