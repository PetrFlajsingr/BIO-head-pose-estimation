import argparse
import time

import cv2
import dlib

from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker
from landmark_recognition import landmarks_for_face

landmark_model_path = "models/shape_predictor_68_face_landmarks.dat"

parser = argparse.ArgumentParser(description='Speed evaluation script')
parser.add_argument('-p',
                    action='store',
                    dest='path',
                    help='path to input file',
                    required=True)

args = parser.parse_args()


def eval(method, file_path, n):
    """
    Head pose estimation for single image
    :param method: method to estimate with {0, 1, 2} - {3D model, tracking, geometry}
    :param file_path: path to source image
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)
    img = cv2.imread(file_path)
    if method == 0:
        head_pose_estimator = HeadPoseModel()
    elif method == 1:
        head_pose_estimator = HeadPoseTracker()
    elif method == 2:
        head_pose_estimator = HeadPoseGeometry()
    else:
        raise Exception("Invalid method:{}".format(method))

    landmarks = landmarks_for_face(detector, predictor, img)
    start = time.time()
    for i in range(n):
        yaw, pitch, roll = head_pose_estimator.pose_for_landmarks(img, landmarks)
    end = time.time()
    print("Method:", method, "Took:", end - start, "Avg:", (end - start) / n)
    return end - start


N = 100000
eval(0, args.path, N)
eval(1, args.path, N)
eval(2, args.path, N)

