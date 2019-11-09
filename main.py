import argparse

import cv2
import dlib

from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_multi import MultiHeadPoseEstimator
from head_pose_tracker import HeadPoseTracker
from image_utils import draw_and_show_landmarks_and_head_pose

landmark_model_path = "models/shape_predictor_68_face_landmarks.dat"


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
parser.add_argument('-eval',
                    action='store_true',
                    dest='evaluate',
                    help='Print result of roll, yaw pitch to stdout for evaluation purpose',
                    default=False)
args = parser.parse_args()

if args.input_type != 'camera' and args.path is None:
    print('You need to provide file input. Missing -p.')
    exit(0)


def get_estimator(method, detector, predictor):
    """
    Factory method for head pose estimators
    :param method: method to estimate with {0, 1, 2} - {3D model, tracking, geometry}
    :param detector: face detector
    :param predictor: landmark detector
    """
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
    """
    Head pose estimation for video input {video file, camera}
    :param method: method to estimate with {0, 1, 2} - {3D model, tracking, geometry}
    :param file_path: path to source video file, int for chosen camera input
    :return:
    """
    cap = cv2.VideoCapture(file_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)

    head_pose_estimator = get_estimator(method, detector, predictor)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            success, yaw, pitch, roll = head_pose_estimator.pose_for_image(frame)
            if success:
                if args.evaluate:
                    print(roll, "\t", yaw, "\t", pitch)
                else:
                    draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, frame, yaw, pitch, roll,
                                                          "Using {}.".format(head_pose_estimator.get_name()))
            else:
                draw_and_show_landmarks_and_head_pose([], frame, info_text="Using {}. Face not found.".format(
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
    """
    Head pose estimation for single image
    :param method: method to estimate with {0, 1, 2} - {3D model, tracking, geometry}
    :param file_path: path to source image
    """
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
    if success and args.evaluate:
        print(roll, "\t", yaw, "\t", pitch)
    elif success:
        draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, img, yaw, pitch, roll)
    else:
        draw_and_show_landmarks_and_head_pose([], img, info_text='Face not found.')
    cv2.waitKey()


if args.input_type == 'camera':
    video_estimation(args.method)
elif args.input_type == 'video':
    video_estimation(args.method, args.path)
elif args.input_type == 'image':
    pose_estimation(args.method, args.path)
