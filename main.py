import argparse

import cv2
import dlib

from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker

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
                draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, frame, yaw, pitch, roll, head_pose_estimator.get_name())
            else:
                draw_and_show_landmarks_and_head_pose([], frame, unknown, unknown, unknown, 'Face not found.')
            input = cv2.waitKey(20) & 0xFF
            if input == ord('q'):
                break
            elif input == ord('n'):
                method = (method + 1) % 3
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

    success, yaw, pitch, roll = head_pose_estimator.pose_for_image(img)
    if success:
        draw_and_show_landmarks_and_head_pose(head_pose_estimator.landmarks, img, yaw, pitch, roll)
    else:
        draw_and_show_landmarks_and_head_pose([], img, unknown, unknown, unknown, 'Face not found.')
    cv2.waitKey()


if args.input_type == 'camera':
    video_estimation(args.method)
elif args.input_type == 'video':
    video_estimation(args.method, args.path)
elif args.input_type == 'image':
    pose_estimation(args.method, args.path)

