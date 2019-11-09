import os
import argparse

import dlib
import numpy as np
import cv2

from eval_plot import create_plot
from head_pose_geometry import HeadPoseGeometry
from head_pose_model import HeadPoseModel
from head_pose_tracker import HeadPoseTracker
from landmark_recognition import landmarks_for_face


class EulerAngles:
    def __init__(self, yaw=0, pitch=0, roll=0):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

    def add(self, other):
        self.yaw += other.yaw
        self.pitch += other.pitch
        self.roll += other.roll

    def div(self, other: int):
        self.yaw /= other
        self.pitch /= other
        self.roll /= other

    def as_nparray(self):
        return np.array([self.roll, self.yaw, self.pitch])

landmark_model_path = "models/shape_predictor_68_face_landmarks.dat"

def angles_for_video(estimator, path):
    cap = cv2.VideoCapture(path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_model_path)
    result = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            landmarks = landmarks_for_face(detector, predictor, frame)
            if len(landmarks) == 0 or landmarks is None:
                yaw, pitch, roll = -180, -180, -180
            else:
                yaw, pitch, roll = estimator.pose_for_landmarks(image=frame, landmarks=landmarks)
            result.append(EulerAngles(-yaw, pitch, roll))
        else:
            break
    cap.release()
    return result



parser = argparse.ArgumentParser(description='Evaluate head pose estimation against Gi4E dataset')
parser.add_argument('-p',
                    action='store',
                    dest='path',
                    help='path to input folder',
                    required=True,
                    type=str)
parser.add_argument('-graph',
                    action='store',
                    dest='graph_path',
                    help='path for graph output',
                    default='',
                    type=str)
args = parser.parse_args()

files = list(filter(lambda x: x.endswith('.mp4'), os.listdir(args.path)))
files.sort()

files = files[:2]
all_angles = [EulerAngles(), EulerAngles(), EulerAngles()]
all_computed_data = [[], [], []]

detectors = [HeadPoseModel(), HeadPoseTracker(), HeadPoseGeometry()]


if len(args.graph_path) != 0 and not os.path.exists(args.graph_path):
    os.mkdir(args.graph_path)

for video_file in files:
    print("Evaluating file: ", video_file)
    current_angles = [EulerAngles(), EulerAngles(), EulerAngles()]
    data_file = video_file.replace(".mp4", "_groundtruth3D.txt")  # reference file
    with open("{}/{}".format(args.path, data_file)) as file:  # load data
        raw_data = file.read()
        rows = raw_data.split('\n')
        rows = rows[:-1]
        truth_data = [row.split('\t') for row in rows]

    for method in range(3):  # run detection for each method
        detector = detectors[method]
        angles = angles_for_video(detector, "{}/{}".format(args.path, video_file))

        all_computed_data[method] = all_computed_data[
                                        method] + angles  # append to complete data for deviation computation
        data_length = len(angles)
        # sum differences of angles between ground truth and our result in current file
        for line_index in range(data_length):
            current_angles[method].roll += abs(float(angles[line_index].roll) - float(truth_data[line_index][3]))
            current_angles[method].yaw += abs(float(angles[line_index].yaw) - float(truth_data[line_index][4]))
            current_angles[method].pitch += abs(float(angles[line_index].pitch) - float(truth_data[line_index][5]))
        current_angles[method].div(data_length)

        all_angles[method].add(current_angles[method])  # sum of differences across all files

        print("Average error for method ", method, ":\n\tRoll: ", current_angles[method].roll, "\n\tYaw: ",
              current_angles[method].yaw, "\n\tPitch: ",
              current_angles[method].pitch)

        if len(args.graph_path) != 0:
            correct_yaw = [v[4] for v in truth_data]
            if method == 0:
                estimated_yaw = [-v.yaw for v in angles]
            else:
                estimated_yaw = [v.yaw for v in angles]
            create_plot(correct_yaw, estimated_yaw, args.graph_path, video_file[:-4] + "yaw" + detector.get_name())

            correct_pitch = [v[5] for v in truth_data]
            estimated_pitch = [v.pitch for v in angles]
            create_plot(correct_pitch, estimated_pitch, args.graph_path, video_file[:-4] + "pitch" + detector.get_name())

            correct_roll = [v[3] for v in truth_data]
            estimated_roll = [v.roll for v in angles]
            create_plot(correct_roll, estimated_roll, args.graph_path, video_file[:-4] + "roll" + detector.get_name())

print("------------------------------------------------------------------------------------------------")
all_computed_data = np.array([v.as_nparray for v in all_computed_data], dtype=float)
for method in range(3):
    all_angles[method].div(len(files))
    # compute deviation across all files
    all_computed_data[method] = np.subtract(all_computed_data[method], all_angles[method].as_nparray())
    all_computed_data[method] = np.power(all_computed_data[method], 2)
    deviation = np.sum(all_computed_data[method], axis=0)
    deviation /= (all_computed_data[method].shape[0] - 1)
    deviation = np.sqrt(deviation)
    print("METHOD ", method)
    print("Complete error:", method, ":\n\tRoll: ", all_angles[method].roll, "\n\tYaw: ",
          all_angles[method].yaw, "\n\tPitch: ",
          all_angles[method].pitch)
    print("Complete deviation:", method, ":\n\tRoll: ", deviation[0], "\n\tYaw: ",
          deviation[1], "\n\tPitch: ",
          deviation[2])
print("------------------------------------------------------------------------------------------------")
