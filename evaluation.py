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


def angle_correction(angle, a, b):
    return np.arctan(a / b) + angle


def print_and_write(file, text):
    """
    Prints text that is than written into file
    :param file: output file to write to
    :param text: string to write and print
    :return:
    """
    print(text)
    file.write(text)
    file.write("\n")


parser = argparse.ArgumentParser(description='Evaluate head pose estimation against Gi4E dataset')
parser.add_argument('-p',
                    action='store',
                    dest='path',
                    help='path to input folder',
                    required=True,
                    type=str)
parser.add_argument('-o',
                    action='store',
                    dest='out_path',
                    help='path for output',
                    required=True,
                    type=str)
args = parser.parse_args()

files = list(filter(lambda x: x.endswith('.mp4'), os.listdir(args.path)))
files.sort()



all_angles = [EulerAngles(), EulerAngles(), EulerAngles()]
all_differences = [[], [], []]

detectors = [HeadPoseModel(), HeadPoseTracker(), HeadPoseGeometry()]

if len(args.out_path) != 0 and not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

with open('{}/{}.txt'.format(args.out_path, 'stats'), 'w') as out_file:
    for video_file in files:
        print("Evaluating file: ", video_file)
        current_angles = [EulerAngles(), EulerAngles(), EulerAngles()]
        data_file = video_file.replace(".mp4", "_groundtruth3D.txt")  # reference file
        with open("{}/{}".format(args.path, data_file)) as file:  # load data
            raw_data = file.read()
            rows = raw_data.split('\n')
            rows = rows[:-1]
            truth_data = [row.split('\t') for row in rows]

        detectors[1].reset()
        detectors[1].yaw = float(truth_data[0][4])
        detectors[1].pitch = float(truth_data[0][5])
        detectors[1].roll = float(truth_data[0][3])
        for method in range(3):  # run detection for each method
            detector = detectors[method]
            angles = angles_for_video(detector, "{}/{}".format(args.path, video_file))

            data_length = len(angles)
            # sum differences of angles between ground truth and our result in current file
            for line_index in range(data_length):
                if method != 0:
                    angles[line_index].yaw = angle_correction(angles[line_index].yaw, float(truth_data[line_index][0]),
                                                              float(truth_data[line_index][2]))
                    angles[line_index].pitch = angle_correction(angles[line_index].pitch,
                                                                float(truth_data[line_index][0]),
                                                                float(truth_data[line_index][1]))

                roll_diff = float(angles[line_index].roll) - float(truth_data[line_index][3])
                yaw_diff = float(angles[line_index].yaw) - float(truth_data[line_index][4])
                pitch_diff = float(angles[line_index].pitch) - float(truth_data[line_index][5])
                current_angles[method].roll += abs(roll_diff)
                current_angles[method].yaw += abs(yaw_diff)
                current_angles[method].pitch += abs(pitch_diff)
                all_differences[method].append(
                    [roll_diff, yaw_diff, pitch_diff])  # save values to all differences for deviation computaion

            current_angles[method].div(data_length)

            all_angles[method].add(current_angles[method])  # sum of partial averages of error across all files
            print_and_write(out_file,
                            "Average error for method {}:\n\tRoll: {} \n\tYaw: {} \n\tPitch: {}".format(method,
                                                                                                        current_angles[
                                                                                                            method].roll,
                                                                                                        current_angles[
                                                                                                            method].yaw,
                                                                                                        current_angles[
                                                                                                            method].pitch))

            correct_yaw = [v[4] for v in truth_data]
            if method == 0:
                estimated_yaw = [-v.yaw for v in angles]
            elif method == 2:
                estimated_yaw = [-v.yaw + 3 for v in angles]
            else:
                estimated_yaw = [-v.yaw for v in angles]
            create_plot(correct_yaw, estimated_yaw, args.out_path,
                        "{}_{}_{}".format(detector.get_name(), "yaw", video_file[:-4]))

            correct_pitch = [v[5] for v in truth_data]

            if method == 0:
                estimated_pitch = [-v.pitch + 9 for v in angles]
            elif method == 2:
                estimated_pitch = [-v.pitch + 15 for v in angles]
            else:
                estimated_pitch = [-v.pitch for v in angles]
            create_plot(correct_pitch, estimated_pitch, args.out_path,
                        "{}_{}_{}".format(detector.get_name(), "pitch", video_file[:-4]))

            correct_roll = [v[3] for v in truth_data]
            estimated_roll = [v.roll for v in angles]
            create_plot(correct_roll, estimated_roll, args.out_path,
                        "{}_{}_{}".format(detector.get_name(), "roll", video_file[:-4]))

    print_and_write(out_file,
                    "------------------------------------------------------------------------------------------------")
    print_and_write(out_file, "*RESULT*")
    all_differences = np.array(all_differences, dtype=float)
    for method in range(3):
        all_angles[method].div(len(files))  # div by file count to get complete avg from partial avgs
        record_count = all_differences[method].shape[0]
        avg = np.sum(all_differences[method], axis=0) / record_count  # non-absolute avg for deviation
        # compute deviation across all files
        diff = np.subtract(all_differences[method], avg)  # (xi - avg)
        sq_diff = np.power(diff, 2)  # (xi - avg)^2
        sq_diff_sum = np.sum(sq_diff, axis=0)  # Σ(xi - avg)^2
        deviation = sq_diff_sum / record_count  # (Σ(xi - avg)^2) / N
        deviation = np.sqrt(deviation)  # sqrt((Σ(xi - avg)^2) / N) = sigma (standard deviation)
        print_and_write(out_file, "METHOD {}".format(method))
        print_and_write(out_file, "Complete error:{} :\n\tRoll: {} \n\tYaw: {} \n\tPitch: {}".format(method, all_angles[
            method].roll, all_angles[method].yaw, all_angles[method].pitch))
        print_and_write(out_file,
                        "Complete deviation:{} :\n\tRoll: {} \n\tYaw: {} \n\tPitch: {}".format(method, deviation[0],
                                                                                               deviation[1],
                                                                                               deviation[2]))
    print_and_write(out_file,
                    "------------------------------------------------------------------------------------------------")
