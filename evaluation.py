import os
import argparse
import numpy as np


class angles:
    """
    Class for storing angles
    """
    yaw = 0
    pitch = 0
    roll = 0

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


parser = argparse.ArgumentParser(description='Evaluate head pose estimation against Gi4E dataset')
parser.add_argument('-p',
                    action='store',
                    dest='path',
                    help='path to input folder',
                    required=True,
                    type=str)
args = parser.parse_args()

files = list(filter(lambda x: x.endswith('.mp4'), os.listdir(args.path)))
files.sort()

files = files[:2]
all_angles = [angles(), angles(), angles()]
all_computed_data = [[], [], []]

for video_file in files:
    print("Evaluating file: ", video_file)
    current_angles = [angles(), angles(), angles()]
    data_file = video_file.replace(".mp4", "_groundtruth3D.txt")  # reference file
    with open("{}/{}".format(args.path, data_file)) as file:  # load data
        raw_data = file.read()
        rows = raw_data.split('\n')
        rows = rows[:-1]
        truth_data = [row.split('\t') for row in rows]
    for method in range(3):  # run detection for each method
        raw_data = os.popen(
            "python3 main.py -m 0 -i video -p {} -eval".format("{}/{}".format(args.path, video_file))).read()
        rows = raw_data.split('\n')
        rows = rows[:-1]
        data = [row.split('\t') for row in rows]
        all_computed_data[method] = all_computed_data[
                                        method] + data  # append to complete data for deviation computation
        data_length = len(data)
        for line_index in range(data_length): #sum differences of angles between ground truth and our result in current file
            current_angles[method].roll += abs(float(data[line_index][0]) - float(truth_data[line_index][3]))
            current_angles[method].yaw += abs(float(data[line_index][1]) - float(truth_data[line_index][4]))
            current_angles[method].pitch += abs(float(data[line_index][2]) - float(truth_data[line_index][5]))
        current_angles[method].div(data_length)

        all_angles[method].add(current_angles[method]) #sum of differences across all files

        print("Average error for method ", method, ":\n\tRoll: ", current_angles[method].roll, "\n\tYaw: ",
              current_angles[method].yaw, "\n\tPitch: ",
              current_angles[method].pitch)

print("------------------------------------------------------------------------------------------------")
all_computed_data = np.array(all_computed_data, dtype=float)
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
