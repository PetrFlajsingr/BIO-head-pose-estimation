import cv2
import numpy as np

from geom_utils import euclidean_distance, per_elem_diff
from landmark_constants import *
from landmark_recognition import landmarks_for_face


class HeadPoseGeometry:
    def __init__(self, detector, predictor):
        self.__detector = detector
        self.__predictor = predictor
        self.landmarks = []

    def get_name(self):
        return "Using geometry."

    def pose_for_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.landmarks = landmarks_for_face(self.__detector, self.__predictor, gray)
        if self.landmarks is not None and len(self.landmarks) != 0:
            roll = np.rad2deg(self.__calculate_roll())
            yaw = np.rad2deg(self.__calculate_yaw())
            pitch = np.rad2deg(self.__calculate_pitch())
            return True, yaw, pitch, roll
        else:
            return False, 0.0, 0.0, 0.0

    def __calculate_roll(self):
        return np.arctan((self.landmarks[right_eye_right_corner][y_coord] - self.landmarks[left_eye_left_corner][y_coord])
                         / (self.landmarks[right_eye_right_corner][x_coord] - self.landmarks[left_eye_left_corner][x_coord]))

    def __calculate_yaw(self):
        k = 0.703
        k1 = (self.landmarks[right_ear][y_coord] - self.landmarks[left_ear][y_coord]) \
             / (self.landmarks[right_ear][x_coord] - self.landmarks[left_ear][x_coord])
        q1 = self.landmarks[left_eye_left_corner][y_coord] - k1 * self.landmarks[left_eye_left_corner][x_coord]

        if k1 == 0:
            k1 += 0.0000000001
        k2 = -1 / k1
        q2 = self.landmarks[nostrils_center][y_coord] - k2 * self.landmarks[nostrils_center][x_coord]

        x_p = (q2 - q1) / (k1 - k2)
        y_p = k1 * x_p + q1

        l = euclidean_distance(self.landmarks[right_ear], self.landmarks[left_ear]) * k

        x_s = (self.landmarks[right_ear][x_coord] + self.landmarks[left_ear][x_coord]) / 2
        y_s = k1 * x_s + q1

        yaw = np.arcsin(euclidean_distance((x_p, y_p), (x_s, y_s)) / l)
        if euclidean_distance(self.landmarks[left_ear], self.landmarks[nostrils_center]) \
                > euclidean_distance(self.landmarks[right_ear], self.landmarks[nostrils_center]):
            yaw = -yaw
        return yaw

    def __calculate_pitch(self):
        x_eye_corner_dist, y_eye_corner_dist = \
            per_elem_diff(self.landmarks[right_eye_right_corner], self.landmarks[left_eye_left_corner])
        x_eye_corner_dist /= 2
        y_eye_corner_dist /= 2

        x_mouth_corner_dist, y_mouth_corner_dist = \
            per_elem_diff(self.landmarks[mouth_right_corner], self.landmarks[mouth_left_corner])
        x_mouth_corner_dist /= 2
        y_mouth_corner_dist /= 2

        k = (y_eye_corner_dist - y_mouth_corner_dist) / (x_eye_corner_dist - x_mouth_corner_dist)
        if k == 0:
            k += 0.00000001
        q = y_mouth_corner_dist - k * x_mouth_corner_dist

        k2 = -1 / k
        q2 = self.landmarks[nose_bridge_tip][y_coord] - k2 * self.landmarks[nose_bridge_tip][x_coord]

        x_p = (q2 - q) / (k - k2)
        y_p = k * x_p + q

        if y_eye_corner_dist - y_mouth_corner_dist == 0:
            y_eye_corner_dist += 0.00000001
        return np.arctan(
            ((y_p - y_mouth_corner_dist) / (y_eye_corner_dist - y_mouth_corner_dist) - (3.312 / 7.2)) / (3.75 / 7.2)
        )
