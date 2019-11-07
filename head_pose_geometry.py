import cv2
import numpy as np

from geom_utils import euclidean_distance, slope, epsilon
from landmark_constants import *
from landmark_recognition import landmarks_for_face


class HeadPoseGeometry:
    """
    Head pose estimation using geometry.
    """
    def __init__(self, detector, predictor):
        self.__detector = detector
        self.__predictor = predictor
        self.landmarks = []

    def get_name(self):
        return "geometry"

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
        k1 = slope(self.landmarks[right_ear], self.landmarks[left_ear])
        q1 = self.landmarks[left_eye_left_corner][y_coord] - k1 * self.landmarks[left_eye_left_corner][x_coord]

        if k1 == 0:
            k1 += epsilon
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
        eyes_mid_point = np.add(self.landmarks[right_eye_right_corner], self.landmarks[left_eye_left_corner]) / 2
        mouth_mid_point = np.add(self.landmarks[mouth_right_corner], self.landmarks[mouth_left_corner]) / 2

        if eyes_mid_point[x_coord] == mouth_mid_point[x_coord]:
            eyes_mid_point[x_coord] += epsilon

        k = slope(eyes_mid_point, mouth_mid_point)
        if k == 0:
            k += epsilon
        q = mouth_mid_point[y_coord] - k * mouth_mid_point[x_coord]

        k2 = -1 / k
        q2 = self.landmarks[nose_bridge_tip][y_coord] - k2 * self.landmarks[nose_bridge_tip][x_coord]

        if k == k2:
            k += epsilon
        x_p = (q2 - q) / (k - k2)
        y_p = k * x_p + q

        if eyes_mid_point[y_coord] - mouth_mid_point[y_coord] == 0:
            eyes_mid_point[y_coord] += epsilon
        return np.arctan(
            ((y_p - mouth_mid_point[y_coord]) / (eyes_mid_point[y_coord] - mouth_mid_point[y_coord])
             - (3.312 / 7.2)) / (3.75 / 7.2)
        )
