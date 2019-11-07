import cv2
import numpy as np

from landmark_constants import *
from landmark_recognition import landmarks_for_face


class HeadPoseTracker:
    def __init__(self, detector, predictor):
        self.__detector = detector
        self.__predictor = predictor
        self.__is_initialised = False
        self.__yaw = 0.0
        self.__pitch = 0.0
        self.__roll = 0.0
        self.__init_locations = {}
        self.landmarks = []

    def get_name(self):
        return "Using tracking."

    def __repr__(self):
        return 'Yaw: {}, Pitch: {}, Roll: {}'.format(self.__yaw, self.__pitch, self.__roll)

    def pose_for_image(self, image):
        if not self.__is_initialised:
            self.__init_tracking(image)
        success = False
        if self.__is_initialised:
            success = self.__detect_pose(image)
        return success, self.__yaw, self.__pitch, self.__roll

    def reset(self):
        self.__is_initialised = False

    def __init_tracking(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.landmarks = landmarks_for_face(self.__detector, self.__predictor, gray)

        eye_distance_tohead_depth_ratio = 1.6
        if self.landmarks is not None and len(self.landmarks) != 0:
            self.__init_locations['left_eye'] = self.landmarks[left_eye_left_corner]
            self.__init_locations['right_eye'] = self.landmarks[right_eye_right_corner]
            self.__init_locations['nose'] = self.landmarks[nose_bridge_tip]
            eye_distance = np.sqrt(
                np.power(self.landmarks[left_eye_left_corner][x_coord]
                         - self.landmarks[right_eye_right_corner][x_coord], 2)
                + np.power(self.landmarks[left_eye_left_corner][y_coord]
                           - self.landmarks[right_eye_right_corner][y_coord], 2))
            self.__init_locations['sphere_radius'] = eye_distance * eye_distance_tohead_depth_ratio / 2
            self.__init_locations['sphere_circumference'] = np.pi * 2 * self.__init_locations['sphere_radius']
            self.__is_initialised = True
        else:
            return None

    def __detect_pose(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.landmarks = landmarks_for_face(self.__detector, self.__predictor, gray)

        if self.landmarks is not None and len(self.landmarks) != 0:
            if self.landmarks[left_eye_left_corner][x_coord] != self.landmarks[right_eye_right_corner][x_coord] \
                    and self.landmarks[left_eye_left_corner][y_coord] != self.landmarks[right_eye_right_corner][y_coord]:
                self.__roll = np.rad2deg(np.arctan(
                    (self.landmarks[left_eye_left_corner][x_coord] - self.landmarks[right_eye_right_corner][x_coord])
                    / (self.landmarks[left_eye_left_corner][y_coord] - self.landmarks[right_eye_right_corner][y_coord])))
                is_negative = self.__roll < 0
                self.__roll = 90 - abs(self.__roll)
                if is_negative:
                    self.__roll = -self.__roll
            self.__yaw += (self.__init_locations['nose'][x_coord] - self.landmarks[nose_bridge_tip][x_coord]) \
                          / self.__init_locations['sphere_circumference'] * 360
            self.__pitch += (self.landmarks[nose_bridge_tip][y_coord] - self.__init_locations['nose'][y_coord]) \
                            / self.__init_locations['sphere_circumference'] * 360
            self.__init_locations['left_eye'] = self.landmarks[left_eye_left_corner]
            self.__init_locations['right_eye'] = self.landmarks[right_eye_right_corner]
            self.__init_locations['nose'] = self.landmarks[nose_bridge_tip]
            return True
        else:
            return False
