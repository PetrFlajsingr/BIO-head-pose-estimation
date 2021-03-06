import numpy as np

from landmark_constants import *


class HeadPoseTracker:
    """
    Head pose estimation using tracking.
    Tracking is initialised when face is first detected.
    Euler angles are computed from person's movement.
    """

    def __init__(self):
        self.__is_initialised = False
        self.yaw = 0.0
        self.pitch = 15.0
        self.roll = 0.0
        self.__init_locations = {}
        self.landmarks = []

    def get_name(self):
        return "tracking"

    def __repr__(self):
        return 'Yaw: {}, Pitch: {}, Roll: {}'.format(self.yaw, self.pitch, self.roll)

    def pose_for_landmarks(self, image, landmarks):
        self.landmarks = landmarks
        if not self.__is_initialised:
            self.__init_tracking()
        if self.__is_initialised:
            self.__detect_pose()
        return self.yaw, self.pitch, self.roll

    def reset(self):
        self.__is_initialised = False
        self.yaw = 0.0
        self.pitch = 15.0
        self.roll = 0.0

    def __init_tracking(self):
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
            self.__init_locations['sphere_radius'] = eye_distance * eye_distance_tohead_depth_ratio
            self.__init_locations['sphere_circumference'] = np.pi * 2 * self.__init_locations['sphere_radius']
            self.__is_initialised = True
        else:
            return None

    def __detect_pose(self):
        if self.landmarks[left_eye_left_corner][x_coord] != self.landmarks[right_eye_right_corner][x_coord] \
                and self.landmarks[left_eye_left_corner][y_coord] != self.landmarks[right_eye_right_corner][y_coord]:
            self.roll = np.rad2deg(np.arctan(
                (self.landmarks[left_eye_left_corner][x_coord] - self.landmarks[right_eye_right_corner][x_coord])
                / (self.landmarks[left_eye_left_corner][y_coord] - self.landmarks[right_eye_right_corner][y_coord])))
            is_negative = self.roll < 0
            self.roll = 90 - abs(self.roll)
            if is_negative:
                self.roll = -self.roll
        x_diff = (self.__init_locations['nose'][x_coord] - self.landmarks[nose_bridge_tip][x_coord])
        self.yaw += np.rad2deg(np.arctan(x_diff
                                         / self.__init_locations['sphere_radius']))
        y_diff = (self.landmarks[nose_bridge_tip][y_coord] - self.__init_locations['nose'][y_coord])
        self.pitch -= np.rad2deg(np.arctan(y_diff
                                           / self.__init_locations['sphere_radius']))
        self.__init_locations['left_eye'] = self.landmarks[left_eye_left_corner]
        self.__init_locations['right_eye'] = self.landmarks[right_eye_right_corner]
        self.__init_locations['nose'] = self.landmarks[nose_bridge_tip]
