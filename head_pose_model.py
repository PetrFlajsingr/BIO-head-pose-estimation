import cv2
import math
import numpy as np

from landmark_constants import *


class HeadPoseModel:
    """
    Head pose estimation using 3D model points and PnP.
    """
    def __init__(self):
        self.landmarks = []

    def get_name(self):
        return "3D model"

    def pose_for_landmarks(self, image, landmarks):
        self.landmarks = landmarks
        selected_landmarks = np.array(
            [self.landmarks[nose_bridge_tip],
             self.landmarks[center_chin],
             self.landmarks[left_eye_left_corner],
             self.landmarks[right_eye_right_corner],
             self.landmarks[mouth_left_corner],
             self.landmarks[mouth_right_corner]], dtype="double")

        axis_points, rotate_degree = self.__match_with_model(image.shape, selected_landmarks)

        return rotate_degree[2], rotate_degree[1], rotate_degree[0]

    @staticmethod
    def __match_with_model(image_shape, image_points):
        """
            Computes face rotation from unrotated default 3D model
            :param image_shape: size of image with face
            :param image_points: self.landmarks on face in order (nose, chin, left eye corner, right eye right, left mouth corner, right mouth corner)
            :return: rotation axis to draw to image and tuple of roll, pitch, yaw

            """

        model_3d = np.array([
            (0.0, 0.0, 0.0),  # nose tip
            (0.0, -330.0, -65.0),  # chin
            (-165.0, 170.0, -135.0),  # left eye corner
            (165.0, 170.0, -135.0),  # right eye corner
            (-150.0, -150.0, -125.0),  # left mouth corner
            (150.0, -150.0, -125.0)  # right mouth corner
        ])

        # camera aproximation
        center = (image_shape[1] / 2, image_shape[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        distorsion = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_3d, image_points, camera_matrix,
                                                                      distorsion, flags=cv2.SOLVEPNP_ITERATIVE)
        # axis to draw into image
        axis = np.float32([[500, 0, 0],
                           [0, 500, 0],
                           [0, 0, 500]])
        axis_points = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, distorsion)[0]

        rvecs = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvecs, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return axis_points, (roll, pitch, yaw)
