import numpy as np

from geom_utils import euclidean_distance


class Markers:
    def __init__(self, left_left_eye, left_right_eye, right_left_eye, right_right_eye):
        self.left_left_eye = left_left_eye
        self.left_right_eye = left_right_eye
        self.right_left_eye = right_left_eye
        self.right_right_eye = right_right_eye


def face_orientation2(img_shape, landmarks: Markers, nose_length):
    """

    :param img_shape: img size
    :param landmarks: landmarks in order left_eye_left_corner, left_eye_right_corner, right_eye_left_corner, right_eye_right_corner
    :param nose_length: lenght of nose bridge
    :return: yaw, pitch, roll
    """
    center = (img_shape[1] / 2, img_shape[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)

    roll = np.arctan((landmarks.left_left_eye[1] - landmarks.right_right_eye[1])
                     / (landmarks.left_left_eye[0] - landmarks.right_right_eye[0]))

    # I1 = (landmarks[0][0] - landmarks[1][0]) * (landmarks[2][0] - landmarks[3][0]) / ((
    #        landmarks[0][0] - landmarks[2][0]) * (landmarks[1][0] - landmarks[3][0]))

    I1 = (euclidean_distance(landmarks.left_left_eye, landmarks.left_right_eye)
          * euclidean_distance(landmarks.right_left_eye, landmarks.right_right_eye)) \
         / (euclidean_distance(landmarks.left_left_eye, landmarks.right_left_eye)
            * euclidean_distance(landmarks.left_right_eye, landmarks.right_right_eye))
    Q = (1 / np.sqrt(I1)) - 1

    # u_delta = (landmarks[0][0] - landmarks[1][0])
    u_delta = euclidean_distance(landmarks.left_left_eye, landmarks.left_right_eye)
    # v_delta = (landmarks[2][0] - landmarks[3][0])
    v_delta = euclidean_distance(landmarks.right_right_eye, landmarks.right_left_eye)
    A = (u_delta / v_delta) + 1
    B = (2 / Q + 2) * (u_delta / v_delta - 1)
    C = (2 / Q + 1) * (u_delta / v_delta + 1)
    S = (-B + np.sqrt(pow(B, 2) - 4 * A * C)) / (2 * A)

    M = (v_delta * landmarks.right_left_eye[0]) / (u_delta * landmarks.left_right_eye[0])
    # u1v1_minus = (landmarks[1][0] - landmarks[2][0])
    u1v1_minus = euclidean_distance(landmarks.left_right_eye, landmarks.left_left_eye)
    u2v2_minus = euclidean_distance(landmarks.right_left_eye, landmarks.right_right_eye)
    u1 = ((u_delta * v_delta * M * u1v1_minus) - (
            pow(M, 2) * u2v2_minus * u1v1_minus * pow(u1v1_minus, 2))) / (
                 v_delta * (M * u1v1_minus - u_delta))
    yaw = np.arctan(focal_length / ((S - 1) * u1))

    eye_fissure_width = euclidean_distance(landmarks.right_left_eye, landmarks.right_right_eye)
    # p0 = (0.048 * (landmarks[0][0] - landmarks[3][0])) / 0.086
    p0 = 0.05 * eye_fissure_width / 0.031
    p1 = nose_length
    p1p1 = p1 * p1
    p0p0 = p0 * p0
    focal_length = 40  # 0.055
    focal_lengthfocal_length = focal_length * focal_length
    E = (focal_length / (p0 * (p1p1 + focal_lengthfocal_length))) \
        * (p1p1 - np.sqrt(p0p0 * p1p1 - focal_lengthfocal_length * p1p1 + focal_lengthfocal_length * p0p0))
    pitch = np.arctan(E)

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
