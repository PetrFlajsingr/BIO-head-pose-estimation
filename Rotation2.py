import numpy as np


def face_orientation2(img_shape, landmarks, nose_length):
    """

    :param img_shape: img size
    :param landmarks: landmarks in order left_eye_left_corner, left_eye_right_corner, right_eye_left_corner, right_eye_right_corner
    :param nose_length: lenght of nose bridge
    :return: yaw, pitch, roll
    """
    center = (img_shape[1] / 2, img_shape[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)

    roll = np.arctan((landmarks[0][1] - landmarks[1][1]) / (landmarks[0][0] - landmarks[3][0]))

    I1 = (landmarks[0][0] - landmarks[1][0]) * (landmarks[2][0] - landmarks[3][0]) / (
            landmarks[0][0] - landmarks[2][0]) * (landmarks[1][0] - landmarks[3][0])
    Q = (1 / np.sqrt(I1)) - 1

    u_delta = (landmarks[0][0] - landmarks[1][0])
    v_delta = (landmarks[2][0] - landmarks[3][0])
    A = (u_delta / v_delta) + 1
    B = ((2 / Q) + 2) * ((u_delta / v_delta) - 1)
    C = ((2 / Q) + 1) * ((u_delta / v_delta) + 1)
    S = (-B + np.sqrt(B * B - 4 * A * C) / 2 * A)

    M = (v_delta * landmarks[0][0]) / u_delta * landmarks[3][0]
    u1v1_minus = (landmarks[1][0] - landmarks[2][0])
    u1 = (u_delta * v_delta * M * u1v1_minus) - (
            M * M * (landmarks[0][0] - landmarks[3][0]) * u1v1_minus * u1v1_minus) / (
                 v_delta * ((M * u1v1_minus) - u_delta))
    yaw = np.arctan(focal_length / (S - 1) * u1)

    p0 = (48 * (landmarks[0][0] - landmarks[3][0])) / 86
    p1 = nose_length
    p1p1 = p1 * p1
    p0p0 = p0 * p0
    focal_lengthfocal_length = focal_length * focal_length
    E = (focal_length / p0 * ((p1p1) + (focal_length * focal_length))) * (
                p1p1 + np.sqrt(p0p0 * p1p1 - focal_lengthfocal_length * p1p1 + focal_lengthfocal_length * p0p0))
    pitch = np.arctan(E)

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
