import cv2
import math
import numpy as np


def face_orientation(img_shape, image_points):
    """
    Computes face rotation from unrotated default 3D model
    :param img_shape: size of image with face
    :param image_points: landmarks on face in order (nose, chin, left eye corner, right eye right, left mouth corner, right mouth corner)
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

    #camera aproximation
    center = (img_shape[1] / 2, img_shape[0] / 2)
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
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return axis_points, (str(int(roll)), str(int(pitch)), str(int(yaw)))
