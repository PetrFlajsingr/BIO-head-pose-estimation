import cv2
import numpy as np

from image_utils import rotate_image


def landmarks_for_face(detector, predictor, image):
    angle = 0
    faces = []
    result = []
    while len(faces) == 0 and angle < 1:
        rotated = rotate_image(image, angle)
        faces = detector(rotated)
        for face in faces:
            landmarks = predictor(image, face)
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
            for n in range(0, 68):
                landmark = np.array([landmarks.part(n).x, landmarks.part(n).y, 1], dtype=np.float)
                transformed = np.dot(rot_mat, landmark.T)
                result.append(transformed)

        angle += 10
    return result
