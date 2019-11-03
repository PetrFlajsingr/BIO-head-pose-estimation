import cv2
import numpy as np
import dlib


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def landmarks_for_face(detector, predictor, image):
    angle = 0
    faces = []
    result = []
    while len(faces) == 0 and angle < 360:
        rotated = rotate_image(image, angle)
        faces = detector(rotated)
        for face in faces:
            landmarks = predictor(gray, face)
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, -angle-10, 1.0)
            for n in range(0, 68):
                landmark = np.array([landmarks.part(n).x, landmarks.part(n).y, 1], dtype=np.float)
                transformed = np.dot(rot_mat, landmark.T)
                print(transformed)
                print(landmarks.part(n).x, landmarks.part(n).y)
                print((int(transformed[0]), int(transformed[1])))
                result.append((int(transformed[0]), int(transformed[1])))


        angle += 10
    return result


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

img = cv2.imread('data/4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
landmarks = landmarks_for_face(detector, predictor, gray)

if landmarks is not None:
    print('Face found.')
    for (x, y) in landmarks:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

    cv2.imshow("Frame", img)
    cv2.waitKey()
else:
    print('Face not found.')
