import dlib
import numpy as np

predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

def landmark(image, face):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face).parts()])
    return landmarks

def head_move(last_landmarks, landmarks):
    gap_total = 0
    for point1, point2 in zip(last_landmarks, landmarks):
        # print(point1, point2)
        gap = abs(point1[0, 0] - point2[0, 0]) + abs(point1[0, 1] - point2[0, 1])
        gap_total += gap
    gap_total /= 64
    return gap_total