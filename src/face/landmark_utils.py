import numpy as np
from scipy.spatial.transform import Rotation as Rtool

FACE_LEFT_IRIS = [474, 475, 476, 477]
FACE_RIGHT_IRIS = [469, 470, 471, 472]

FACE_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398, 362]
FACE_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 33, 246, 161, 160, 159, 158, 157, 173, 133]

LIP_TOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_BOTTOM = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def process_face_landmark(landmark_3d, width, height):
    x_axis = landmark_3d[280] - landmark_3d[50]
    x_axis += landmark_3d[352] - landmark_3d[123]
    x_axis += landmark_3d[280] - landmark_3d[50]
    x_axis += landmark_3d[376] - landmark_3d[147]
    x_axis += landmark_3d[416] - landmark_3d[192]
    x_axis += landmark_3d[298] - landmark_3d[68]
    x_axis += landmark_3d[301] - landmark_3d[71]

    y_axis = landmark_3d[10] - landmark_3d[152]
    y_axis += landmark_3d[151] - landmark_3d[152]
    y_axis += landmark_3d[8] - landmark_3d[17]
    y_axis += landmark_3d[5] - landmark_3d[200]
    y_axis += landmark_3d[6] - landmark_3d[199]
    y_axis += landmark_3d[8] - landmark_3d[18]
    y_axis += landmark_3d[9] - landmark_3d[175]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = Rtool.from_matrix(np.transpose(np.array([x_axis, y_axis, z_axis]))) * Rtool.from_rotvec([-0.25, 0, 0])
    rotvec = matrix.as_rotvec()
    return rotvec

def iris_position(landmark_3d):
    left_pos = [0, 0, 0]
    for idx in FACE_LEFT_IRIS:
        left_pos += landmark_3d[idx]
    left_pos = [abs(item/len(FACE_LEFT_IRIS)) for item in left_pos]

    right_pos = [0, 0, 0]
    for idx in FACE_RIGHT_IRIS:
        right_pos += landmark_3d[idx]
    right_pos = [abs(item/len(FACE_RIGHT_IRIS)) for item in right_pos]
    return (left_pos[0], left_pos[1]), (right_pos[0], right_pos[1])

def eye_position(landmark_3d):
    left_pos = [0, 0, 0]
    for idx in FACE_LEFT_EYE:
        left_pos += landmark_3d[idx]
    left_pos = [abs(item/len(FACE_LEFT_EYE)) for item in left_pos]

    right_pos = [0, 0, 0]
    for idx in FACE_RIGHT_EYE:
        right_pos += landmark_3d[idx]
    right_pos = [abs(item/len(FACE_RIGHT_EYE)) for item in right_pos]
    return (left_pos[0], left_pos[1]), (right_pos[0], right_pos[1])

def mouth_handler(landmark_3d):
    LIP_TOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LIP_BOTTOM = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    lip_t = 0
    lip_b = 0
    for idx in LIP_TOP:
        lip_t += landmark_3d[idx, 1]
    for idx in LIP_BOTTOM:
        lip_b += landmark_3d[idx, 1]
    open_pix = lip_t / len(LIP_TOP) - lip_b / len(LIP_BOTTOM)
    if open_pix < 1:
        open_pix = 0
    return open_pix

def eye_handler(landmark_3d, eye_idxs):
    eye_t = 0
    eye_b = 0
    for idx in eye_idxs[:len(eye_idxs)//2]:
        eye_t += landmark_3d[idx, 1]

    for idx in eye_idxs[len(eye_idxs)//2+1:]:
        eye_b += landmark_3d[idx, 1]
    open_pix = (eye_b - eye_t) / len(LIP_BOTTOM) * 2
    if open_pix < 0:
        open_pix = 0
    return open_pix

def landmark_handler(landmark_3d, width, height):
    infos = {}
    rotvec = process_face_landmark(landmark_3d, width, height)
    infos["face_horizontal_orientation"] = 90*rotvec[1]
    infos["face_vertical_orientation"]= -90*rotvec[0]

    left_iris, right_iris = iris_position(landmark_3d)
    left_eye, right_eye = eye_position(landmark_3d)

    infos["face_iris_vertical"] = left_iris[0] - left_eye[0] + right_iris[0] - right_eye[0]
    infos["face_iris_horizontal"] = left_iris[1] - left_eye[1] + right_iris[1] - right_eye[1]

    infos["mouth_open"] = mouth_handler(landmark_3d)
    infos["left_eye_open"] = eye_handler(landmark_3d, FACE_LEFT_EYE)
    infos["right_eye_open"] = eye_handler(landmark_3d, FACE_RIGHT_EYE)
    return infos