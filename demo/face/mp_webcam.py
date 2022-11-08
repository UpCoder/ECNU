import cv2
import time
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as Rtool
# from face_emotion import emotion_predict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

font = cv2.FONT_HERSHEY_SIMPLEX

FACE_LEFT_IRIS = [474, 475, 476, 477]
FACE_RIGHT_IRIS = [469, 470, 471, 472]

FACE_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398, 362]
FACE_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 33, 246, 161, 160, 159, 158, 157, 173, 133]


def iris_position(landmark_3d):
    left_pos = [0, 0, 0]
    for idx in FACE_LEFT_IRIS:
        left_pos += landmark_3d[idx]
    left_pos = [abs(int(item/len(FACE_LEFT_IRIS))) for item in left_pos]

    right_pos = [0, 0, 0]
    for idx in FACE_RIGHT_IRIS:
        right_pos += landmark_3d[idx]
    right_pos = [abs(int(item/len(FACE_RIGHT_IRIS))) for item in right_pos]
    return (left_pos[0], left_pos[1]), (right_pos[0], right_pos[1])

def eye_position(landmark_3d):
    left_pos = [0, 0, 0]
    for idx in FACE_LEFT_EYE:
        left_pos += landmark_3d[idx]
    left_pos = [abs(int(item/len(FACE_LEFT_EYE))) for item in left_pos]

    right_pos = [0, 0, 0]
    for idx in FACE_RIGHT_EYE:
        right_pos += landmark_3d[idx]
    right_pos = [abs(int(item/len(FACE_RIGHT_EYE))) for item in right_pos]
    return (left_pos[0], left_pos[1]), (right_pos[0], right_pos[1])

def process_landmark(face_landmarks, width, height):
    landmark1 = face_landmarks.landmark ## 478 is known number
    np_array = np.zeros((len(landmark1), 3), np.float64)
    for i in range(len(landmark1)):
        np_array[i, 0] = landmark1[i].x
        np_array[i, 1] = landmark1[i].y
        np_array[i, 2] = landmark1[i].z
    np_array[:, 0] = np_array[:, 0] * width
    np_array[:, 1] = - np_array[:, 1] * height
    np_array[:, 2] = - np_array[:, 2] * width
    return np_array

def process_face_landmark(landmark_3d, width, height):
    # landmark_3d = process_landmark(face_landmarks, width, height)
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

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(701)

cap.set(3, 1280)
cap.set(4, 960)

iris_eye_pos_list = []
attention = 0
frame_cnt = 0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    start_time = time.time()
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    frame_cnt += 1
    height, width = image.shape[0], image.shape[1]
    # print(height, width)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        landmark_3d = process_landmark(face_landmarks, width, height)
        rotvec = process_face_landmark(landmark_3d, width, height)

        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_tesselation_style())
        #
        # # 轮廓
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_contours_style())
        #
        # ## 虹膜
        # mp_drawing.draw_landmarks(
        #     image=image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec=None,
        #     connection_drawing_spec=mp_drawing_styles
        #     .get_default_face_mesh_iris_connections_style())


        left_iris, right_iris = iris_position(landmark_3d)
        cv2.circle(image, left_iris, 1, color=(0, 0, 255))
        cv2.circle(image, right_iris, 1, color=(0, 0, 255))
        left_eye, right_eye = eye_position(landmark_3d)

        iris_eye_pos = [left_iris[0]-left_eye[0], left_iris[1]-left_eye[1], right_iris[0]-right_eye[0], right_iris[1]-left_eye[1]]
        iris_eye_pos_list.insert(0, iris_eye_pos)
        iris_eye_pos_list = iris_eye_pos_list[:20]


    # calc attention
    if len(iris_eye_pos_list) >= 2 and frame_cnt % 3 == 0:
        active_cnt = 0
        active_threshold = 2
        increase_rate = 0.1

        for i in range(len(iris_eye_pos_list) - 5):
            iris_diff = [iris_eye_pos_list[i][idx]-iris_eye_pos_list[i+2][idx] for idx in range(len(iris_eye_pos_list[0]))]

            for item in iris_diff:
                if abs(item) > active_threshold:
                    active_cnt = 1
        if active_cnt == 1:
            active_cnt = -1
        else:
            active_cnt = 1
        attention = max(min(attention + increase_rate*active_cnt, 10), 0)

    # start_time = time.time()
    # top = min([abs(int(item)) for item in landmark_3d[:, 1]])
    # bottom = max([abs(int(item)) for item in landmark_3d[:, 1]])
    # left = min([abs(int(item)) for item in landmark_3d[:, 0]])
    # right = max([abs(int(item)) for item in landmark_3d[:, 0]])
    # print(top, bottom, left, right)
    # crop = image[top:bottom, left:right]
    # emotion = emotion_predict(crop)
    # print('emotion cost:', time.time()-start_time)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    cv2.putText(image, "Horizontal:{:6.2f} degree".format(90*rotvec[1]), (20, 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Vertical  :{:6.2f} degree".format(-90*rotvec[0]), (20, 60), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Attention :{:6.1f} ".format(attention), (20, 120), font, 0.8, (0, 0, 0), 1,  cv2.LINE_AA)
    # cv2.putText(image, "Emotion : {} ".format(emotion), (20, 150), font, 0.8, (0, 0, 0), 1,  cv2.LINE_AA)

    print(time.time() - start_time)

    cv2.imshow('Face', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()