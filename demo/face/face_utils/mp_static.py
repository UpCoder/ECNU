import cv2
import math
import numpy as np
import mediapipe as mp
from scipy.spatial.transform import Rotation as Rtool

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

font = cv2.FONT_HERSHEY_SIMPLEX


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

def process_face_landmark(face_landmarks, width, height):
    landmark_3d = process_landmark(face_landmarks, width, height)
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
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    height, width = image.shape[0], image.shape[1]
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
        rotvec = process_face_landmark(face_landmarks, width, height)
        print(rotvec)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        ## 轮廓
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        ## 虹膜
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    cv2.putText(image, "z:{:.2f}".format(-90*rotvec[0]), (20, 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "y:{:.6f}".format(rotvec[1]), (20, 60), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "x:{:.6f}".format(rotvec[2]), (20, 90), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(image, "x/y:{:.6f}".format(math.atan(rotvec[2]/rotvec[1])), (20, 120), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "y/x:{:.6f}".format(math.atan(rotvec[1]/rotvec[2])/math.pi*180), (20, 150), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Face', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()