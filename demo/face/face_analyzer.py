import cv2
import numpy as np
import mediapipe as mp

from landmark_utils import landmark_handler

font = cv2.FONT_HERSHEY_SIMPLEX

class FaceAnalyzer(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        # drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.text_size = 0.6
        self.text_space = 20

    def process_landmark(self, face_landmarks, width, height):
        landmark1 = face_landmarks.landmark  ## 478 is known number
        np_array = np.zeros((len(landmark1), 3), np.float64)
        for i in range(len(landmark1)):
            np_array[i, 0] = landmark1[i].x
            np_array[i, 1] = landmark1[i].y
            np_array[i, 2] = landmark1[i].z
        np_array[:, 0] = np_array[:, 0] * width
        np_array[:, 1] = - np_array[:, 1] * height
        np_array[:, 2] = - np_array[:, 2] * width
        return np_array

    def draw_face(self, image, face_landmarks):
        # FaceMesh
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        # 轮廓
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())
        ## 虹膜
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())

    def face_infer(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_landmarks = results.multi_face_landmarks[0]

            self.draw_face(image, face_landmarks)
            self.landmark_3d = self.process_landmark(face_landmarks, self.width, self.height)

        image = cv2.flip(image, 1)
        infos = landmark_handler(self.landmark_3d, self.width, self.height)

        cv2.putText(image, "Horizontal Ori:{:6.2f} degree".format(infos["H_orient"]), (20, 1*self.text_space),
                    font, self.text_size, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "Vertical Ori :{:6.2f} degree".format(infos["V_orient"]), (20, 2*self.text_space),
                    font, self.text_size, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(image, "Horizontal Gaze :{:6.2f} ".format(infos["H_iris"]), (20, 4*self.text_space),
                    font, self.text_size, (0, 0, 0), 1,  cv2.LINE_AA)
        cv2.putText(image, "Vertical Gaze :{:6.2f} ".format(infos["V_iris"]), (20, 5*self.text_space),
                    font, self.text_size, (0, 0, 0), 1,  cv2.LINE_AA)

        cv2.putText(image, "Mouth Open :{:6.2f} ".format(infos["mouth_open"]), (20, 7*self.text_space),
                    font, self.text_size, (0, 0, 0), 1,  cv2.LINE_AA)
        cv2.putText(image, "Left Eye Open :{:6.2f} ".format(infos["left_eye_open"]), (20, 8*self.text_space),
                    font, self.text_size, (0, 0, 0), 1,  cv2.LINE_AA)
        cv2.putText(image, "Right Eye Open :{:6.2f} ".format(infos["right_eye_open"]), (20, 9*self.text_space),
                    font, self.text_size, (0, 0, 0), 1,  cv2.LINE_AA)

        return image, infos