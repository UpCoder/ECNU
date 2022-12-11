import cv2
import time
import numpy as np
import mediapipe as mp
import threading

from face.landmark_utils import landmark_handler
from face.expression_model.model import FacialExpressionModel
from face.head_nod import NodDetecter

font = cv2.FONT_HERSHEY_SIMPLEX

class FaceAnalyzer(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.has_face = False
        self.image = None
        self.infos = {}

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        # drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.expression_model = FacialExpressionModel("./face/expression_model/model.json",
                                                      "./face/expression_model/model_weights.h5")

        self.text_size = 0.6
        self.text_space = 20
        self.text_cnt = 0

        self.landmark_3d = None

        # expression
        self.expression = ""
        self.smile_score = 0
        self.frown_score = 0

        # nod detect
        self.nod_cnt = 0
        self.nod_detecter = NodDetecter(10)

    def set_size(self, width, height):
        self.width = width
        self.height = height

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

    def draw_face(self, image):
        if not self.has_face:
            return image
        # FaceMesh
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=self.face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        # 轮廓
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=self.face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())
        ## 虹膜
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=self.face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
        return image


    def face_rect_by_landmark(self, face_landmarks):
        if face_landmarks is None:
            return None

        x_min = self.width
        x_max = 0
        y_min = self.height
        y_max = 0
        for point in face_landmarks:
            x_min = min(x_min, int(abs(point[0])))
            x_max = max(x_max, int(abs(point[0])))
            y_min = min(y_min, int(abs(point[1])))
            y_max = max(y_max, int(abs(point[1])))
        print(x_min, x_max, y_min, y_max)
        x_buff, y_buff = int((x_max-x_min)/4), int((y_max-y_min)/4)
        # rect = [max(0, x_min-x_buff), min(self.width, x_max+x_buff), max(0, y_min-y_buff), min(self.height, y_max+y_buff)]
        rect = [x_min, x_max, y_min, y_max]
        print(rect)
        return rect

    def _put_text(self, image, text, str_val):
        self.text_cnt += 1
        cv2.putText(image, "{}:{}".format(text, str_val), (20, self.text_cnt * self.text_space),
                    font, self.text_size, (0, 0, 0), 1, cv2.LINE_AA)
        return

    def face_infer(self):
        try:
            start_time = time.time()
            raw_image = self.image
            self.set_size(raw_image.shape[1], raw_image.shape[0])
            self.text_cnt = 0
            self.infos = {}

            raw_image.flags.writeable = False
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(raw_image)
            raw_image.flags.writeable = True
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

            # Have face in image
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                self.has_face = True
                self.face_landmarks = results.multi_face_landmarks[0]
                self.landmark_3d = self.process_landmark(self.face_landmarks, self.width, self.height)
                self.infos = landmark_handler(self.landmark_3d, self.width, self.height)

                self.nod_detecter.update(self.infos["face_vertical_orientation"])
                if self.nod_detecter.detect():
                    self.nod_cnt += 1
                    self.nod_detecter.clear()
                    self.infos['face_nod_count'] = self.nod_cnt

                expression_start_time = time.time()
                face_rect = self.face_rect_by_landmark(self.landmark_3d)
                face_img = raw_image[face_rect[2]:face_rect[3], face_rect[0]:face_rect[1]]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_img = cv2.resize(face_img, (48, 48))
                pred, score = self.expression_model.predict_emotion(face_img[np.newaxis, :, :, np.newaxis])
                self.expression = pred
                self.smile_score = 0
                self.frown_score = 0
                if pred == 'Happy':
                    self.smile_score = max((score-0.5)*2, 0)
                elif pred == 'Angry' or pred == 'Sad':
                    self.frown_score = score

                self.infos['face_expression'] = self.expression
                self.infos['face_smile'] = self.smile_score
                self.infos['face_frown'] = self.frown_score
                print("expression:{}  time:{}".format(pred, time.time()-expression_start_time))

                # self._put_text(draw_image, "expression", pred)
                # self._put_text(draw_image, "smile_score", "{:.2f}".format(self.smile_score))
                # self._put_text(draw_image, "frown_score", "{:.2f}".format(self.frown_score))
                # self._put_text(draw_image, "Mouth Open", "{:4.2f}".format(infos["mouth_open"]))
                # self._put_text(draw_image, "Left Eye Open", "{:4.2f}".format(infos["left_eye_open"]))
                # self._put_text(draw_image, "Right Eye Open", "{:4.2f}".format(infos["right_eye_open"]))
                # self._put_text(draw_image, "H_orient", "{:4.2f}".format(infos["H_orient"]))
                # self._put_text(draw_image, "V_orient", "{:4.2f}".format(infos["V_orient"]))
                # self._put_text(draw_image, "Nod", str(self.nod_cnt))

            else:
                self.has_face = False

            # self.draw_face(draw_image)
            # image = cv2.flip(image, 1)
            print('mesh_time:', time.time() - start_time)
        except Exception as e:
            print('Error:', str(e))
            return
        return

    def start(self, image):
        self.image = image
        self.face_thread = threading.Thread(target=self.face_infer, args=())
        self.face_thread.start()

    def wait(self):
        self.face_thread.join()