
import cv2
import numpy as np
from deepface import DeepFace
from deepface.detectors import FaceDetector
# from keras.preprocessing import image

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])
if tf_major_version == 1:
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow.keras.preprocessing import image

backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
]

def deepface_detector(img, detector_backend="opencv"):
    face_detector = FaceDetector.build_model(detector_backend)
    face_objs = FaceDetector.detect_faces(face_detector, detector_backend, img)

    print(face_objs)
    cv2.imshow('text', face_objs[0][0])
    cv2.waitKey(0)
    return face_objs[0][0]

def preprocess_face(img, target_size=(224, 224), grayscale = False):
    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    img_pixels = image.img_to_array(img) #what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255 #normalize input in [0, 1]

    return img_pixels

emotion_model = DeepFace.build_model("Emotion")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def deepface_emotion(img):
    img_gray = preprocess_face(img, target_size=(48, 48), grayscale=True)
    emotion_predictions = emotion_model.predict(img_gray, verbose=0)[0, :]
    dominant_emotion = emotion_labels[np.argmax(emotion_predictions)]
    return dominant_emotion, emotion_predictions[np.argmax(emotion_predictions)]


if __name__ == '__main__':
    print('test...')
    img_path = 'test_img.jpg'
    img = cv2.imread(img_path)

    face_img = deepface_detector(img, detector_backend='opencv')
    rtn = deepface_emotion(face_img)
    print(rtn)



