import cv2
import numpy as np
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions
from keras.preprocessing import image

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])
if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image

model = Emotion.loadModel()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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

	#------------------------------------------

	#double check: if target image is not still the same size with target.
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	return img_pixels



def emotion_predict(image):
	img = preprocess_face(image, target_size=(48, 48), grayscale=True)

	emotion_predictions = model.predict(img, verbose=0)[0, :]
	sum_of_predictions = emotion_predictions.sum()

	resp_obj = {}

	for i in range(0, len(emotion_labels)):
		emotion_label = emotion_labels[i]
		emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
		resp_obj[emotion_label] = emotion_prediction

	dominant_emotion = emotion_labels[np.argmax(emotion_predictions)]
	print(dominant_emotion)

	return dominant_emotion