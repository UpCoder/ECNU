import cv2
import math
import numpy as np
import pyaudio


p = pyaudio.PyAudio()


print(p.get_device_count())

for i in range(p.get_device_count()):
    device = p.get_device_info_by_index(i)
    if device.get('maxInputChannels') > 0 and 'Logi' in device.get('name'):
        print(device)



# cap = cv2.VideoCapture(701)
#
# cap.set(3, 8000)
# cap.set(4, 4000)
# while cap.isOpened():
#     for i in range(3, 6):
#         print(cap.get(i))
#     success, image = cap.read()
#     image = cv2.flip(image, 1)
#
#     cv2.imshow('Face', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# cap.release()
#
# FACE_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398, 362]
# FACE_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 33, 246, 161, 160, 159, 158, 157, 173, 133]
#
# print(len(FACE_LEFT_EYE), len(FACE_RIGHT_EYE))

