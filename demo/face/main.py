
import numpy as np
import cv2
import time
import timeit

from face_analyzer import FaceAnalyzer


width = 1280
height = 960
FA = FaceAnalyzer(width, height)

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while cap.isOpened():
    ret_flag, im_row = cap.read()
    im_rd = im_row.copy()

    image, infos = FA.face_infer(im_rd)

    cv2.imshow("camera", image)
    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
