import cv2
import time
import argparse
import os
from sensor.camera import Camera
from face.face_analyzer import FaceAnalyzer
from commu.client import SocketClient
from body.body_processor import VideoProcessor
from demo.demo_asr.AudioProcessor import start_pipeline
from commu.record import Recorder


if __name__ == '__main__':
    camera = Camera()
    camera.set_size(640, 480)

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, im_row = camera.video_capure.read()
        print('camera caption cost:', time.time() - start_time)

        # cv2.imshow("demo", im_row)
        # k = cv2.waitKey(1)

    camera.video_capure.release()
    cv2.destroyAllWindows()