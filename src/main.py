import cv2
import time
import argparse

from sensor.camera import Camera
from face.face_analyzer import FaceAnalyzer
from commu.client import SocketClient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    parser.add_argument('--send_data', type=bool, default=False)
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    print(args)

    camera = Camera()
    camera.set_size(args.width, args.height)

    face = FaceAnalyzer(args.width, args.height)

    socket_client = SocketClient()
    socket_client.connet(host=args.host, port=args.port)

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, im_row = camera.video_capure.read()
        print('camera caption cost:', time.time() - start_time)
        im_rd = im_row.copy()
        print(im_rd.shape)
        im_rd = im_rd[:480, 150:150+340]
        print(im_rd.shape)

        image, infos = face.face_infer(im_rd)
        print(infos)

        send_time = time.time()
        if args.send_data:
            socket_client.send_image(image, infos)
        print('send_data cost:', time.time() - send_time)

        cv2.imshow("demo", image)
        k = cv2.waitKey(1)

        print('Single frame cost:', time.time() - start_time)
        print('######'*5)
        if k == ord('q'):
            break

    camera.video_capure.release()
    cv2.destroyAllWindows()