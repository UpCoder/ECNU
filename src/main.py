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

    camera = Camera()
    camera.set_size(args.width, args.height)

    face = FaceAnalyzer(args.width, args.height)

    socket_client = SocketClient()
    socket_client.connet(host=args.host, port=args.port)

    while camera.video_capure.isOpened():
        ret_flag, im_row = camera.video_capure.read()
        im_rd = im_row.copy()

        image, infos = face.face_infer(im_rd)
        print(infos)

        if args.send_data:
            socket_client.send_image(image, infos)
            break

        cv2.imshow("demo", image)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break

    camera.video_capure.release()
    cv2.destroyAllWindows()