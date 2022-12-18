import cv2
import time
import argparse

from sensor.camera import Camera
from face.face_analyzer import FaceAnalyzer
from commu.client import SocketClient
from body.body_processor import VideoProcessor
#from demo.demo_asr.AudioProcessor import start_pipeline
from commu.record import Recorder


if __name__ == '__main__':
    #start_pipeline('localhost', 8889)
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--send_data', type=bool, default=True)
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--record_file', type=str, default='record.txt')
    args = parser.parse_args()
    print(args)

    recorder = Recorder(args.record_file)
    recorder.start()

    camera = Camera()
    camera.set_size(args.width, args.height)

    face = FaceAnalyzer(args.width, args.height, recorder.q1)
    #body = VideoProcessor(calc_frame_interval=10000000000, body_processor_method='yolov7')
    body = VideoProcessor()

    socket_client = SocketClient()
    socket_client.connet(host=args.host, port=args.port)

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, im_row = camera.video_capure.read()
        # im_row = cv2.imread('demo.jpg')
        #print('camera caption cost:', time.time() - start_time)
        im_row = im_row[:480, 150:150 + 340]

        im_rd = im_row.copy()
        im_rd1 = im_row.copy()
        #print(im_rd.shape)

        face.start(im_rd)
        body.start_processing_frame_thread(im_rd1, im_rd1, True)

        face.wait()
        body.wait_processing_frame_thread()
        image = body.processing_frame_result['annotation_image']
        face.draw_face(image)

        infos = {
            **face.infos,
            # **body_info
            **body.processing_frame_result['metrics']
        }

        image = cv2.flip(image, 1)

        send_time = time.time()
        if args.send_data:
            socket_client.send_image(image, infos)
        #print('send_data cost:', time.time() - send_time)

        cv2.imshow("demo", image)
        k = cv2.waitKey(1)

        print('Single frame cost:', time.time() - start_time)
        print('######'*5)
        if k == ord('q'):
            break

    camera.video_capure.release()
    cv2.destroyAllWindows()