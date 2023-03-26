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
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--send_data', type=bool, default=False)
    parser.add_argument('--host', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=8888)
    parser.add_argument('--record_file_dir', type=str, default='./')
    parser.add_argument('--record_wav_dir', type=str, default='./wav')
    args = parser.parse_args()
    print(args)
    timestamp_now = time.time()
    record_file_path = os.path.join(args.record_file_dir, '{}_record.txt'.format(timestamp_now))
    wav_dir = os.path.join(args.record_wav_dir, '{}'.format(timestamp_now))
    recorder = Recorder(record_file_path, wav_dir)
    recorder.start()
    start_pipeline(recorder.q1)
    camera = Camera()
    camera.set_size(args.width, args.height)

    face = FaceAnalyzer(args.width, args.height, recorder.q1)
    body = VideoProcessor(calc_frame_interval=10000000000,
                          body_processor_method='yolov7',
                          recorder_queue=recorder.q1)

    socket_client = SocketClient()
    socket_client.connet(host=args.host, port=args.port)

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, im_row = camera.video_capure.read()
        # im_row = cv2.imread('demo.jpg')
        im_row = im_row[:480, 150:150 + 340]

        im_rd = im_row.copy()
        im_rd1 = im_row.copy()

        face.start(im_rd)
        # image = im_rd
        #image, _, body_info = body.processing_frame(im_rd1, im_rd1, need_info=True)
        body.start_processing_frame_thread(im_rd1, im_rd1, True)

        face.wait()
        body.wait_processing_frame_thread()
        image = body.processing_frame_result['annotation_image']
        # image = im_rd1
        face.draw_face(image)

        infos = {
            'origin': 'face',
            **face.infos,
            # **body_info
            **body.processing_frame_result['metrics']
        }
        recorder.q1.put(infos)
        image = cv2.flip(image, 1)

        send_time = time.time()
        if args.send_data:
            socket_client.send_image(image, infos)

        if not args.send_data:
            """
            当不发送数据的时候，才会显示，否则会导致read时间上涨
            """
            cv2.imshow("demo", image)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        print('Single frame cost: ', time.time() - start_time)
        print('fps: {}'.format(1 / (time.time() - start_time)))
        # print('######'*5)


    camera.video_capure.release()
    cv2.destroyAllWindows()
    recorder.end()