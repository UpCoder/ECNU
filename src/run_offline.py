import cv2
import time
import argparse
import os
from sensor.camera import Camera
from face.face_analyzer import FaceAnalyzer
from commu.client import SocketClient
from body.body_processor import VideoProcessor
#from demo.demo_asr.AudioProcessor import start_pipeline
from commu.record import Recorder
from glob import glob

def center_crop(img, height=480, width=340):
    img_h, img_w = img.shape[0], img.shape[1]
    resize_w = int(img_w*height/img_h)

    img = cv2.resize(img, (resize_w, height))
    left_w = (resize_w - width)//2
    img = img[:, left_w:left_w + width]

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--show_data', type=bool, default=False)
    parser.add_argument('--record_wav_dir', type=str, default='./wav')
    #parser.add_argument('--video_path', type=str, default='26.mp4')
    parser.add_argument('--video_dir', type=str, default='E:\\videos')
    # parser.add_argument('--video_dir', type=str, default='C:\\Users\\30644\\Desktop\\videos')
    parser.add_argument('--save_dir', type=str, default='E:\\interview_feature\\visual')
    args = parser.parse_args()
    print(args)

    for mp4_path in glob(os.path.join(args.video_dir, '*.mp4')):
        print(f'handler: {mp4_path}')

        timestamp_now = time.time()
        record_predix = mp4_path.split('\\')[-1].split('.')[0]
        record_file_path = os.path.join(args.save_dir, '{}_{}_visual_record.txt'.format(record_predix, timestamp_now))

        wav_dir = os.path.join(args.save_dir, args.record_wav_dir, '{}'.format(timestamp_now))
        recorder = Recorder(record_file_path, wav_dir)
        recorder.start()
        #start_pipeline(recorder.q1)
        camera = Camera(video_path=mp4_path)
        video_fps = camera.video_capure.get(cv2.CAP_PROP_FPS)

        face = FaceAnalyzer(args.width, args.height, recorder.q1)
        body = VideoProcessor(calc_frame_interval=10000000000,
                              body_processor_method='yolov7',
                              recorder_queue=recorder.q1)

        frame_count = 0
        save_frame_count = 0
        save_fps = 10
        while camera.video_capure.isOpened():
            try:
                start_time = time.time()
                frame_count += 1
                ret_flag, im_row = camera.video_capure.read()

                """
                按指定帧率save_fps处理offline视频
                """
                if frame_count/video_fps > save_frame_count/save_fps:
                    save_frame_count += 1
                else:
                    continue

                """
                等比缩放图片与Crop
                """
                im_row = center_crop(im_row)

                im_rd = im_row.copy()
                im_rd1 = im_row.copy()

                face.start(im_rd)
                # image = im_rd1
                #image, _, body_info = body.processing_frame(im_rd1, im_rd1, need_info=True)
                body.start_processing_frame_thread(im_rd1, im_rd1, True)

                face.wait()
                body.wait_processing_frame_thread()
                image = body.processing_frame_result['annotation_image']
                face.draw_face(image)

                infos = {
                    'origin': 'face',
                    **face.infos,
                    # **body_info
                    **body.processing_frame_result['metrics']
                }


                infos['timestamp'] = frame_count/video_fps
                print('timestamp:', infos['timestamp'])
                recorder.q1.put(infos)
                image = cv2.flip(image, 1)

                send_time = time.time()

                if args.show_data:
                    """
                    当不发送数据的时候，才会显示，否则会导致read时间上涨
                    """
                    cv2.imshow("demo", image)
                    k = cv2.waitKey(1)
                    if k == ord('q'):
                        break

                print('Single frame cost: ', time.time() - start_time)
                print('fps: {}'.format(1 / (time.time() - start_time)))
                print('######'*5)
            except Exception as e:
                print("ERROR:", str(e))
                break

        camera.video_capure.release()
        cv2.destroyAllWindows()
        recorder.end()