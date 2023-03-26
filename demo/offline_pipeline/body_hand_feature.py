import cv2
from src.body.body_processor import VideoProcessor

body = VideoProcessor(calc_frame_interval=10000000000,
                      body_processor_method='yolov7',
                      recorder_queue=None)


def pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    print(cap)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            im_rd1 = frame.copy()
            # Display the resulting frame
            body.start_processing_frame_thread(im_rd1, im_rd1, True)

            body.wait_processing_frame_thread()
            image = body.processing_frame_result['annotation_image']

            cv2.imshow('Frame', image)
            print('metrics: {}'.format(**body.processing_frame_result['metrics']))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = 'E:\\BaiduNetdiskDownload\\1.mp4'
    pipeline(video_path)