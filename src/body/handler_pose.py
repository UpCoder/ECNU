import cv2
import mediapipe as mp
import numpy as np
from demo.body_pix.keypoint_yolov7 import processing_pose_frame as processing_pose_frame_yolov7
# from demo_release import inference_frame_image as processing_pose_frame_light
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.4)
BG_COLOR = [255, 255, 255]
class NormalizedLandmarkListSelf(object):
    def __init__(self):
        self.landmark = None


def processing_frame(method, frame, annotation_image, coord_names=['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                     is_white_bg=False, pose_result=None, connections=None):
    if method == 'mediapipe':
        return processing_frame_mediapipe(frame, annotation_image, coord_names, is_white_bg, pose_result, connections)
    elif method == 'yolov7':
        return processing_pose_frame_yolov7(frame, annotation_image, coord_names, None)
    elif method == 'light':
        return processing_pose_frame_light(frame, annotation_image, coord_names, None)
    else:
        raise ValueError(f'method={method} do not support, only support mediapipe/yolov7 now!')


def processing_frame_mediapipe(frame, annotation_image,
                               coord_names=['LEFT_SHOULDER', 'RIGHT_SHOULDER'],
                               is_white_bg=False,
                               pose_result=None,
                               connections=None):
    image_height, image_width, _ = frame.shape
    # Convert the BGR image to RGB before processing.
    if pose_result is None:
        results = pose.process(frame)
    else:
        results = pose_result

    if not results.pose_landmarks:
        return annotation_image, {}, results
    # print(
    #     f'Nose coordinates: ('
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
    #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    # )
    coord_points = {}
    for coord_name in coord_names:
        visibility = results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].visibility
        if visibility >= 0.9:
            coord_points[coord_name] = {
                'x': results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].x,
                'y': results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].y,
                'z': results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].z,
                'vis': visibility
            }
        else:
            coord_points[coord_name] = {
                'x': -1,
                'y': -1,
                'z': -1,
                'vis': visibility
            }

    if annotation_image is not None:
        if is_white_bg:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(annotation_image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotation_image = np.where(condition, annotation_image, bg_image)
        # Draw pose landmarks on the image.
        # from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
        mark_list = NormalizedLandmarkListSelf()
        mark_list.landmark = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)]
            for coord_name in coord_names
        ]
        # for ele in results.pose_landmarks:
        #     print(f'ele: {ele}', type(ele))
        # print('results.pose_landmarks: ', type(results.pose_landmarks), results.pose_landmarks)
        # print('mark_list: ', type(mark_list), mark_list)
        mp_drawing.draw_landmarks(
            annotation_image,
            # results.pose_landmarks,
            mark_list,
            mp_pose.POSE_CONNECTIONS if connections is None else connections,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    return annotation_image, coord_points, results


def processing_image_demo(image_path, coord_names=['LEFT_SHOULDER', 'RIGHT_SHOULDER']):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )
    coord_points = {}
    for coord_name in coord_names:
        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
        coord_points[coord_name] = (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].x * image_width,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.__getitem__(coord_name)].y * image_height
        )

    annotated_image = image.copy()

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('pose.png', annotated_image)
    return coord_points
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def webcam_demo():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # print('x: ', results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
        #       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        # print('y: ', results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        #       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        print('z: ', results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z,
              results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()


if __name__ == '__main__':
    coord_points = processing_image_demo(
        # 'body_test1.png'
        'body_test2.jpg'
    )
    print(coord_points)
    webcam_demo()