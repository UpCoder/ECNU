import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5
     )


def processing_frame(image, annotation_image=None,
                     coord_names=[],
                     hand_result=None):
    image = image[:, :, ::-1]
    if hand_result is None:
        results = hands.process(image)
    else:
        results = hand_result
    # print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
        return annotation_image, None, None, results
    image_height, image_width, _ = image.shape

    left_hand_coords = {}
    right_hand_coords = {}
    for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_coords = {}
        for coord_name in coord_names:
            hand_coords[coord_name] = {
                'x': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].x,
                'y': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].y,
                'z': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].z
            }
        # print(hand_info.classification[0].label)
        if hand_info.classification[0].label == 'Right':
            right_hand_coords = hand_coords
        elif hand_info.classification[0].label == 'Left':
            left_hand_coords = hand_coords
        if annotation_image is not None:
            annotation_image = np.asarray(annotation_image).astype(np.uint8)
            # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                annotation_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                None, # mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    annotation_image = np.asarray(annotation_image).astype(np.uint8)
    return annotation_image, left_hand_coords, right_hand_coords, results


def preprocessing_image_demo(image, is_path=True, coord_names=[]):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
     ) as hands:
        if is_path:
            image = cv2.imread(image)
            image = image[:, :, ::-1]
            # Convert the BGR image to RGB before processing.
        results = hands.process(image)
        # print('Handedness:', results.multi_handedness)
        # print('Handedness:', results.multi_hand_landmarks)
        if not results.multi_hand_landmarks:
            return
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        left_hand_coords = {}
        right_hand_coords = {}
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_coords = {}
            for coord_name in coord_names:
                hand_coords[coord_name] = {
                    'x': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].x,
                    'y': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].y,
                    'z': hand_landmarks.landmark[mp_hands.HandLandmark.__getitem__(coord_name)].z
                }
            print(hand_info.classification[0].label)
            if hand_info.classification[0].label == 'Right':
                right_hand_coords = hand_coords
            elif hand_info.classification[0].label == 'Left':
                left_hand_coords = hand_coords
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            'hand.jpg', cv2.flip(annotated_image, 1))
    return left_hand_coords, right_hand_coords


def webcam_demo():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
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
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()


if __name__ == '__main__':
    left_hand_info, right_hand_info = preprocessing_image_demo(
        'body_test3.png',
        is_path=True,
        coord_names=[
            'INDEX_FINGER_PIP',
            'INDEX_FINGER_DIP',
            'MIDDLE_FINGER_PIP',
            'MIDDLE_FINGER_DIP',
            'PINKY_PIP',
            'PINKY_DIP',

        ]
    )
    print(f'left_hand_info: {left_hand_info}')
    print(f'right_hand_info: {right_hand_info}')
    # webcam_demo()