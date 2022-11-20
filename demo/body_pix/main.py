import json
import os

import cv2
import tensorflow as tf
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from utils import parse_video, write_videos
from handler_hand import processing_frame as processing_hand_frame
from handler_pose import processing_frame as processing_pose_frame
model_name = 'MOBILENET_FLOAT_100_STRIDE_16'
model_name2var = {
    'MOBILENET_RESNET50_FLOAT_STRIDE_32': BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_32,
    'MOBILENET_FLOAT_100_STRIDE_16': BodyPixModelPaths.MOBILENET_FLOAT_100_STRIDE_16
}
# bp_model = load_model(download_model(model_name2var[model_name]))


def handler_body_seg(frame, model):
    pass


def pipeline_video(video_path, model):
    print(f'Start pipeline, {os.path.basename(video_path)}')
    save_dir = f'./{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    frames = parse_video(video_path)

    body_segs = []  # 身体分割结果
    hand_segs = []   # 手分割结果
    body_skeletons = []  # 身体skeleton结果
    for frame in tqdm(frames):
        frame = np.asarray(frame)
        prediction = model.predict_single(frame)  # Passing the image to the model
        all_mask = prediction.get_mask(threshold=0.3).numpy().astype(np.uint8)
        hand_mask = prediction.get_part_mask(all_mask, part_names=[
            'left_hand',
            'right_hand',
            'left_upper_arm_front',
            'left_upper_arm_back',
            'right_upper_arm_front',
            'right_upper_arm_back',
            'left_lower_arm_front',
            'left_lower_arm_back',
            'right_lower_arm_front',
            'right_lower_arm_back',
        ])

        # all_mask = np.asarray(all_mask).repeat(3, axis=-1)
        # print(np.sum(all_mask), np.sum(hand_mask))
        # hand_mask = np.asarray(hand_mask).repeat(3, axis=-1)
        poses = prediction.get_poses()
        image_with_pose = draw_poses(
            frame.copy(),
            poses,
            keypoints_color=(255, 100, 100),
            skeleton_color=(100, 100, 255)
        )
        processing_hand_frame(frame, image_with_pose)
        body_skeletons.append(image_with_pose)
        body_seg = cv2.bitwise_and(frame, frame, mask=all_mask)
        hand_seg = cv2.bitwise_and(frame, frame, mask=hand_mask)
        hand_segs.append(hand_seg)
        body_segs.append(body_seg)
    write_videos(
        body_skeletons,
        fps=5,
        save_path=os.path.join(save_dir, 'skeleton.avi')
    )
    write_videos(
        np.asarray(np.asarray(body_segs) * 200, np.int),
        fps=5,
        save_path=os.path.join(save_dir, 'body_seg.avi')
    )
    write_videos(
        np.asarray(np.asarray(hand_segs) * 255, np.int),
        fps=5,
        save_path=os.path.join(save_dir, 'hand_seg.avi')
    )

def pipeline_video_mediapipe_VideoProcessor(video_path):
    from VideoProcessor import VideoProcessor
    target_fps = 29
    save_fps = target_fps // 2
    calc_frame_interval = 4
    frames = parse_video(video_path, target_fps=target_fps)
    video_processor = VideoProcessor(window_size=target_fps // 4,
                                     calc_frame_interval=calc_frame_interval)
    print(f'Start pipeline, {os.path.basename(video_path)}')
    model_name = 'mediapipe_VideoProcessor'
    save_dir = f'./{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    annotation_frames, annotation_frames_with_txt = video_processor.processing_frames(frames)
    # for idx, annotation_frame in enumerate(annotation_frames):
    #     cv2.imwrite(os.path.join(save_dir, f'0_{idx}.jpg'), annotation_frame)
    # for idx, annotation_frame in enumerate(annotation_frames_with_txt):
    #     cv2.imwrite(os.path.join(save_dir, f'1_{idx}.jpg'), annotation_frame)
    write_videos(
        annotation_frames_with_txt,
        fps=save_fps,
        save_path=os.path.join(save_dir, os.path.basename(video_path))
    )

def pipeline_video_mediapipe(video_path):
    print(f'Start pipeline, {os.path.basename(video_path)}')
    model_name = 'mediapipe'
    save_dir = f'./{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_fps = 29
    frames = parse_video(video_path, target_fps=target_fps)
    save_fps = target_fps // 2
    body_segs = []  # 身体分割结果
    hand_segs = []   # 手分割结果
    body_skeletons = []  # 身体skeleton结果
    global_coord_points = []
    for frame in tqdm(frames):
        frame = np.asarray(frame)
        body_seg = frame.copy()
        print(1, np.shape(body_seg))
        body_seg, coord_points = processing_pose_frame(
            frame, body_seg
        )
        print(2, np.shape(body_seg))
        body_segs.append(body_seg)
        hand_seg = frame.copy()
        hand_seg = processing_hand_frame(frame, hand_seg)
        print(3, np.shape(hand_seg))
        hand_segs.append(hand_seg)
        global_coord_points.append(coord_points)
    # write_videos(
    #     body_skeletons,
    #     fps=5,
    #     save_path=os.path.join(save_dir, 'skeleton.avi')
    # )
    json.dump({
        'coords': global_coord_points
    }, open('shoulder_coords.json', 'w'))
    write_videos(
        body_segs,
        fps=save_fps,
        save_path=os.path.join(save_dir, os.path.basename(video_path).replace('.mp4', '_body.avi'))
    )
    write_videos(
        hand_segs,
        fps=save_fps,
        save_path=os.path.join(save_dir, os.path.basename(video_path).replace('.mp4', '_hand.avi'))
    )


def demo():
    print('load model finish')
    image_path = "body_test2.jpg"
    # image = cv2.imread(image_path)
    image = tf.keras.preprocessing.image.load_img(image_path)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    """
    'left_face',
    'right_face',
    'left_upper_arm_front',
    'left_upper_arm_back',
    'right_upper_arm_front',
    'right_upper_arm_back',
    'left_lower_arm_front',
    'left_lower_arm_back',
    'right_lower_arm_front',
    'right_lower_arm_back',
    'left_hand',
    'right_hand',
    'torso_front',
    'torso_back',
    'left_upper_leg_front',
    'left_upper_leg_back',
    'right_upper_leg_front',
    'right_upper_leg_back',
    'left_lower_leg_front',
    'left_lower_leg_back',
    'right_lower_leg_front',
    'right_lower_leg_back',
    'left_feet',
    'right_feet'
    """
    costs = []
    for idx in range(100):
        s = time.time()
        print(np.shape(image_array), np.min(image_array), np.max(image_array))

        prediction = bp_model.predict_single(image_array) # Passing the image to the model
        all_mask = prediction.get_mask(threshold=0.3).numpy().astype(np.uint8)
        hand_mask = prediction.get_part_mask(all_mask, part_names=[
            'left_hand',
            'right_hand',
            'left_upper_arm_front',
            'left_upper_arm_back',
            'right_upper_arm_front',
            'right_upper_arm_back',
            'left_lower_arm_front',
            'left_lower_arm_back',
            'right_lower_arm_front',
            'right_lower_arm_back',
        ])
        all_mask = np.asarray(all_mask).repeat(3, axis=-1)
        hand_mask = np.asarray(hand_mask).repeat(3, axis=-1)
        print(np.shape(all_mask), np.shape(hand_mask))
        print(np.min(all_mask), np.max(all_mask))
        print(np.min(hand_mask), np.max(hand_mask))

        e = time.time()
        poses = prediction.get_poses()
        image_with_pose = draw_poses(
            image_array.copy(),
            poses,
            keypoints_color=(255, 100, 100),
            skeleton_color=(100, 100, 255)
        )
        tf.keras.preprocessing.image.save_img(
            image_path.replace('.jpg', 'pose.jpg'),
            image_with_pose
        )
        print('cost: ', e-s)
        costs.append(e-s)
    print(np.mean(costs))

    all_mask = cv2.bitwise_and(image, image, mask=all_mask)
    hand_mask = cv2.bitwise_and(image, image, mask=hand_mask)
    cv2.imwrite(image_path.replace('.jpg', '_mask.jpg'), all_mask)
    cv2.imwrite(image_path.replace('.jpg', '_hand_mask.jpg'), hand_mask)


if __name__ == '__main__':
    # pipeline_video(
    #     'C:\\Users\\cs_li\\Documents\\WXWork\\1688854406374298\\Cache\\Video\\2022-10\\'
    #     'WIN_20221029_19_10_12_Pro.mp4',
    #     bp_model
    # )
    # pipeline_video_mediapipe(
    #     'C:\\Users\\cs_li\\Documents\\WXWork\\1688854406374298\\Cache\\Video\\2022-10\\3.mp4'
    # )
    # pipeline_video_mediapipe_VideoProcessor(
    #     'C:\\Users\\cs_li\\Documents\\WXWork\\1688854406374298\\Cache\\Video\\2022-10\\3.mp4'
    # )
    pipeline_video_mediapipe_VideoProcessor(
        'C:\\Users\\cs_li\\Documents\\大五人格访谈视频+简短问卷\\cy.mp4'
    )
