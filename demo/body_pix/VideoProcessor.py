import cv2
import numpy as np
from tqdm import tqdm

from handler_hand import processing_frame as processing_hand_frame
from handler_pose import processing_frame as processing_pose_frame

HOLDING = 'Hold'
UPING = 'Up'
DOWNING = 'Down'
LEFTING = 'Left'
RIGHTING = 'Right'
FORWARDING = 'Forward'
BACKWARDING = 'Backward'


def handler_move(infos, window_size, last_move):
    def __mean_infos(infos_):
        return np.mean([info_['x'] for info_ in infos_]), \
               np.mean([info_['y'] for info_ in infos_]), \
               np.mean([info_['z'] for info_ in infos_])

    if len(infos) % window_size == 0:
        # 说明此时才开始判断
        info_size = len(infos)
        if info_size - window_size == 0:
            return last_move
        cur_batch = infos[info_size - window_size:-1]
        last_bacth = infos[info_size - 2 * window_size: info_size:window_size]
        cur_x, cur_y, cur_z = __mean_infos(cur_batch)
        last_x, last_y, last_z = __mean_infos(last_bacth)
        rl_res = RIGHTING if cur_x > last_x else HOLDING if cur_x == last_y else LEFTING
        ud_res = DOWNING if cur_y > last_y else HOLDING if cur_y == last_y else UPING
        fb_res = FORWARDING if cur_z > last_z else HOLDING if cur_z == last_z else BACKWARDING
        return rl_res, ud_res, fb_res
    return last_move


class HandProcessor(object):
    def __init__(self, window_size=16):
        self.key_point_names = [
            'INDEX_FINGER_PIP',
            'INDEX_FINGER_DIP',
            'MIDDLE_FINGER_PIP',
            'MIDDLE_FINGER_DIP',
            'PINKY_PIP',
            'PINKY_DIP',
        ]
        self.cur_left_hand = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.cur_right_hand = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.left_hand_infos = []
        self.right_hand_infos = []
        self.window_size = window_size
        self.left_hand_last_move = (HOLDING, HOLDING, HOLDING)
        self.right_hand_last_move = (HOLDING, HOLDING, HOLDING)
        self.last_hand_result = None

    def process(self, frame, annotation_image=None, refuse=False):
        if refuse and self.last_hand_result is not None:
            annotation_image, left_coord_infos, right_coord_infos, hand_result = processing_hand_frame(
                image=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names,
                hand_result=self.last_hand_result
            )
        else:
            annotation_image, left_coord_infos, right_coord_infos, hand_result = processing_hand_frame(
                image=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names
            )
            self.last_hand_result = hand_result
        self.cur_left_hand = self.handler_hand_info_by_keypoints(left_coord_infos)
        self.cur_right_hand = self.handler_hand_info_by_keypoints(right_coord_infos)
        self.left_hand_infos.append(self.cur_left_hand)
        self.right_hand_infos.append(self.cur_right_hand)
        self.left_hand_last_move = handler_move(self.left_hand_infos, self.window_size, self.left_hand_last_move)
        self.right_hand_last_move = handler_move(self.right_hand_infos, self.window_size, self.right_hand_last_move)
        return annotation_image, self.left_hand_last_move, self.right_hand_last_move

    def handler_hand_info_by_keypoints(self, coord_infos):
        if coord_infos is None:
            return {
                'x': -1,
                'y': -1,
                'z': -1
            }
        xs = [
            coord_infos.get(key_point_name)['x'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        ys = [
            coord_infos.get(key_point_name)['y'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        zs = [
            coord_infos.get(key_point_name)['z'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        return {
            'x': np.mean(xs),
            'y': np.mean(ys),
            'z': np.mean(zs)
        }

    def reset(self):
        self.cur_left_hand = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.cur_right_hand = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.left_hand_infos = []
        self.right_hand_infos = []
        self.left_hand_last_move = (HOLDING, HOLDING, HOLDING)
        self.right_hand_last_move = (HOLDING, HOLDING, HOLDING)


class BodyProcessor(object):
    def __init__(self, window_size=16):
        self.key_point_names = [
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER'
        ]
        self.cur_body_info = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.body_infos = []
        self.window_size = window_size
        self.body_last_move = (HOLDING, HOLDING, HOLDING)
        self.last_body_result = None

    def process(self, frame, annotation_image, refuse=False):
        if refuse and self.last_body_result is not None:
            annotation_image, cur_body_coords, body_result = processing_pose_frame(
                frame=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names,
                pose_result=self.last_body_result
            )
        else:
            annotation_image, cur_body_coords, body_result = processing_pose_frame(
                frame=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names
            )
            self.last_body_result = body_result
        self.cur_body_info = self.handler_body_info_by_keypoints(cur_body_coords)
        self.body_infos.append(self.cur_body_info)
        self.body_last_move = handler_move(self.body_infos, self.window_size, self.body_last_move)
        return annotation_image, self.body_last_move

    def handler_body_info_by_keypoints(self, coord_infos):
        if coord_infos is None:
            return {
                'x': -1,
                'y': -1,
                'z': -1
            }
        xs = [
            coord_infos.get(key_point_name)['x'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        ys = [
            coord_infos.get(key_point_name)['y'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        zs = [
            coord_infos.get(key_point_name)['z'] if coord_infos.get(key_point_name) is not None else -1
            for key_point_name in self.key_point_names
        ]
        return {
            'x': np.mean(xs),
            'y': np.mean(ys),
            'z': np.mean(zs)
        }

    def reset(self):
        self.cur_body_info = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.body_infos = []
        self.body_last_move = (HOLDING, HOLDING, HOLDING)


class VideoProcessor(object):
    def __init__(self, window_size=16, txt_loc=(10, 60),
                 line_width=30, calc_frame_interval=4):
        self.body_processor = BodyProcessor(window_size=window_size)
        self.hand_processor = HandProcessor(window_size=window_size)
        self.txt_loc = txt_loc
        self.line_width = line_width
        self.count_idx = 0
        self.calc_frame_interval = calc_frame_interval

    def processing_frame(self, frame_image, annotation_image):
        if self.count_idx % self.calc_frame_interval == 0:
            refuse = False
        else:
            refuse = True
        self.count_idx += 1
        if annotation_image is None:
            annotation_image = frame_image.copy()

        annotation_image, (body_lr_str, body_ud_str, body_fb_str) = self.body_processor.process(frame_image,
                                                                                                annotation_image,
                                                                                                refuse=refuse)
        annotation_image, (lh_lr_str, lh_ud_str, lh_fb_str), \
        (rh_lr_str, rh_ud_str, rh_fb_str) = self.hand_processor.process(frame=frame_image,
                                                                        annotation_image=annotation_image,
                                                                        refuse=refuse)
        annotation_image_txt = annotation_image.copy()
        annotation_image_txt = self.put_txts(annotation_image_txt, [
            f'Body Left/Right: {body_lr_str}',
            f'Body Up/Down: {body_ud_str}',
            f'Body Forward/Backward: {body_fb_str}',
            f'Left Hand Left/Right: {lh_lr_str}',
            f'Left Hand Up/Down: {lh_ud_str}',
            f'Right Hand Left/Right: {rh_lr_str}',
            f'Right Hand Up/Down: {rh_ud_str}'
        ], self.txt_loc, self.line_width)
        return annotation_image, annotation_image_txt

    def put_txts(self, frame, txts, start_loc, line_width, font=cv2.FONT_HERSHEY_SIMPLEX):
        for idx, txt in enumerate(txts):
            cur_x = start_loc[0]
            cur_y = start_loc[1] + line_width * idx
            cv2.putText(frame, txt, (cur_x, cur_y), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

    def processing_frames(self, frames):
        annotation_frames = []
        annotation_frames_txt = []
        for frame in tqdm(frames):
            frame_res = self.processing_frame(frame, None)
            annotation_frames.append(frame_res[0])
            annotation_frames_txt.append(frame_res[1])
        return annotation_frames, annotation_frames_txt

    def reset(self):
        """
        重置
        :return:
        """
        self.hand_processor.reset()
        self.body_processor.reset()
        self.count_idx = 0


if __name__ == '__main__':
    pass
