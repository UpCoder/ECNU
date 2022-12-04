import cv2
import time
import numpy as np
from tqdm import tqdm

from src.body.handler_hand import processing_frame as processing_hand_frame
from src.body.handler_pose import processing_frame as processing_pose_frame

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


class BodyInfoSingle(object):
    def __init__(self):
        self.body_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.left_elbow_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.right_elbow_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.left_wrist_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.right_wrist_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.left_shoulder_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.right_shoulder_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.left_arm_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }
        self.right_arm_coord = {
            'x': 0.,
            'y': 0.,
            'z': 0.
        }

    @classmethod
    def build(cls, coords):
        obj = cls()
        obj.left_shoulder_coord = {
            'x': np.mean([coord_info['LEFT_SHOULDER']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['LEFT_SHOULDER']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['LEFT_SHOULDER']['z'] for coord_info in coords]),
        }
        obj.right_shoulder_coord = {
            'x': np.mean([coord_info['RIGHT_SHOULDER']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['RIGHT_SHOULDER']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['RIGHT_SHOULDER']['z'] for coord_info in coords]),
        }
        obj.left_wrist_coord = {
            'x': np.mean([coord_info['LEFT_WRIST']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['LEFT_WRIST']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['LEFT_WRIST']['z'] for coord_info in coords]),
        }
        obj.right_wrist_coord = {
            'x': np.mean([coord_info['RIGHT_WRIST']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['RIGHT_WRIST']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['RIGHT_WRIST']['z'] for coord_info in coords]),
        }
        obj.left_elbow_coord = {
            'x': np.mean([coord_info['LEFT_ELBOW']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['LEFT_ELBOW']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['LEFT_ELBOW']['z'] for coord_info in coords]),
        }
        obj.right_elbow_coord = {
            'x': np.mean([coord_info['RIGHT_ELBOW']['x'] for coord_info in coords]),
            'y': np.mean([coord_info['RIGHT_ELBOW']['y'] for coord_info in coords]),
            'z': np.mean([coord_info['RIGHT_ELBOW']['z'] for coord_info in coords]),
        }
        obj.body_coord = {
            'x': np.mean([obj.left_shoulder_coord['x'], obj.right_shoulder_coord['x']]),
            'y': np.mean([obj.left_shoulder_coord['y'], obj.right_shoulder_coord['y']]),
            'z': np.mean([obj.left_shoulder_coord['z'], obj.right_shoulder_coord['z']]),
        }
        obj.left_arm_coord = {
            'x': np.mean([obj.left_wrist_coord['x'], obj.left_elbow_coord['x']]),
            'y': np.mean([obj.left_wrist_coord['y'], obj.left_elbow_coord['y']]),
            'z': np.mean([obj.left_wrist_coord['z'], obj.left_elbow_coord['z']]),
        }
        obj.right_arm_coord = {
            'x': np.mean([obj.right_wrist_coord['x'], obj.right_elbow_coord['x']]),
            'y': np.mean([obj.right_wrist_coord['y'], obj.right_elbow_coord['y']]),
            'z': np.mean([obj.right_wrist_coord['z'], obj.right_elbow_coord['z']]),
        }
        return obj


class BodyInfo(object):
    def __init__(self, key_point_names, window_size=5):
        self.coords = []
        self.body_info = []
        self.body_info_window_avg = [BodyInfoSingle()]
        self.key_point_names = key_point_names
        self.body_swing_count = 0   # 身体摆动次数
        self.left_arm_swing_count = 0   # 左胳膊摆动次数
        self.right_arm_swing_count = 0  # 右胳膊摆动次数
        self.window_size = window_size

    def add_info(self, coord_info):
        self.coords.append(coord_info)
        self.body_info.append(BodyInfoSingle.build([coord_info]))
        if len(self.coords) % self.window_size == 0:
            self.body_info_window_avg.append(BodyInfoSingle.build(
                self.coords[len(self.coords) - self.window_size:]
            ))
            print(f'add body_info windows avg: {self.body_info_window_avg[-1].body_coord}')
    def merge(self, left_rights, distances):
        """
        将[0, 0, 0, 1, 1, 1, 0, 0] => [0, 1, 0]
        并且将对应的距离相加
        """
        left_right_merge = [left_rights[0]]
        distance_merge = [distances[0]]
        for left_right, distance in zip(left_rights[1:], distances[1:]):
            if left_right == left_right_merge[-1]:
                distance_merge[-1] += distance
            else:
                left_right_merge.append(left_right)
                distance_merge.append(distance)
        return left_right_merge, distance_merge

    def calc_metrics(self):
        if len(self.body_info_window_avg) < 2:
            return {
                'body_swing': str(0),
                'body_arm_swing': str(0),
                'body_tension': '{:.5f}'.format(0.),
                'body_distance': '{:.5f}'.format(0)
            }
        axis_name = 'x'
        last_info = self.body_info_window_avg[0]
        body_left_right = []    # 身体向左移动还是向右移动
        body_distances = []     # 身体移动的距离
        left_arm_left_right = []    # 左胳膊向左移动还是向右移动
        left_arm_distance = []  # 左胳膊移动的距离
        right_arm_left_right = []   # 右胳膊向左移动还是向右移动
        right_arm_distance = []     # 右胳膊移动的距离
        body_width_avg = [abs(last_info.left_shoulder_coord[axis_name] - last_info.right_shoulder_coord[axis_name])]
        for cur_body_info in self.body_info_window_avg[1:]:
            body_width_avg.append(abs(cur_body_info.left_shoulder_coord[axis_name] -
                                      cur_body_info.right_shoulder_coord[axis_name]))
            if cur_body_info.body_coord[axis_name] > last_info.body_coord[axis_name]:
                body_left_right.append(1)
            else:
                body_left_right.append(0)
            body_distances.append(abs(cur_body_info.body_coord[axis_name] - last_info.body_coord[axis_name]))

            if cur_body_info.left_arm_coord[axis_name] > last_info.left_arm_coord[axis_name]:
                left_arm_left_right.append(1)
            else:
                left_arm_left_right.append(0)
            left_arm_distance.append(abs(cur_body_info.left_arm_coord[axis_name] - last_info.left_arm_coord[axis_name]))

            if cur_body_info.right_arm_coord[axis_name] > last_info.right_arm_coord[axis_name]:
                right_arm_left_right.append(1)
            else:
                right_arm_left_right.append(0)
            right_arm_distance.append(abs(cur_body_info.right_arm_coord[axis_name] -
                                          last_info.right_arm_coord[axis_name]))
            last_info = cur_body_info
        body_swing_threshold = np.mean(body_width_avg) * 0.2
        print(f'body_swing_threshold: {body_swing_threshold}')

        left_arm_swing_threshold = np.mean(body_width_avg) * 0.1
        right_arm_swing_threshold = np.mean(body_width_avg) * 0.1

        # 计算身体摆动的次数
        print(f'body left/right: {body_left_right}')
        print(f'body distances: {body_distances}')
        body_left_right_merge, body_distances_merge = self.merge(body_left_right, body_distances)
        print(f'merge body left/right: {body_left_right_merge}')
        print(f'merge body distances: {body_distances_merge}')
        count_body_swing = 0
        for idx in range(1, len(body_left_right_merge), 2):
            move_distance = max(body_distances_merge[idx-1], body_distances_merge[idx])
            if move_distance >= body_swing_threshold:
                count_body_swing += 1

        # 计算左胳膊摆动的次数
        print(f'left arm left/right: {left_arm_left_right}')
        print(f'left arm distances: {left_arm_distance}')
        left_arm_left_right_merge, left_arm_distances_merge = self.merge(left_arm_left_right, left_arm_distance)
        count_left_arm_swing = 0
        for idx in range(1, len(left_arm_left_right_merge), 2):
            move_distance = max(left_arm_distances_merge[idx - 1], left_arm_distances_merge[idx])
            if move_distance >= left_arm_swing_threshold:
                count_left_arm_swing += 1

        # 计算右胳膊摆动的次数
        print(f'right arm left/right: {right_arm_left_right}')
        print(f'right arm distances: {right_arm_distance}')
        right_arm_left_right_merge, right_arm_distances_merge = self.merge(right_arm_left_right, right_arm_distance)
        print(f'merge right arm left/right: {right_arm_left_right_merge}')
        print(f'merge right arm distances: {right_arm_distances_merge}')
        count_right_arm_swing = 0
        for idx in range(1, len(right_arm_left_right_merge), 2):
            move_distance = max(right_arm_distances_merge[idx - 1], right_arm_distances_merge[idx])
            if move_distance >= right_arm_swing_threshold:
                count_right_arm_swing += 1

        return {
            'body_swing': str(count_body_swing),
            'body_arm_swing': str(count_left_arm_swing + count_right_arm_swing),
            'body_tension': '{:.5f}'.format(0.),
            'body_distance': '{:.5f}'.format(np.sum(body_distances_merge))
        }


class BodyProcessor(object):
    def __init__(self, window_size=10):
        self.key_point_names = [
            'LEFT_SHOULDER',
            'RIGHT_SHOULDER',
            'LEFT_ELBOW',   # 肘部
            'RIGHT_ELBOW',  # 肘部
            'LEFT_WRIST',   # 手腕
            'RIGHT_WRIST'   # 手腕
        ]
        self.key_point_connections = frozenset(
            [
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16)
            ]
        )
        self.cur_body_info = {
            'x': 0,
            'y': 0,
            'z': 0
        }
        self.body_infos = []
        self.body_infos_obj = BodyInfo(self.key_point_names, window_size=window_size)
        self.window_size = window_size
        self.body_last_move = (HOLDING, HOLDING, HOLDING)
        self.last_body_result = None

    def process(self, frame, annotation_image, refuse=False):
        if refuse and self.last_body_result is not None:
            annotation_image, cur_body_coords, body_result = processing_pose_frame(
                frame=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names,
                pose_result=self.last_body_result,
                connections=self.key_point_connections
            )
        else:
            annotation_image, cur_body_coords, body_result = processing_pose_frame(
                frame=frame,
                annotation_image=annotation_image,
                coord_names=self.key_point_names,
                connections=self.key_point_connections
            )
            self.last_body_result = body_result
        for keypoint in self.key_point_names:
            if cur_body_coords.get(keypoint, None) is None:
                cur_body_coords[keypoint] = {
                    'x': 0.,
                    'y': 0,
                    'z': 0
                }
        self.body_infos_obj.add_info(cur_body_coords)
        # self.body_infos.coords.append(cur_body_coords)
        self.cur_body_info = self.handler_body_info_by_keypoints(cur_body_coords)
        self.body_infos.append(self.cur_body_info)
        self.body_last_move = handler_move(self.body_infos, self.window_size, self.body_last_move)
        return annotation_image, self.body_last_move, self.body_infos_obj.calc_metrics()

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

    def processing_frame(self, frame_image, annotation_image, need_info=False):
        if self.count_idx % self.calc_frame_interval == 0:
            refuse = False
        else:
            refuse = True
        refuse = False
        self.count_idx += 1
        if annotation_image is None:
            annotation_image = frame_image.copy()

        annotation_image, (body_lr_str, body_ud_str, body_fb_str), body_info_metrics = self.body_processor.process(
            frame_image,
            annotation_image,
            refuse=refuse)
        annotation_image, (lh_lr_str, lh_ud_str, lh_fb_str), \
        (rh_lr_str, rh_ud_str, rh_fb_str) = self.hand_processor.process(frame=frame_image,
                                                                        annotation_image=annotation_image,
                                                                        refuse=refuse)
        annotation_image_txt = annotation_image.copy()
        print_lines = [
            f'Body Left/Right: {body_lr_str}',
            f'Body Up/Down: {body_ud_str}',
            f'Body Forward/Backward: {body_fb_str}',
            f'Left Hand Left/Right: {lh_lr_str}',
            f'Left Hand Up/Down: {lh_ud_str}',
            f'Right Hand Left/Right: {rh_lr_str}',
            f'Right Hand Up/Down: {rh_ud_str}',
        ]
        print_lines.extend(
            [
                f'{key}: {value}'
                for key, value in body_info_metrics.items()
            ]
        )
        annotation_image_txt = self.put_txts(annotation_image_txt, print_lines, self.txt_loc, self.line_width)
        body_hand_metrics = {}
        if need_info:
            body_hand_metrics = {
                **body_info_metrics
            }
        else:
            pass
        return annotation_image, annotation_image_txt, body_hand_metrics

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
    from src.sensor.camera import Camera
    print('step: 0')
    camera = Camera()
    camera.set_size(640, 480)
    print('step: 1')
    body = VideoProcessor()

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, im_row = camera.video_capure.read()
        print('camera caption cost:', time.time() - start_time)
        im_row = im_row[:480, 150:150 + 340]

        im_rd = im_row.copy()
        im_rd1 = im_row.copy()
        print(im_rd.shape)

        image, image_with_metrics, body_info = body.processing_frame(im_rd, im_rd1)
        infos = {
            **body_info
        }
        send_time = time.time()

        cv2.imshow("demo", image_with_metrics)
        k = cv2.waitKey(1)

        print('Single frame cost:', time.time() - start_time)
        print('######' * 5)
        if k == ord('q'):
            break

    camera.video_capure.release()
    cv2.destroyAllWindows()