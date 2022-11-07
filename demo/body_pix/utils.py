import json

import cv2
import numpy as np
from tqdm import tqdm


def parse_video(video_path, target_fps=1):
    """
    extract video frames
    :param video_path: 视频路径
    :param fps: frame-per-second
    :return:
    """
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'origin, fps={fps}, total_frame={total_frame}')
    res = capture.set(cv2.CAP_PROP_FPS, int(target_fps))
    print(f'change fps result: {res}')
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'after, fps={fps}, total_frame={total_frame}')
    frames = []
    rate = fps // target_fps
    print(rate)
    for i in tqdm(range(int(total_frame))):
        ret = capture.grab()
        if not ret:
            print('Tag1')
            break
        if i % rate == 0:
            ret, frame = capture.read()
            if not ret:
                print('Tag2')
                break
            frames.append(frame)
    return frames


def write_videos(frames, fps, save_path='testwrite.avi'):

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(save_path, fourcc, fps, (1920, 1080), True)
    for frame in frames:
        out.write(frame)
    out.release()

COLORS = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 0],
]


def draw_keypoints(coord_points, target_shape, fps, coord_names):
    last_coord = {
        coord_name: {
            'x': -1,
            'y': -1,
            'z': -1
        }
        for coord_name in coord_names
    }
    coord_name2color = {
        coord_name: COLORS[idx]
        for idx, coord_name in enumerate(coord_names)
    }
    thickness = 10
    lineType = 8
    idx = 0
    frames = [np.zeros(target_shape)]
    while idx < len(coord_points):
        frame = frames[-1].copy()
        for coord_name in coord_names:
            print(coord_points[idx: idx + fps])
            cur_batch = [data.get(coord_name, None) for data in coord_points[idx: idx + fps]]
            xs, ys, zs = [], [], []
            for coord_point in cur_batch:
                if coord_point is None:
                    continue
                if coord_point['x'] > 0 and coord_point['y'] > 0:
                    xs.append(coord_point['x'])
                    ys.append(coord_point['y'])
                    zs.append(coord_point['z'])
            if len(xs) == 0:
                continue
            x_mean, y_mean, z_mean = int(np.mean(xs) * target_shape[1]), int(np.mean(ys) * target_shape[0]), int(np.mean(zs))
            if last_coord[coord_name]['x'] == -1 or last_coord[coord_name]['y'] == -1:
                print(x_mean, y_mean)
                cv2.line(frame, (x_mean, y_mean), (x_mean, y_mean), coord_name2color[coord_name], thickness, lineType)
            else:
                print(
                    (last_coord[coord_name]['x'], last_coord[coord_name]['y']),
                    (x_mean, y_mean)
                )
                frame = cv2.line(frame, (last_coord[coord_name]['x'], last_coord[coord_name]['y']),
                                 (x_mean, y_mean), coord_name2color[coord_name], thickness, lineType)
                print('test', frame.shape)

            last_coord[coord_name]['x'] = x_mean
            last_coord[coord_name]['y'] = y_mean
            last_coord[coord_name]['z'] = z_mean
        idx += fps
        print(f'{idx} / {len(coord_points)}, {np.sum(frame == 255)}')

        cv2.imwrite(f'tmp/{len(frames)}.jpg', frame)
        frames.append(frame)
    print(np.shape(frames))
    # print(np.sum(np.asarray(np.asarray(frames) == 255), axis=(1, 2, 3)))
    # print(np.sum(np.asarray(np.asarray(frames) == 255), axis=(1, 2, 3)).argmax())
    # slice_idx = np.sum(np.asarray(np.asarray(frames) == 255), axis=(1, 2, 3)).argmax()
    # cv2.imwrite('test.jpg', frames[slice_idx])
    # print(np.asarray(frames) == 255)
    write_videos(
        frames,
        fps=1,
        save_path='testwrite.avi'
    )


if __name__ == '__main__':
    # frames = parse_video(
    #     'C:\\Users\\cs_li\\Documents\\WXWork\\1688854406374298\\Cache\\Video\\2022-10\\'
    #     '3.mp4',
    #     1
    # )
    # print(f'frame count: {len(frames)}, shape: {np.shape(frames)}')

    # write_videos(frames, 1.0)

    draw_keypoints(
        json.load(open('shoulder_coords.json', 'r'))['coords'],
        [1080, 1920, 3],
        29 // 2,
        [
          'LEFT_SHOULDER',
          'RIGHT_SHOULDER'
        ]
    )
