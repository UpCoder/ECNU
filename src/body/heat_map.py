import cv2
import time
import math

class body_heat_map(object):
    def __init__(self, image_size, size=20):
        self.height = image_size[0]
        self.width = image_size[1]
        self.size = size
        self.map = {}
        # 左右肩、左右肘、左右手腕
        self.hit_body = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]
        self.shoulder = ["LEFT_SHOULDER", "RIGHT_SHOULDER"]
        self.elbow = ["LEFT_ELBOW", "RIGHT_ELBOW"]
        self.wrist = ["LEFT_WRIST", "RIGHT_WRIST"]

        # debug画图用的字体和字体大小
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_size = 0.6
        self.alpha = 0.7

    def draw_image(self, frame, frame_cnt):
        heat_image = frame.copy()
        # cv2.rectangle(heat_image, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        for xy_name in self.map.keys():
            x_coord, y_coord = xy_name.split('_')
            x_coord = int(x_coord)
            y_coord = int(y_coord)
            for body_name in self.map[xy_name].keys():
                if body_name in self.shoulder:
                    cnt = self.map[xy_name][body_name]
                    cv2.rectangle(heat_image, (x_coord * self.size, y_coord * self.size),
                                  ((x_coord + 1) * self.size, (y_coord + 1) * self.size),
                                  (0, 0, min(int(cnt / frame_cnt * 255), 255)), -1)
                if body_name in self.elbow:
                    cnt = self.map[xy_name][body_name]
                    cv2.rectangle(heat_image, (x_coord * self.size, y_coord * self.size),
                                  ((x_coord + 1) * self.size, (y_coord + 1) * self.size),
                                  (0, min(int(cnt / frame_cnt * 255), 255), 0), -1)
                if body_name in self.wrist:
                    cnt = self.map[xy_name][body_name]
                    cv2.rectangle(heat_image, (x_coord * self.size, y_coord * self.size),
                                  ((x_coord + 1) * self.size, (y_coord + 1) * self.size),
                                  (min(int(cnt / frame_cnt * 255), 255), 0, 0), -1)
        return heat_image

    def process(self, frame, body_coords, frame_cnt, draw=False):
        frame_cnt = min(100, frame_cnt)

        # body六个关键点转化成网格坐标，写入self.map
        for body_name in body_coords:
            if body_name not in self.hit_body:
                continue
            coords = body_coords[body_name]
            if coords['x'] >= 0 and coords['y'] >= 0:
                x_coord, y_coord = math.floor(coords['x']/self.size), math.floor(coords['y']/self.size)
                xy_name = f"{x_coord}_{y_coord}"
                if xy_name not in self.map:
                    self.map[xy_name] = {}
                if body_name not in self.map[xy_name]:
                    self.map[xy_name][body_name] = 0
                self.map[xy_name][body_name] += 1

        if draw:
            heat_image = self.draw_image(frame, frame_cnt)
            cv2.addWeighted(heat_image, self.alpha, frame, 1 - self.alpha, 0, frame)
        return frame