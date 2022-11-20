import cv2

class Camera(object):
    def __init__(self, width=1280, height=960):
        self.width, self.height = width, height
        self.video_capure = cv2.VideoCapture(0)

    def find_camera(self):
        pass

    def set_size(self, width=1280, height=960):
        self.video_capure.set(3, width)
        self.video_capure.set(4, height)


