import dlib

detector = dlib.get_frontal_face_detector()

def detect(image):
    faces = detector(image, 0)
    return faces