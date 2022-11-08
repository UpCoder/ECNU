
import numpy as np
import cv2
import time
import timeit
import matplotlib.pyplot as plt

from face_detect import detect
from face_landmarks import landmark, head_move
from face_emotion import emotion_predict

# dis_list = []
# def plot(move_dis):
#     global dis_list
#     plt.axis([0, 100, 0, 50])
#     dis_list.insert(0, move_dis)
#     dis_list = dis_list[:100]
#     plt.cla()
#     x= list(range(len(dis_list)))
#     plt.plot(x, dis_list)

cap = cv2.VideoCapture(0)


font = cv2.FONT_HERSHEY_SIMPLEX

last_landmarks = []

cnt = 1
while cap.isOpened():
    ret_flag, im_row = cap.read()
    # cv2.imwrite("demo_image/img1.jpg", im_row)

    im_rd = im_row.copy()
    im_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    # start point
    start = timeit.default_timer()

    faces = detect(im_gray)
    if len(faces) > 0:
        # emotion
        face = faces[0]
        crop = im_rd[face.top():face.bottom(), face.left():face.right()]
        start_time = time.time()
        emotion = emotion_predict(crop)
        print('emotion cost:', time.time()-start_time)
        cv2.putText(im_rd, "emotion: " + str(emotion), (10, 30), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        #landmark
        start_time = time.time()
        landmarks = landmark(im_rd, faces[0])
        print('landmark cost:', time.time()-start_time)
        for idx, point in enumerate(landmarks):
            # 68 点的坐标
            pos = (point[0, 0], point[0, 1])
            cv2.circle(im_rd, pos, 2, color=(250, 230, 230))

        if len(last_landmarks) > 0:
            move_dis = head_move(last_landmarks, landmarks)
            cv2.putText(im_rd, "head move: " + str(move_dis), (10, 60), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            # plot(move_dis)

        last_landmarks = landmarks

    else:
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)

    stop = timeit.default_timer()
    print("%-15s %f" % ("Time cost:", (stop - start)))

    # cv2.imshow("camera_row", img_rd)
    cv2.imshow("camera", im_rd)

    cnt += 1
    k = cv2.waitKey(1)


    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == '__main__':
    pass
