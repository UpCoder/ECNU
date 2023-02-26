import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import threading
import math
import time
from multiprocessing import Process, Queue


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
weigths = torch.load('E:\\PycharmProjects\\ECNU\\demo\\body_pix\\yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)
    # model.to(device)
'''
keep same with mediapipe pose
NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
'''
yolov7_skeleton_keypoint_names = [
        'NOSE',
        'LEFT_EYE',
        'RIGHT_EYE',
        'LEFT_EAR',
        'RIGHT_EAR',
        'LEFT_SHOULDER',
        'RIGHT_SHOULDER',
        'LEFT_ELBOW',
        'RIGHT_ELBOW',
        'LEFT_WRIST',
        'RIGHT_WRIST',
        'LEFT_HIP',
        'RIGHT_HIP',
        'LEFT_KNEE',
        'RIGHT_KNEE',
        'LEFT_ANKLE',
        'RIGHT_ANKLE'
    ]


def processing_pose_frame(frame, annotation_image, key_point_names, connections):
    s_time = time.time()
    # image = letterbox(frame, 960, stride=64, auto=True)[0]
    origin_shape = np.shape(frame)[:2]
    processing_shape = (640, 640)
    x_ratio = origin_shape[0] / processing_shape[0]
    y_ratio = origin_shape[1] / processing_shape[1]
    image = cv2.resize(frame, processing_shape)
    # annotation_image = cv2.resize(annotation_image, processing_shape)
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)
    output, _ = model(image)

    s_time_nms = time.time()
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    # nimg = image[0].permute(1, 2, 0) * 255
    # nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = np.asarray(annotation_image).astype(np.uint8)
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    # print('output shape: ', output.shape)
    # if output.shape[0] > 1:
    #     raise ValueError('do not support batch > 2')
    res_coords = {}
    for idx in range(output.shape[0]):
        coords_origin = output[idx, 7:].T
        coords_origin_list = []
        for idx_ in range(0, len(coords_origin), 3):
            coords_origin[idx_] = coords_origin[idx_] * y_ratio
            coords_origin[idx_+1] = coords_origin[idx_+1] * x_ratio
            coords_origin_list.append((coords_origin[idx_], coords_origin[idx_ + 1], coords_origin[idx_+2]))
        s_time_plot = time.time()
        plot_skeleton_kpts(nimg,
                           # output[idx, 7:].T,
                           coords_origin,
                           3,
                           np.shape(nimg),
                           draw_keypoints=key_point_names)
        # coords = np.reshape(output[idx, 7:].T, [-1, 3])
        # print(f'coords: {coords}')
        for key_point_name, coord_info in zip(yolov7_skeleton_keypoint_names, coords_origin_list):
            # x = coord_info[0] / processing_shape[1]
            # y = coord_info[1] / processing_shape[0]
            # x = coord_info[0] * y_ratio
            # y = coord_info[1] * x_ratio
            x = coord_info[0]
            y = coord_info[1]
            score = coord_info[2]
            if score > 0.5:
                res_coords[key_point_name] = {
                    'x': x,
                    'y': y,
                    'z': 0,
                    'score': score
                }
    annotation_image = np.asarray(nimg).astype(np.uint8)
    # nimg = cv2.resize(nimg, origin_shape)
    return annotation_image, res_coords, None


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None, draw_keypoints=[]):
    # print(f'kpts: {type(kpts)}, {kpts}, {np.shape(kpts)}')
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])


    # 1-‘nose’ 2-‘left_eye’ 3-‘right_eye’ 4-‘left_ear’ 5-‘right_ear’ 6-‘left_shoulder’ 7-‘right_shoulder’
    # 8-‘left_elbow’ -‘right_elbow’ 10-‘left_wrist’ 11-‘right_wrist’ 12-‘left_hip’ 13-‘right_hip’ 14-‘left_knee’
    # 15-‘right_knee’ 16-‘left_ankle’ 17-‘right_ankle’

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        if yolov7_skeleton_keypoint_names[kid - 1] not in draw_keypoints:
            continue
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        # print(sk_id, sk, len(yolov7_skeleton_keypoint_names))
        if yolov7_skeleton_keypoint_names[sk[0] - 1] not in draw_keypoints or \
                yolov7_skeleton_keypoint_names[sk[1] - 1] not in draw_keypoints:
            continue
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % orig_shape[0] == 0 or pos1[1] % orig_shape[1] == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % orig_shape[0] == 0 or pos2[1] % orig_shape[1] == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def output_to_keypoint(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.detach().cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.detach().cpu().numpy()[index])])
    return np.array(targets)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression_kpt(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), kpt_label=False, nc=None, nkpt=None):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if nc is None:
        nc = prediction.shape[2] - 5 if not kpt_label else prediction.shape[2] - 56 # number of classes
    # xc = prediction[..., 4] > conf_thres  # candidates
    xc = prediction[:, :, 4] > conf_thres

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # a = x[:, 5:5+nc]
        # b = x[:, 4:5]
        # x[:, 5:5+nc] = a * b  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            if not kpt_label:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            else:
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]


        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class Camera(threading.Thread):
    __slots__ = ['camera', 'Flag', 'count', 'width', 'heigth', 'frame']

    def __init__(self):
        threading.Thread.__init__(self)
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.Flag = 0
        self.count = 1
        self.width = 480
        self.heigth = 340
        self.name = ''
        self.path = ''
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heigth)
        # for i in range(46):
        # print("No.={} parameter={}".format(i,self.camera.get(i)))

    def run(self):
        while True:
            ret, self.frame = self.camera.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
            self.frame = cv2.flip(self.frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
            if self.Flag == 1:
                print("拍照")
                if self.name == '' and self.path == '':
                    cv2.imwrite(str(self.count) + '.jpg', self.frame)  # 将画面写入到文件中生成一张图片
                elif self.name != '':
                    cv2.imwrite(self.name + '.jpg', self.frame)
                self.count += 1
                self.Flag = 0
            if self.Flag == 2:
                print("退出")
                self.camera.release()  # 释放内存空间
                cv2.destroyAllWindows()  # 删除窗口
                break

    def take_photo(self):
        self.Flag = 1

    def exit_program(self):
        self.Flag = 2

    def set_name(self, str):
        self.name = str

    def set_path(self, str):
        self.path = str


def show_window(cap):
    while True:
        cv2.namedWindow("window", 1)  # 1代表外置摄像头
        cv2.resizeWindow("window", cap.width, cap.heigth)  # 指定显示窗口大小
        image = cap.frame
        # image = letterbox(image, (512, 320), stride=64, auto=True)[0]
        # image = transforms.ToTensor()(image)
        # image = torch.tensor(np.array([image.numpy()]))
        #
        # if torch.cuda.is_available():
        #     image = image.half().to(device)
        # output, _ = model(image)
        #
        # output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
        #                                  kpt_label=True)
        # with torch.no_grad():
        #     output = output_to_keypoint(output)
        # nimg = image[0].permute(1, 2, 0) * 255
        # nimg = nimg.cpu().numpy().astype(np.uint8)
        # nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        # for idx in range(output.shape[0]):
        #     plot_skeleton_kpts(nimg, output[idx, 7:].T, 3, nimg.shape)

        keypoint_names = ['LEFT_SHOULDER',
                          'RIGHT_SHOULDER',
                          'LEFT_ELBOW',  # 肘部
                          'RIGHT_ELBOW',  # 肘部
                          'LEFT_WRIST',  # 手腕
                          'RIGHT_WRIST'  # 手腕
                          ]
        nimg, _, _ = processing_pose_frame(cap.frame, cap.frame, keypoint_names, None)
        # cv2.imshow('window', cap.frame)
        cv2.imshow('window', nimg)
        c = cv2.waitKey(50)  # 按ESC退出画面
        if c == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    cap = Camera()
    cap.start()
    time.sleep(5)
    # from tqdm import tqdm
    # for idx in tqdm(range(10000)):
    #     image = cap.frame
    #     image = letterbox(image, 960, stride=64, auto=True)[0]
    #     image_ = image.copy()
    #     image = transforms.ToTensor()(image)
    #     image = torch.tensor(np.array([image.numpy()]))
    #
    #     if torch.cuda.is_available():
    #         image = image.half().to(device)
    #     output, _ = model(image)
    #     print('nc: ', model.yaml['nc'])
    #     print('nkpt: ', model.yaml['nkpt'])
    #     output = non_max_suppression_kpt(output, 0.25, 0.65, nc=1, nkpt=17,
    #                                      kpt_label=True)
    print('show window start')
    thread = threading.Thread(target=show_window, args=(cap, ))
    thread.start()
    time.sleep(100000)


