import numpy as np
import cv2
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf


font = {
    "family": "serif",
    "weight": "normal",
    "size": "22",
    "serif": "DejaVu Sans",
}
matplotlib.rc("font", **font)
from pose_classification_kit.config import OPENPOSE_PATH
# OPENPOSE_PATH = Path("C:/") / "Program files" / "OpenPose"

try:
    # sys.path.append(str(OPENPOSE_PATH / "build" / "python" / "openpose" / "Release"))
    # releasePATH = OPENPOSE_PATH / "build" / "x64" / "Release"
    # binPATH = OPENPOSE_PATH / "build" / "bin"
    # modelsPATH = OPENPOSE_PATH / "models"
    # os.environ["PATH"] = (
    #     os.environ["PATH"] + ";" + str(releasePATH) + ";" + str(binPATH) + ";"
    # )
    # import pyopenpose as op
    dir_path = os.path.join(OPENPOSE_PATH, 'python')
    print(dir_path)
    sys.path.append(os.path.join(OPENPOSE_PATH, 'bin', 'python', 'openpose', 'Release'))
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
    OPENPOSE_MODELS_PATH = os.path.join(OPENPOSE_PATH, 'models')
    modelsPATH = OPENPOSE_MODELS_PATH
    import pyopenpose as op

    OPENPOSE_LOADED = True
    print("OpenPose ({}) loaded.".format(str(OPENPOSE_PATH)))
except:
    OPENPOSE_LOADED = False
    print("OpenPose ({}) loading failed.".format(str(OPENPOSE_PATH)))

try:
    import tensorflow as tf

    GPU_LIST = tf.config.experimental.list_physical_devices("GPU")
    if GPU_LIST:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in GPU_LIST:
                # Prevent Tensorflow to take all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(
                    len(GPU_LIST), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    TF_LOADED = True
except:
    TF_LOADED = False


def format_data(handKeypoints, hand_id: int):
    """Return the key points of the hand seen in the image (cf. videoSource).
    Args:
        hand_id (int): 0 -> Left hand | 1 -> Right hand
    Returns:
        np.ndarray((3,21),float): Coordinates x, y and the accuracy score for each 21 key points.
                                    None if the given hand is not detected.
    """
    openhand_format = None
    personID = 0

    nbrPersonDetected = handKeypoints.shape[1] if handKeypoints.ndim > 2 else 0
    handAccuaracyScore = 0.0
    if nbrPersonDetected > 0:
        handAccuaracyScore = handKeypoints[hand_id, personID].T[2].sum()
        handDetected = handAccuaracyScore > 1.0
        if handDetected:
            handKeypoints = handKeypoints[hand_id, personID]
            # Initialize with the length of the first segment of each fingers
            lengthFingers = [
                np.sqrt(
                    (handKeypoints[0, 0] - handKeypoints[i, 0]) ** 2
                    + (handKeypoints[0, 1] - handKeypoints[i, 1]) ** 2
                )
                for i in [1, 5, 9, 13, 17]
            ]
            for i in range(3):  # Add length of other segments of each fingers
                for j in range(len(lengthFingers)):
                    x = (
                        handKeypoints[1 + j * 4 + i + 1, 0]
                        - handKeypoints[1 + j * 4 + i, 0]
                    )
                    y = (
                        handKeypoints[1 + j * 4 + i + 1, 1]
                        - handKeypoints[1 + j * 4 + i, 1]
                    )
                    lengthFingers[j] += np.sqrt(x ** 2 + y ** 2)
            normMax = max(lengthFingers)

            handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
            handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]

            outputArray = np.array(
                [
                    (handKeypoints.T[0] - handCenterX) / normMax,
                    -(handKeypoints.T[1] - handCenterY) / normMax,
                    (handKeypoints.T[2]),
                ]
            )

            openhand_format = []
            for i in range(outputArray.shape[1]):
                openhand_format.append(outputArray[0, i])  # add x
                openhand_format.append(outputArray[1, i])  # add y
            openhand_format = np.array(openhand_format)

    return openhand_format, handAccuaracyScore


def getFPS(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def getFrameNumber(video) -> int:
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        frame = video.get(cv2.cv.CAP_PROP_FRAME_COUNT)
    else:
        frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(frame)


def getHeight(video) -> int:
    return int(video.get(4))


def getWidth(video) -> int:
    return int(video.get(3))


def create_plot(classifier_labels, prediction_probabilities, save_url):
    assert len(classifier_labels) == len(prediction_probabilities)
    fig, ax = plt.subplots(figsize=(4, 10))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.96, bottom=0.04)
    plt.box(on=None)
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    plt.tick_params(
        axis="y", direction="in", pad=-50, which="both", left=False, labelleft=True
    )
    ax.set_yticks(np.arange(len(prediction_probabilities)))
    ax.set_yticklabels(classifier_labels, ha="left")
    ax.barh(
        np.arange(len(prediction_probabilities)),
        prediction_probabilities,
        color="#9500ff",
    )
    fig.savefig(save_url, transparent=True, dpi=108, pad_inches=0.0)
    plt.close(fig)


def getLengthLimb(data, keypoint1: int, keypoint2: int):
    if data[keypoint1, 2] > 0.0 and data[keypoint2, 2] > 0:
        return np.linalg.norm([data[keypoint1, 0:2] - data[keypoint2, 0:2]])
    return 0


def getBodyData(datum, personID=0):

    outputArray = None
    accuaracyScore = 0.0
    if datum.poseKeypoints is not None and len(datum.poseKeypoints.shape) > 0:

        # Read body data
        outputArray = datum.poseKeypoints[personID]
        accuaracyScore = outputArray[:, 2].sum()

        # Find bouding box
        min_x, max_x = float("inf"), 0.0
        min_y, max_y = float("inf"), 0.0
        for keypoint in outputArray:
            if keypoint[2] > 0.0:  # If keypoint exists in image
                min_x = min(min_x, keypoint[0])
                max_x = max(max_x, keypoint[0])
                min_y = min(min_y, keypoint[1])
                max_y = max(max_y, keypoint[1])

        # Centering
        np.subtract(
            outputArray[:, 0],
            (min_x + max_x) / 2,
            where=outputArray[:, 2] > 0.0,
            out=outputArray[:, 0],
        )
        np.subtract(
            (min_y + max_y) / 2,
            outputArray[:, 1],
            where=outputArray[:, 2] > 0.0,
            out=outputArray[:, 1],
        )

        # Scaling
        normalizedPartsLength = np.array(
            [
                getLengthLimb(outputArray, 1, 8) * (16.0 / 5.2),  # Torso
                getLengthLimb(outputArray, 0, 1) * (16.0 / 2.5),  # Neck
                getLengthLimb(outputArray, 9, 10) * (16.0 / 3.6),  # Right thigh
                getLengthLimb(outputArray, 10, 11)
                * (16.0 / 3.5),  # Right lower leg
                getLengthLimb(outputArray, 12, 13) * (16.0 / 3.6),  # Left thigh
                getLengthLimb(outputArray, 13, 14) * (16.0 / 3.5),  # Left lower leg
                getLengthLimb(outputArray, 2, 5) * (16.0 / 3.4),  # Shoulders
            ]
        )

        # Mean of non-zero values
        normalizedPartsLength = normalizedPartsLength[normalizedPartsLength > 0.0]
        if len(normalizedPartsLength) > 0:
            scaleFactor = np.mean(normalizedPartsLength)
        else:
            # print("Scaling error")
            return None, 0.0

        np.divide(outputArray[:, 0:2], scaleFactor, out=outputArray[:, 0:2])

        if np.any((outputArray > 1.0) | (outputArray < -1.0)):
            # print("Scaling error")
            return None, 0.0

        outputArray = outputArray.T

    return outputArray, accuaracyScore


def run_cam():
    current_path = 'C:\\Users\\30644\\.conda\\envs\\pose_kit\\Lib\\site-packages\\pose_classification_kit\\models\\' \
                   'Body\\9Class_3x64_BODY25'
    body_classifier = tf.keras.models.load_model(
        os.path.join(current_path, '9Classes_3x64_body25.h5')
    )
    # current_path = 'C:\\Users\\30644\\.conda\\envs\\pose_kit\\Lib\\site-packages\\pose_classification_kit\\models\\' \
    #                'Body\\9Class_3x64_BODY18'
    # body_classifier = tf.keras.models.load_model(
    #     os.path.join(current_path, '9Classes_3x64_body18.h5')
    # )
    # current_path = 'C:\\Users\\30644\\.conda\\envs\\pose_kit\\Lib\\site-packages\\pose_classification_kit\\models\\' \
    #                'Body\\20Class_CNN_BODY25'
    # body_classifier = tf.keras.models.load_model(
    #     os.path.join(current_path, 'CNN-2Conv1D-64x3filter-2dense-2x128_body25.h5')
    # )
    from pose_classification_kit.datasets.body_models import BODY25, BODY18, BODY18_FLAT, BODY25_FLAT, \
        BODY25_to_BODY18_indices
    modelInputShape = body_classifier.layers[0].input_shape[1:]
    if modelInputShape[0] == 25:
        currentBodyModel = BODY25
    elif modelInputShape[0] == 18:
        currentBodyModel = BODY18
    elif modelInputShape[0] == 50:
        currentBodyModel = BODY25_FLAT
    elif modelInputShape[0] == 36:
        currentBodyModel = BODY18_FLAT
    else:
        currentBodyModel = None
    print(body_classifier.layers[0].input_shape)
    body_classifiers = (body_classifier, )

    # if os.path.isfile(classifier_path / "class.txt"):
    #     with open(classifier_path / "class.txt", "r") as file:
    #         first_line = file.readline()
    #         classifier_labels = first_line.split(",")
    """
    就坐的
    支架_右臂升起
    站立
    T
    树_左
    树_右
    向上致敬
    Warrior2_左
    Warrio2_右
    """
    classifier_labels = ["Seated", "Stand_RightArmRaised", "standing", "T", "Tree_left", "Tree_right", "UpwardSalute",
                         "Warrior2_left", "Warrior2_right"]
    for i in range(len(classifier_labels)):
        classifier_labels[i] = classifier_labels[i].replace("_", " ")

    # Load OpenPose
    params = dict()
    params["model_folder"] = str(modelsPATH)
    params["face"] = True
    params["hand"] = True
    params["disable_multi_thread"] = False
    netRes = 8  # Default 22
    params["net_resolution"] = "-1x" + str(16 * netRes)

    opWrapper = op.WrapperPython()
    datum = op.Datum()
    opWrapper.configure(params)
    opWrapper.start()

    # Analyse video
    from src.sensor.camera import Camera
    import time
    print('step: 0')
    camera = Camera()
    camera.set_size(640, 480)
    print('step: 1')

    while camera.video_capure.isOpened():
        start_time = time.time()
        ret_flag, frame = camera.video_capure.read()
        # print('camera caption cost:', time.time() - start_time)
        im_row = frame[:480, 150:150 + 340]

        im_rd = im_row.copy()
        im_rd1 = im_row.copy()
        im_rd2 = im_row.copy()
        im_rd3 = im_row.copy()

        hand_id = 0


        # OpenPose analysis
        if type(frame) != type(None):
            datum.cvInputData = frame
            # opWrapper.emplaceAndPop([datum])
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # op.VectorDatum([datum])
            frame = datum.cvOutputData
        else:
            break
        wrists_positions = [(0, 0), (0, 0)]
        if datum.poseKeypoints is None:
            print('datum.poseKeypoints is None')
            continue
        if datum.poseKeypoints.ndim > 1:
            body_keypoints = np.array(datum.poseKeypoints[0])
            wrists_positions = [
                (body_keypoints[0][0], body_keypoints[0][1]),
                (body_keypoints[0][0], body_keypoints[0][1]),
            ]
            bodyKeypoints, _ = getBodyData(datum, 0)
        if bodyKeypoints is None:
            print('bodyKeypoints is None, continue')
            continue
        print('body_keypoints shape: ', bodyKeypoints.shape)
        if currentBodyModel == BODY25:
            inputData = bodyKeypoints[:2].T
        elif currentBodyModel == BODY25_FLAT:
            inputData = np.concatenate(bodyKeypoints[:2].T, axis=0)
        elif currentBodyModel == BODY18:
            inputData = bodyKeypoints.T[BODY25_to_BODY18_indices][:, :2]
        elif currentBodyModel == BODY18_FLAT:
            inputData = np.concatenate(bodyKeypoints.T[BODY25_to_BODY18_indices][:, :2], axis=0)
        else:
            raise ValueError('do not support')
        try:
            prediction = body_classifier.predict(np.asarray([inputData], np.float))[0]
        except Exception as e:
            print(e)
            prediction = None
        currentPrediction = classifier_labels[np.argmax(prediction)]
        print(f'Prob: {prediction}')
        print(f'currentPrediction: {currentPrediction}')

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        thickness = 2
        color = (255, 0, 149)
        (label_width, label_height), baseline = cv2.getTextSize(
            currentPrediction, font, scale, thickness
        )
        txt_position = tuple(
            map(
                lambda i, j: int(i - j),
                wrists_positions[hand_id],
                (label_width + 80, 70),
            )
        )
        cv2.putText(
            frame,
            currentPrediction,
            txt_position,
            font,
            scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

        # Display image
        cv2.imshow("frame", frame)

        # Write image
        # video_out.write(frame)

        # Create probabilities barchart
        # create_plot(
        #     classifier_labels[:-1],
        #     prediction_probabilities[:-1],
        #     barchart_out_path / "{}.png".format(frame_index),
        # )

        # cv2.imshow("yolo", body_yolo.processing_frame_result['annotation_image'])
        # cv2.imshow("mp", body_mp.processing_frame_result['annotation_image'])
        k = cv2.waitKey(1)

        # print('Single frame cost:', time.time() - start_time)
        # print('######' * 5)
        if k == ord('q'):
            break
    camera.video_capure.release()
    cv2.destroyAllWindows()


if __name__ == "__main__" and OPENPOSE_LOADED:
    # run()
    run_cam()