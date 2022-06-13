import cv2
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
import keras

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5)


def extract_handpoints(ori_frame):
    # input: 圖片, output: 21手部點的座標
    frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        keypoint_pos = []
        for i in landmarks:
            # Acquire x, y but don't forget to convert to integer.
            x = int(i.x * frame.shape[1])
            y = int(i.y * frame.shape[0])
            # Annotate landmarks or do whatever you want.
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            keypoint_pos.append((x, y))
        if len(keypoint_pos) not in [0, 21]:
            return [], ori_frame
        return keypoint_pos, frame
    else:
        return [], ori_frame


def sub_t(tuple_a, tuple_b):
    try:
        return tuple_a[0] - tuple_b[0], tuple_a[1] - tuple_b[1]
    except:
        return ()


def handpoints_to_vectors(handpoints):
    if len(handpoints) != 21:
        return []
    else:
        return [
            sub_t(handpoints[2], handpoints[0]),
            sub_t(handpoints[4], handpoints[2]),
            sub_t(handpoints[5], handpoints[0]),
            sub_t(handpoints[8], handpoints[5]),
            sub_t(handpoints[9], handpoints[0]),
            sub_t(handpoints[12], handpoints[9]),
            sub_t(handpoints[13], handpoints[0]),
            sub_t(handpoints[16], handpoints[13]),
            sub_t(handpoints[17], handpoints[0]),
            sub_t(handpoints[20], handpoints[17]),
        ]


def norm_vectors(vector_list):
    result = []
    for vec in vector_list:
        length = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        if length == 0:
            result.append((0, 0))
        else:
            result.append((
                vec[0] / length,
                vec[1] / length
            ))
    return result


def get_nn_input_from_img(frame):
    result = []
    for vec in norm_vectors(
            handpoints_to_vectors(
                extract_handpoints(frame)[0]
            )
    ):
        result.append(vec[0])
        result.append(vec[1])
    return result


# prepare input data, label
def loaddata_mediapipe(folder1_path, folder2_path, folader3_path, rand_state=random.randint(0, 99999)):
    folders = [folder1_path, folder2_path, folader3_path]
    x, y = [], []
    for folder in folders:
        label = os.path.join(folder, 'label.txt')
        with open(label, 'r') as f:
            context = [int(i) for i in f.read().split()]

            data = os.path.join(folder, 'data')
            filenames = os.listdir(data)
            # P.S. 須確保label順序與檔名順序相同
            if len(filenames) != len(context):
                raise Exception("data length != label length")
            for i in range(0, len(filenames)):
                imgname = os.path.join(data, filenames[i])
                img = cv2.imread(str(imgname))
                if len(extract_handpoints(img)[0]) == 21:
                    x.append(get_nn_input_from_img(img))
                    y.append(context[i])
    x_train, x_test = train_test_split(x, random_state=rand_state, train_size=0.8)
    y_train, y_test = train_test_split(y, random_state=rand_state, train_size=0.8)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def predict_nn(frame):
    if not os.path.isfile("./model_nn_mediapipe.h5"):
        raise FileNotFoundError("cannot found model file")
    model = keras.models.load_model("model_nn_mediapipe.h5")
    in_f = get_nn_input_from_img(frame)
    if len(in_f) == 0:
        return 0
    out = np.argmax(model.predict([in_f]), axis=1)[0]
    # empty: 0, up: 1, down: 2, left: 3, right: 4
    return out


if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(image)
            print(predict_nn(image))

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
