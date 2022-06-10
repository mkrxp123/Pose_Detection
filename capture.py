import cv2
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='data create')
parser.add_argument(
    '--dataset_path', type=str, required=True,
    help='path to the dataset'
)
args = parser.parse_args()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
ret, frame = cap.read()
imgs = np.expand_dims(frame, axis=0)
start = time.time()
while(True):
    # 擷取影像
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 顯示圖片
    cv2.imshow('live', frame)
    imgs = np.append(imgs, np.expand_dims(frame, axis=0), axis=0)
    
    # 按下 q 鍵離開迴圈
    if cv2.waitKey(1) == ord('q'):
        break

end = time.time()
cap.release()
cv2.destroyAllWindows()
print(end-start)
img_size = (128, 96)
for i, img in enumerate(imgs):
    cv2.imwrite(f'{args.dataset_path}/{i:04d}.png', cv2.resize(img, img_size))