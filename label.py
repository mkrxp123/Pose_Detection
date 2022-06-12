import os
import cv2
import numpy as np

def label(img_name):
    # 顯示圖片
    img = cv2.imread(img_name)
    cv2.imshow('live', img)
    
    key = cv2.waitKey(0)  
    cv2.destroyAllWindows()
    return key

folder = '3'
# 把圖片資料夾存改名data後放在另一個資料夾叫<folder>
img_list = [f"./{folder}/data/" + i for i in os.listdir(f"./{folder}/data")]
try:
    Y = np.loadtxt(f'./{folder}/label.txt')
    print('loaded label')
except:
    Y = np.zeros(len(img_list), dtype=int)
    print('create new label')
save = True
start_idx, end_idx = 0, len(img_list)
for i, img_name in enumerate(img_list[start_idx:end_idx]): # selected by index
    k = label(img_name)
    # empty: 0, up: 1, down: 2, left: 3, right: 4
    if k == ord('5'):
        Y[start_idx + i] = 0
    elif k == ord('8'):
        Y[start_idx + i] = 1
    elif k == ord('2'):
        Y[start_idx + i] = 2
    elif k == ord('4'):
        Y[start_idx + i] = 3
    elif k == ord('6'):
        Y[start_idx + i] = 4
    else:
        save = False
        break
if save:
    print(Y)
    np.savetxt(f'./{folder}/label.txt', Y, fmt='%d')