import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def loaddata(folder1_path, folder2_path, folader3_path, train_size):
    folders = [folder1_path, folder2_path, folader3_path]
    X, Y = [], []
    for folder in folders:
        # get X from img
        data = os.path.join(folder, 'data')
        for root, dirs, files in os.walk(data):
            for name in files:               
                imgname = os.path.join(root, name)
                img = cv2.imread(str(imgname))
                X.append(img)
        label = os.path.join(folder, 'label.txt')
        # get Y from label.txt
        with open(label, 'r') as f:
            context = [int(i) for i in f.read().split()]
            Y += context

    # split to train, valid
    dataset = list(range(len(Y)))
    train_index, val_index = train_test_split(dataset, random_state=777, train_size=train_size)
    x_train = [x for i, x in enumerate(X) if i in train_index]
    y_train = [y for i, y in enumerate(Y) if i in train_index]
    x_test = [x for i, x in enumerate(X) if i in val_index]
    y_test = [y for i, y in enumerate(Y) if i in val_index]
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

if __name__ == '__main__':
    f1 = '1'
    f2 = '2'
    f3 = '3'
    f1path = os.path.join(os.path.abspath(os.getcwd()), f1)
    f2path = os.path.join(os.path.abspath(os.getcwd()), f2)
    f3path = os.path.join(os.path.abspath(os.getcwd()), f3)
    x_train, x_test, y_train, y_test = loaddata(f1path, f2path, f3path, train_size=0.8)
    '''
    for test
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    '''