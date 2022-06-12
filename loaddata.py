#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# In[2]:
def loaddata(folder1_path, folder2_path, folader3_path):
    folders = [folder1_path, folder2_path, folader3_path]
    x, y = [], []
    for folder in folders:
        data = os.path.join(folder, 'data')
        for root, dirs, files in os.walk(data):
            for name in files:               
                imgname = os.path.join(root, name)
                img = cv2.imread(str(imgname))
                x.append(img)
        label = os.path.join(folder, 'label.txt')
        with open(label, 'r') as f:
            context = [int(i) for i in f.read().split()]
            y += context
    x_train, x_test = train_test_split(x, random_state=777, train_size=0.8)
    y_train, y_test = train_test_split(y, random_state=777, train_size=0.8)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

# In[3]:
if __name__ == '__main__':
    f1 = '3'
    f2 = '3'
    f3 = '3'
    f1path = os.path.join(os.path.abspath(os.getcwd()), f1)
    f2path = os.path.join(os.path.abspath(os.getcwd()), f2)
    f3path = os.path.join(os.path.abspath(os.getcwd()), f3)
    x_train, x_test, y_train, y_test = loaddata(f1path, f2path, f3path)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)