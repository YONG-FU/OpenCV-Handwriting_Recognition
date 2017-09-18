import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
import os, errno
import glob, os

def EnhanceImage():
    img = cv2.imread('image14.jpg',0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    cv2.imwrite('res.jpg',res)

EnhanceImage()