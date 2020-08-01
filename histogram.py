import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class Histogram:
    def __init__(self, img: np.ndarray):
        img = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
        self._result = np.array([img[:, i] for i in range(img.shape[1])])

        return

    def compute_hist(self):
        self._hist_b = plt.hist(self._result[0], 256, [0, 256], color='b')
        self._hist_g = plt.hist(self._result[1], 256, [0, 256], color='g')
        self._hist_r = plt.hist(self._result[2], 256, [0, 256], color='r')

    def plot_hist(self):
        plt.figure()
        plt.show()
        return
