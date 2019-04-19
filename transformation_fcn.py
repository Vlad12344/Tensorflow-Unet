import cv2
import numpy as np


def resize(img, resize=None):

    if resize != None and type(resize) == tuple:
        img = cv2.resize(img, resize)
    elif resize is None:
        return img
    else:
        raise Exception('Type of resize variable must be Tuple')

    return img


def crop(img, ROI=None):

    if ROI != None and type(ROI) == list:
        img = img[int(ROI[1]):int(ROI[1] + ROI[3]),
                       int(ROI[0]):int(ROI[0] + ROI[2])]
    elif ROI is None:
        return img
    else:
        raise Exception('Type of ROI variable must be List')

    return img
