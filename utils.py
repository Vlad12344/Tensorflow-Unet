import numpy as np


def create_labels(mask, classes):
    height, lenght = mask.shape[0], mask.shape[1]
    y = np.zeros((height, lenght), dtype=np.int32)

    for k, class_ in enumerate(classes):
        points = np.argwhere(mask == class_)
        for point in points:
            y[point[0], point[1]] = k

    return y


def norm_data(img):
    return (img - img.mean()) / img.std()
