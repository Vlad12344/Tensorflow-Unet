import numpy as np


def create_labels(mask, classes):
    height, lenght, N = mask.shape[0], mask.shape[1], len(classes)
    y = np.zeros((height, lenght, N))

    for k, class_ in enumerate(classes):
        points = np.argwhere(mask == class_)
        for point in points:
            y[point[0], point[1], k] = 1

    return y


def norm_data(img):
    return (img - img.mean()) / img.std()
