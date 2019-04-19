import os
import cv2
import numpy as np
from utils import create_labels, norm_data
from transformation_fcn import resize, crop


class Image_Generator():

    def __init__(self, path, batch_size=2, resize=None, ROI=None, classes=None):

        self.ROI = ROI
        self.path = path
        self.resize = resize
        self.classes = classes
        self.batch_size=batch_size

        self.images = [os.path.join(os.path.join(self.path, 'train/images'), x)
                       for x in os.listdir(os.path.join(self.path, 'train/images'))]
        self.masks = [os.path.join(os.path.join(self.path, 'train/masks'), x)
                      for x in os.listdir(os.path.join(self.path, 'train/masks'))]

    def flow(self):
        i = 0

        while True:
            images_batch = []
            labels_batch = []
            for b in range(self.batch_size):
                if i == len(self.images):
                    i = 0

                image = cv2.imread(self.images[i])
                image = resize(image, resize=self.resize)
                image = crop(image, ROI=self.ROI)

                label_img = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
                label_img = resize(label_img, resize=self.resize)
                label_img = crop(label_img, ROI=self.ROI)

                images_batch.append(norm_data(image))
                labels_batch.append(create_labels(label_img, self.classes))

                i += 1

            yield np.array(images_batch), np.array(labels_batch)
