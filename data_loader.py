import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor
from utils import create_labels, norm_data
from transformation_fcn import resize, crop


class Image_Generator():

    def __init__(self,
                 path,
                 ROI=None,
                 resize=None,
                 batch_size=2,
                 classes=None):

        self.ROI = ROI
        self.path = path
        self.resize = resize
        self.classes = classes
        self.batch_size = batch_size

        # if len(os.listdir(sels.path)) != 2:
        #     raise Exception('The Dataset must be consist with 2 folders: images and masks')

        

        self.images = [os.path.join(os.path.join(self.path, 'images'), x)
                       for x in os.listdir(os.path.join(self.path, 'images'))]
        self.masks = [os.path.join(os.path.join(self.path, 'masks'), x)
                      for x in os.listdir(os.path.join(self.path, 'masks'))]

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

            yield np.array(images_batch, dtype=np.float32), np.array(labels_batch, dtype=np.float32)
