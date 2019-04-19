import numpy as np
import tensorflow as tf

np.random.seed(678)
tf.set_random_seed(5678)


def tf_relu(x): return tf.nn.relu(x)


def d_tf_relu(s): return tf.cast(tf.greater(s, 0), dtype=tf.float32)


def tf_softmax(x): return tf.nn.softmax(x)


def np_sigmoid(x): 1 / (1 + np.exp(-1 * x))


class convlayer_left():

    def __init__(self, ker, in_ch, out_ch):
        self.w = tf.Variable(tf.random_normal(
            [ker, ker, in_ch, out_ch], stddev=0.05))

    def feedforward(self, inp, stride=1, dilate=1):
        self.inp = inp
        self.layer = tf.nn.conv2d(inp, self.w, strides=[
                                  1, stride, stride, 1], padding='SAME')
        self.layerA = tf_relu(self.layer)

        return self.layerA


class convlayer_right():

    def __init__(self, ker, in_ch, out_ch):
        self.w = tf.Variable(tf.random_normal(
            [ker, ker, in_ch, out_ch], stddev=0.05))

    def feedforward(self, inp, stride=1, dilate=1, output=1):
        self.inp = inp

        current_shape_size = inp.shape

        self.layer = tf.nn.conv2d_transpose(inp, self.w, output_shape=[current_shape_size[0].value] + [int(current_shape_size[1].value * 2), int(
            current_shape_size[2].value * 2), int(current_shape_size[3].value / 2)], strides=[1, 2, 2, 1], padding='SAME')
        self.layerA = tf_relu(self.layer)

        return self.layerA
