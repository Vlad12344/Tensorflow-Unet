import tensorflow as tf
from unet_parts import convlayer_left, convlayer_right

class Unet():

    def __init__(self, 
                 input_depth=3,
                 n_classes=5):

        self.input_depth = input_depth
        self.n_classes = n_classes

    def train(self, x):

    # --- make layer ---
    # left
        l1_1 = convlayer_left(3, self.input_depth, 32)
        l1_2 = convlayer_left(3, 32, 32)

        l2_1 = convlayer_left(3, 32, 64)
        l2_2 = convlayer_left(3, 64, 64)

        l3_1 = convlayer_left(3, 64, 128)
        l3_2 = convlayer_left(3, 128, 128)

        l4_1 = convlayer_left(3, 128, 256)
        l4_2 = convlayer_left(3, 256, 256)

        l5_1 = convlayer_left(3, 256, 512)
        l5_2 = convlayer_left(3, 512, 512)

    # right
        l6_1 = convlayer_right(3, 384, 768)
        l6_2 = convlayer_left(3, 384, 384)
        l6_3 = convlayer_left(3, 384, 192)

        l7_1 = convlayer_right(3, 160, 320)
        l7_2 = convlayer_left(3, 160, 160)
        l7_3 = convlayer_left(3, 160, 128)

        l8_1 = convlayer_right(3, 96, 192)
        l8_2 = convlayer_left(3, 96, 96)
        l8_3 = convlayer_left(3, 96, 48)

        l9_1 = convlayer_right(3, 40, 80)
        l9_2 = convlayer_left(3, 40, 40)
        l9_3 = convlayer_left(3, 40, 32)

        l10_final = convlayer_left(1, 32, self.n_classes)

        # ---- make graph ----
        layer1_1 = l1_1.feedforward(x)
        layer1_2 = l1_2.feedforward(layer1_1)

        layer2_Input = tf.nn.max_pool(layer1_2, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='VALID')
        layer2_1 = l2_1.feedforward(layer2_Input)
        layer2_2 = l2_2.feedforward(layer2_1)

        layer3_Input = tf.nn.max_pool(layer2_2, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='VALID')
        layer3_1 = l3_1.feedforward(layer3_Input)
        layer3_2 = l3_2.feedforward(layer3_1)

        layer4_Input = tf.nn.max_pool(layer3_2, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='VALID')
        layer4_1 = l4_1.feedforward(layer4_Input)
        layer4_2 = l4_2.feedforward(layer4_1)

        layer5_Input = tf.nn.max_pool(layer4_2, ksize=[1, 2, 2, 1], strides=[
                                      1, 2, 2, 1], padding='VALID')
        layer5_1 = l5_1.feedforward(layer5_Input)
        layer5_2 = l5_2.feedforward(layer5_1)

        layer6_Input = tf.concat([layer5_2, layer5_Input], axis=3)
        layer6_1 = l6_1.feedforward(layer6_Input)
        layer6_2 = l6_2.feedforward(layer6_1)
        layer6_3 = l6_3.feedforward(layer6_2)

        layer7_Input = tf.concat([layer6_3, layer4_Input], axis=3)
        layer7_1 = l7_1.feedforward(layer7_Input)
        layer7_2 = l7_2.feedforward(layer7_1)
        layer7_3 = l7_3.feedforward(layer7_2)

        layer8_Input = tf.concat([layer7_3, layer3_Input], axis=3)
        layer8_1 = l8_1.feedforward(layer8_Input)
        layer8_2 = l8_2.feedforward(layer8_1)
        layer8_3 = l8_3.feedforward(layer8_2)

        layer9_Input = tf.concat([layer8_3, layer2_Input], axis=3)
        layer9_1 = l9_1.feedforward(layer9_Input)
        layer9_2 = l9_2.feedforward(layer9_1)
        layer9_3 = l9_3.feedforward(layer9_2)

        layer10 = l10_final.feedforward(layer9_3)

        return layer10