import numpy as np
import tensorflow as tf

from unet import Unet
from os import listdir
from os.path import join
from optparse import OptionParser
from data_loader import Image_Generator


def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--path', dest='dataset_path', type='str',
                      help='path to the dataset')
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-i', '--num_iter', dest='num_iter', default=100,
                      type='float', help='learning rate')

    (options, args) = parser.parse_args()
    return options


def train_net(net,
              dataset_path,
              epochs=2,
              batchsize=2,
              lr=0.001,
              classes=[164, 117, 45, 194, 16],
              img_size=(1280, 720)):

    input_shape = [batchsize, img_size[1], img_size[0], 3]

    x = tf.placeholder(shape=input_shape, dtype=tf.float32)
    y = tf.placeholder(shape=[batchsize, img_size[1],
                              img_size[0], len(classes)], dtype=tf.float32)

    net_out = net.train(x)

    cost = tf.reduce_mean(tf.square(net_out - y))
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    train_generator = Image_Generator(dataset_path,
                                      batch_size=batchsize,
                                      resize=img_size,
                                      classes=classes)

    num_iter = int(len(listdir(join(dataset_path, 'train', 'images'))) / batchsize)

    # --- start session ---
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for iteration in range(epochs):

            # train
            for current_batch_index in range(num_iter):
                current_batch, current_label = next(train_generator.flow())

                print(current_batch.shape, current_label.shape)
                sess_results = sess.run(
                    [train], feed_dict={x: current_batch, y: current_label})

                print(' Iter: ', iter, " Cost:  %.32f" %
                      sess_results[0], end='\r')
            print('\n-----------------------')


if __name__ == '__main__':
    args = get_args()

    net = Unet(input_depth=3, 
    		   n_classes=5)

    train_net(net,
              args.dataset_path,
              epochs=args.epochs,
              batchsize=args.batchsize,
              lr=args.lr)
