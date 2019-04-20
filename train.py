import os
import sys
import numpy as np
import tensorflow as tf
import multiprocessing

from unet import Unet
from os import listdir
from os.path import join
from optparse import OptionParser
from data_loader import Image_Generator


def get_args():
  parser = OptionParser()
  parser.add_option('-p', '--path', dest='dataset_path', type='str',
                    help='path to the dataset')
  parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                    help='number of epochs')
  parser.add_option('-b', '--batch-size', dest='batchsize', default=3,
                    type='int', help='batch size')
  parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                    type='float', help='learning rate')
  parser.add_option('-s', '--save', dest='save_path',
                    default='/media/vlados/FreeSpace/CV_NN/Tensorflow_Unet/weights',
                    type='str', help='save model from path')

  (options, args) = parser.parse_args()
  return options


def train_net(net,
              dataset_path,
              epochs=2,
              batchsize=2,
              lr=0.01,
              classes=[0, 164, 117, 45, 194, 16],
              img_size=(1280, 720),
              save_path=None):

  input_shape = [batchsize, img_size[1], img_size[0], 3]

  x = tf.placeholder(shape=input_shape, dtype=tf.float32)
  y = tf.placeholder(shape=[batchsize, img_size[1],
                            img_size[0]], dtype=tf.int32)

  net_out = net.train(x)

  cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_out, labels=y))   
  train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost) 

  train_generator = Image_Generator(dataset_path,
                                    batch_size=batchsize,
                                    resize=img_size,
                                    classes=classes)

  # test_generator = Image_Generator(dataset_path,
  # 									batch_size=batchsize,
  # 									resize=)

  num_iter = int(
      len(listdir(join(dataset_path, 'train', 'images'))) / batchsize)

  saver = tf.train.Saver()

  # --- start session ---
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # train
    for iteration in range(epochs):
      for current_batch_index in range(num_iter):
        current_batch, current_label = next(train_generator.flow())
        sess_results = sess.run(
            [cost, train], feed_dict={x: current_batch, y: current_label})

        print(' Iter: ', num_iter, " Cost:  %.32f" %
              sess_results[0], end='\r')

      if save_path is not None:
        saver.save(sess, join(save_path, 'model'))
      print('\n-----------------------')


if __name__ == '__main__':
  args = get_args()

  net = Unet(input_depth=3,
             n_classes=6)

  try:
    train = train_net(net,
                      args.dataset_path,
                      lr=args.lr,
                      epochs=args.epochs,
                      batchsize=args.batchsize,
                      save_path=args.save_path)
    p = multiprocessing.Process(target=train)
    p.start()

  except KeyboardInterrupt:
    print('Interrupt')
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
