# -*- coding: utf-8 -*-

import tensorflow as tf
from nics_at.models.base import QCNN

class Mnist_madry(QCNN):
    TYPE = "mnist_madry"
    def __init__(self, namescope, params={}):
        super(Mnist_madry, self).__init__(namescope, params)

    def _get_logits(self, inputs):
        # first convolutional layer
        W_conv1 = self._weight_variable([5,5,1,32], "conv1")
        b_conv1 = self._bias_variable([32], "conv1")
        h_conv1 = tf.nn.relu(self._conv2d(inputs, W_conv1) + b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        # second convolutional layer
        W_conv2 = self._weight_variable([5,5,32,64], "conv2")
        b_conv2 = self._bias_variable([64], "conv2")

        h_conv2 = tf.nn.relu(self._conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        # first fully connected layer
        W_fc1 = self._weight_variable([7 * 7 * 64, 1024], "fc1")
        b_fc1 = self._bias_variable([1024], "fc1")

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # output layer
        W_fc2 = self._weight_variable([1024,10], "fc2")
        b_fc2 = self._bias_variable([10], "fc2")

        self.logits = tf.matmul(h_fc1, W_fc2) + b_fc2
        return {
            "logits": self.logits
        }

    @staticmethod
    def _weight_variable(shape, name):
        return tf.get_variable("{}/weight".format(name), shape=shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))

    @staticmethod
    def _bias_variable(shape, name):
        return tf.get_variable("{}/bias".format(name), shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides=[1,2,2,1],
                              padding="SAME")
