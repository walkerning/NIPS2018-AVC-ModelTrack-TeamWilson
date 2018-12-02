# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from nics_at.models.base import QCNN

class VGG9(QCNN):
    TYPE = "vgg9"
    def __init__(self, namescope, params={}):
        super(VGG9, self).__init__(namescope, params)
        self.num_classes = params.get("num_classes", 200)
        self.substract_mean = params.get("substract_mean", [123.68, 116.78, 103.94])
        if isinstance(self.substract_mean, str):
            self.substract_mean = np.load(self.substract_mean) # load mean from npy
        self.div = np.array(params.get("div", 1))
        self.filter_size_div = params.get("filter_size_div", 1)

    def _get_logits(self, inputs):
        weight_decay = self.weight_decay
        def conv_relu_pool(input_, index_, filters_, use_pool = True, kernel_size_ = 3, stride_ = 1, name_scope = ""):
            if self.filter_size_div != 1:
                filters_ = filters_ // self.filter_size_div
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=(kernel_size_,kernel_size_),
             strides=(stride_,stride_), padding="same", use_bias=False,
             kernel_regularizer=None if self.test_only else tf.contrib.layers.l2_regularizer(scale=weight_decay),
             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=name_scope + "conv"+str(index_))
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=self.training, scale=True,
                 scope=name_scope+"bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name=name_scope + "relu"+str(index_))
            if use_pool:
                pool_ = tf.layers.max_pooling2d(relu_, name=name_scope+"pool"+str(index_), pool_size=(2, 2), strides=2)
                return conv_, relu_, pool_
            else:
                return conv_, relu_
        inputs = inputs - tf.cast(tf.constant(self.substract_mean), tf.float32)
        if self.div is not None and not np.all(self.div == 1.):
            inputs = inputs / self.div
        conv1, relu1, pool1 = conv_relu_pool(inputs, 1, 64)
        conv2, relu2, pool2 = conv_relu_pool(pool1, 2, 128)
        conv3_1, relu3_1 = conv_relu_pool(pool2, 3, 256, use_pool=False)
        conv3_2, relu3_2, pool3 = conv_relu_pool(relu3_1, 4, 256, use_pool=True)
        conv4_1, relu4_1 = conv_relu_pool(pool3, 5, 512, use_pool=False)
        conv4_2, relu4_2, pool4 = conv_relu_pool(relu4_1, 6, 512, use_pool=True)
        conv5_1, relu5_1 = conv_relu_pool(pool4, 7, 512, use_pool=False)
        conv5_2, relu5_2, pool5 = conv_relu_pool(relu5_1, 8, 512, use_pool=True)
        flat = tf.contrib.layers.flatten(pool5)
        logits = tf.layers.dense(flat, units=self.num_classes, name="logits",
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                 kernel_regularizer=None if self.test_only else tf.contrib.layers.l2_regularizer(scale=weight_decay))
        return {"logits": logits}
