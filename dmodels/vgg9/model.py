# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after, handle_name_space

class VGG(BaseTFModel):
    FRAMEWORK = "tensorflow"
    def __init__(self, num_classes=200, substract_mean=[123.68, 116.78, 103.94], div=1., use_bn=True, use_bias=True, use_bn_renorm=False, name_space=""):
        # no configuration now
        super(VGG, self).__init__()
        self.name_space = handle_name_space(name_space)
        self.num_classes = num_classes
        self.substract_mean = substract_mean
        if isinstance(self.substract_mean, str):
            self.substract_mean = np.load(self.substract_mean) # load mean from npy
        self.div = div
        self.use_bn = use_bn
        self.use_bn_renorm = use_bn_renorm
        self.use_bias = use_bias

    @tf_vars_before_after
    def __call__(self, inputs, training):
        def conv_relu_pool(input_, index_, filters_, use_pool = True, kernel_size_ = 3, stride_ = 1, name_scope = ""):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=(kernel_size_,kernel_size_),
                                     strides=(stride_,stride_), padding="same", use_bias=False,
                                     name=name_scope + "conv"+str(index_))
            if self.use_bn:
                bn_ = tf.contrib.layers.batch_norm(conv_, is_training=training, scale=True,
                                                   scope=name_scope+"bn"+str(index_), decay=0.9,
                                                   renorm=self.use_bn_renorm)
            else:
                bn_ = conv_
            relu_ = tf.nn.relu(bn_, name=name_scope + "relu"+str(index_))
            if use_pool:
                pool_ = tf.layers.max_pooling2d(relu_, name=name_scope+"pool"+str(index_), pool_size=(2, 2), strides=2)
                return conv_, relu_, pool_
            else:
                return conv_, relu_
        with tf.variable_scope(self.name_space):
            inputs = inputs - tf.cast(tf.constant(self.substract_mean), tf.float32)
            if self.div and not np.all(self.div == 1.):
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
            logits = tf.layers.dense(flat, units=self.num_classes, name="logits", use_bias=self.use_bias)
        return logits
Model = VGG
