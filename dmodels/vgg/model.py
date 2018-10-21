# -*- coding: utf-8 -*-
import tensorflow as tf

from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after

class VGG(BaseTFModel):
    FRAMEWORK = "tensorflow"
    def __init__(self, name_space=""):
        # no configuration now
        super(VGG, self).__init__()
        self.name_space = name_space

    @tf_vars_before_after
    def __call__(self, inputs, training):
        def conv_relu_pool(input_, index_, filters_, use_pool = True, kernel_size_ = 3, stride_ = 1, name_scope = ""):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=(kernel_size_,kernel_size_),
                                     strides=(stride_,stride_), padding="same", use_bias=False,
                                     #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                     #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name=name_scope + "conv"+str(index_))
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=training, scale=True,
                                               scope=name_scope+"bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name=name_scope + "relu"+str(index_))
            if use_pool:
                pool_ = tf.layers.max_pooling2d(relu_, name=name_scope+"pool"+str(index_), pool_size=(2, 2), strides=2)
                return conv_, relu_, pool_
            else:
                return conv_, relu_
        with tf.variable_scope(self.name_space):
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
            inputs = inputs - tf.constant(_CHANNEL_MEANS)                 
            conv1, relu1, pool1 = conv_relu_pool(inputs, 1, 64)
            conv2, relu2, pool2 = conv_relu_pool(pool1, 2, 128)
            conv3_1, relu3_1 = conv_relu_pool(pool2, 3, 256, use_pool=False)
            conv3_2, relu3_2, pool3 = conv_relu_pool(relu3_1, 4, 256, use_pool=True)
            conv4_1, relu4_1 = conv_relu_pool(pool3, 5, 512, use_pool=False)
            conv4_2, relu4_2, pool4 = conv_relu_pool(relu4_1, 6, 512, use_pool=True)
            conv5_1, relu5_1 = conv_relu_pool(pool4, 7, 512, use_pool=False)
            conv5_2, relu5_2, pool5 = conv_relu_pool(relu5_1, 8, 512, use_pool=True)
            flat = tf.contrib.layers.flatten(pool5)
            ip1 = tf.layers.dense(flat, units=2048, name="ip1")
            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            relu6 = tf.nn.relu(ip1, name="relu6")
            ip2 = tf.layers.dense(relu6, units=2048, name="ip2")
            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
            relu7 = tf.nn.relu(ip2, name="relu7")
            logits = tf.layers.dense(relu7, units=200, name="logits")
            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        return logits
Model = VGG
