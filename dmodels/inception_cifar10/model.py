# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after, handle_name_space

def conv_relu(input_, index_, filters_, kernel_size_=(3, 3), stride_=2, name_scope="", weight_decay=0., training=False):
    if training == False:
        conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=kernel_size_,
                                 strides=(stride_, stride_), padding="same", use_bias=False,
                                 name=name_scope+"conv"+str(index_))
    else:
        conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=kernel_size_,
                                 strides=(stride_, stride_), padding="same", use_bias=False,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=name_scope+"conv"+str(index_))
    bn_ = tf.contrib.layers.batch_norm(conv_, is_training=training, scale=True,
                                       scope=name_scope+"bn"+str(index_), decay=0.9)
    relu_ = tf.nn.relu(bn_, name=name_scope+"relu"+str(index_))
    return relu_

def inception_module(input_, index_, br1, br2_1, br2_2, br3_1, br3_2, br4, pool=False, weight_decay=0., training=False):
    scope_ = "inception" + str(index_)
    br1_c = conv_relu(input_, 1, br1,   (1, 1), 1, scope_, weight_decay, training)
    br2_c = conv_relu(input_, 2, br2_1, (1, 1), 1, scope_, weight_decay, training)
    br2_c = conv_relu(br2_c,    3, br2_2, (3, 3), 1, scope_, weight_decay, training)
    br3_c = conv_relu(input_, 4, br3_1, (1, 1), 1, scope_, weight_decay, training)
    br3_c = conv_relu(br3_c,    5, br3_2, (3, 3), 1, scope_, weight_decay, training)
    br3_c = conv_relu(br3_c,    6, br3_2, (3, 3), 1, scope_, weight_decay, training)
    br4_c = tf.layers.max_pooling2d(input_, pool_size=(3, 3), strides=(1, 1), padding="SAME")
    br4_c = conv_relu(br4_c,    7, br4,   (1, 1), 1, scope_, weight_decay, training)
    merge_ = tf.concat([br1_c, br2_c, br3_c, br4_c], 3)
    if pool:
        merge_ = tf.layers.max_pooling2d(merge_, pool_size=(3, 3), strides=(2, 2), padding="SAME")
    return merge_


class Inception(BaseTFModel):
    FRAMEWORK = "tensorflow"
    def __init__(self, num_classes=10, substract_mean=[ 125.31 ,  122.96 ,  113.86], div=[  51.59,   50.85 ,   51.26 ], name_space=""):
        # no configuration now
        super(Inception, self).__init__()
        self.name_space = handle_name_space(name_space)
        self.num_classes = num_classes
        self.substract_mean = substract_mean
        if isinstance(self.substract_mean, str):
            self.substract_mean = np.load(self.substract_mean) # load mean from npy
        self.div = div

    @tf_vars_before_after
    def __call__(self, inputs, training):
        self.weight_decay = 0.
        self.training = training
        with tf.variable_scope(self.name_space):
            inputs = inputs - tf.cast(tf.constant(self.substract_mean), tf.float32)
            if self.div is not None and not np.all(self.div == 1.):
                inputs = inputs / self.div
            c = conv_relu(inputs, 1, 192, (3, 3), 1, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 1, br1=64,
                                 br2_1=96, br2_2=128,
                                 br3_1=16, br3_2=32,
                                 br4=32, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 2, br1=128,
                                 br2_1=128, br2_2=192,
                                 br3_1=32, br3_2=96,
                                 br4=64, weight_decay=self.weight_decay, training=self.training, pool=True) # stride 2
            c = inception_module(c, 3, br1=192,
                                 br2_1=96, br2_2=208,
                                 br3_1=16, br3_2=48,
                                 br4=64, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 4, br1=160,
                                 br2_1=112, br2_2=224,
                                 br3_1=24, br3_2=64,
                                 br4=64, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 5, br1=128,
                                 br2_1=128, br2_2=256,
                                 br3_1=24, br3_2=64,
                                 br4=64, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 6, br1=112,
                                 br2_1=144, br2_2=288,
                                 br3_1=32, br3_2=64,
                                 br4=64, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 7, br1=256,
                                 br2_1=160, br2_2=320,
                                 br3_1=32, br3_2=128,
                                 br4=128, weight_decay=self.weight_decay, training=self.training, pool=True) # stride 2
            c = inception_module(c, 8, br1=256,
                                 br2_1=160, br2_2=320,
                                 br3_1=32, br3_2=128,
                                 br4=128, weight_decay=self.weight_decay, training=self.training)
            c = inception_module(c, 9, br1=384,
                                 br2_1=192, br2_2=384,
                                 br3_1=48, br3_2=128,
                                 br4=128, weight_decay=self.weight_decay, training=self.training)
            c = tf.layers.average_pooling2d(c, pool_size=(8, 8), strides=(1,1), padding="VALID")
            c = tf.contrib.layers.flatten(c)
            c = tf.layers.dense(c, units=self.num_classes, name="ip1")
        return c

Model = Inception
