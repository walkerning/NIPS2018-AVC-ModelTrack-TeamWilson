# -*- coding: utf-8 -*-

import tensorflow as tf

from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after

class Denoise(BaseTFModel):
    def __init__(self, **cfg):
        super(Denoise, self).__init__()
        self.name_space = cfg["name_space"]
        # self.training_status = tf.placeholder_with_default(False, shape=())
        self.forward = []
        self.backward = []
        self.fwd_out = [64, 128, 256, 256, 256]
        self.num_fwd = [2, 3, 3, 3, 3]
        self.back_out = [64, 128, 256, 256]
        self.num_back = [2, 3, 3, 3]
        # self.weight_decay = FLAGS.weight_decay

    @tf_vars_before_after
    def __call__(self, input_, training, reuse=False):
        # this denoiser is channel_last
        self.training_status = training
        def conv_bn_relu(input_, index_, filters_, stride_ = 1):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=(3,3),
                                     strides=(stride_,stride_), padding="same", use_bias=False, name="conv"+str(index_))
             # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
             #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), 
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=self.training_status, scale=True, scope="bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name="relu"+str(index_))
            return relu_
        with tf.variable_scope(self.name_space, reuse=reuse):
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
            x = input_ - tf.constant(_CHANNEL_MEANS)
            counter = 0
            for i in range(len(self.num_fwd)):
                for j in range(self.num_fwd[i]):
                    stride_ = 1
                    if j == 0:
                        if i != 0:
                            stride_ = 2
                    x = conv_bn_relu(x, counter, self.fwd_out[i], stride_)
                    self.forward.append(x)
                    counter += 1
            self.backward.append(self.forward[-1])
            for i in range(len(self.num_back) - 1, -1, -1):
                upsample = tf.image.resize_bilinear(self.backward[-1], self.forward[i].shape[1:3])
                x = tf.concat([upsample, self.forward[i]], 3)
                x = conv_bn_relu(x, counter, self.back_out[i])
                self.backward.append(x)
                counter += 1
            x = tf.layers.conv2d(x, filters=3, kernel_size=(1,1), padding="same", use_bias=False, name="last_conv")
            #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            self.output = x + input_
        return self.output

Model = Denoise
