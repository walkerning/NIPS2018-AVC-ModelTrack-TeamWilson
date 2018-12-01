# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from nics_at.models.base import QCNN
from nics_at.models.utils import inception_module, conv_relu

class Inception(QCNN):
    TYPE = "inception_cifar10"
    def __init__(self, namescope, params={}):
        super(Inception, self).__init__(namescope, params)
        self.num_classes = params.get("num_classes", 10)
        self.substract_mean = params.get("substract_mean", [125.31 ,  122.96 ,  113.86])
        if isinstance(self.substract_mean, str):
            self.substract_mean = np.load(self.substract_mean) # load mean from npy
        self.div = np.array(params.get("div", [  51.59,   50.85 ,   51.26 ]))

    def _get_logits(self, inputs):
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
        c = tf.layers.average_pooling2d(c, pool_size=(8, 8), strides=(1,1), padding="VALID") #
        c = tf.contrib.layers.flatten(c)
        c = tf.layers.dense(c, units=self.num_classes, name="ip1",
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
        return {"logits": c}
