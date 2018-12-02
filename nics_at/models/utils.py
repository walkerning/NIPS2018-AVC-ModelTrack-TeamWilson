# -*- coding: utf-8 -*-

import tensorflow as tf

def conv_relu(input_, index_, filters_, kernel_size_=(3, 3), stride_=2, name_scope="", weight_decay=0., training=False):
    conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=kernel_size_,
                             strides=(stride_, stride_), padding="same", use_bias=False,
                             kernel_regularizer=None if weight_decay is None else tf.contrib.layers.l2_regularizer(scale=weight_decay),
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
