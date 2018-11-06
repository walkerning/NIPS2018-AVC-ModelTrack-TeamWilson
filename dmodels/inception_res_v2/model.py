# -*- coding: utf-8 -*-
import tensorflow as tf

from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after, handle_name_space

class InceptionResV2(BaseTFModel):
    FRAMEWORK = "tensorflow"
    def __init__(self, name_space=""):
        super(InceptionResV2, self).__init__()
        self.name_space = handle_name_space(name_space)

    def __call__(self, inputs, training):
        def conv_relu(input_, index_, filters_, kernel_size_ = (3, 3), stride_ = 2, name_scope=""):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=kernel_size_,
                                     strides=(stride_,stride_), padding="same", use_bias=False,
                                     name=name_scope+"conv"+str(index_))
            #kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
            #kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=training, scale=True,
                 scope=name_scope+"bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name=name_scope+"relu"+str(index_))
            return relu_
        #8*8*320
        ###Inception-A
        def inception_res_a(input_, index_):
            scope_ = "inception-res-a"+str(index_)
            br1 = conv_relu(input_, 1, 32, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 32, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 32, (3, 3), 1, scope_)
            br3 = conv_relu(input_, 4, 32, (1, 1), 1, scope_)
            br3 = conv_relu(br3, 5, 48, (3, 3), 1, scope_)
            br3 = conv_relu(br3, 6, 64, (3, 3), 1, scope_)
            merge_ = tf.concat([br1, br2, br3], 3)
            br = conv_relu(merge_, 7, 320, (1, 1), 1, scope_)
            return br + input_
        ###Inception-B
        def inception_res_b(input_, index_):
            scope_ = "inception-b"+str(index_)
            br1 = conv_relu(input_, 1, 192, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 128, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 160, (1, 7), 1, scope_)
            br2 = conv_relu(br2, 4, 192, (7, 1), 1, scope_)
            merge_ = tf.concat([br1, br2], 3)
            br = conv_relu(merge_, 5, 1088, (1, 1), 1, scope_)
            return br + input_
        def inception_res_c(input_, index_):
            scope_ = "inception-c"+str(index_)
            br1 = conv_relu(input_, 1, 192, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 192, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 224, (1, 3), 1, scope_)
            br2 = conv_relu(br2, 4, 256, (3, 1), 1, scope_)
            merge_ = tf.concat([br1, br2], 3)
            br = conv_relu(merge_, 5, 2080, (1, 1), 1, scope_)  
            return br + input_
        def reduction_res_a(input_):
            scope_ = "reduction-a"
            br1 = conv_relu(input_, 1, 384, (3, 3), 2, scope_)
            br2 = conv_relu(input_, 2, 256, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 256, (3, 3), 1, scope_)
            br2 = conv_relu(br2, 4, 384, (3, 3), 2, scope_)
            br3 = tf.layers.max_pooling2d(input_, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
            merge_ = tf.concat([br1, br2, br3], 3)
            return merge_
        def reduction_res_b(input_):
            scope_ = "reduction-b"
            br1 = conv_relu(input_, 1, 256, (1, 1), 1, scope_)
            br1 = conv_relu(br1, 2, 384, (3, 3), 2, scope_)
            br2 = conv_relu(input_, 3, 256, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 4, 288, (3, 3), 2, scope_)
            br3 = conv_relu(input_, 5, 256, (1, 1), 1, scope_)
            br3 = conv_relu(br3, 6, 288, (3, 3), 1, scope_)
            br3 = conv_relu(br3, 7, 320, (3, 3), 2, scope_)
            br4 = tf.layers.max_pooling2d(input_, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
            merge_ = tf.concat([br1, br2, br3, br4], 3)
            return merge_
        with tf.variable_scope(self.name_space):
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
            inputs = inputs - tf.constant(_CHANNEL_MEANS)
            ###stem
            c = conv_relu(inputs, 1, 32, (3, 3), 2)
            c = conv_relu(c, 2, 64, (3, 3), 1)
            #32*32*64
            c = tf.layers.max_pooling2d(inputs=c, pool_size=(3,3),
                        strides=(2,2), padding='SAME')
            c = conv_relu(c, 3, 80, (1, 1), 1)
            c = conv_relu(c, 4, 192, (3, 3), 1)
            c = tf.layers.max_pooling2d(inputs=c, pool_size=(3,3),
                        strides=(2,2), padding='SAME')
            #8*8*192
            c0 = conv_relu(c, 5, 96, (1, 1), 1)
            c1 = conv_relu(c, 6, 48, (1, 1), 1)
            c1 = conv_relu(c1, 7, 64, (5, 5), 1)
            c2 = conv_relu(c, 8, 64, (1, 1), 1)
            c2 = conv_relu(c2, 9, 96, (3, 3), 1)
            c2 = conv_relu(c2, 10, 96, (3, 3), 1)
            c3 = tf.layers.average_pooling2d(inputs=c, pool_size=(3,3),
                        strides=(1,1), padding='SAME')
            c3 = conv_relu(c3, 11, 64, (1, 1), 1)
            c = tf.concat([c0, c1, c2, c3], 3)
            #inception-A phase 8*8*384
            for i in range(5):
                c = inception_res_a(c, i)
            #reduction-A phase 4*4*1152
            c = reduction_res_a(c)
            #inception-B phase 4*4*1152
            for i in range(10):
                c = inception_res_b(c, i)
            #reduction-B phase 2*2*2048
            c = reduction_res_b(c)
            #reduction-C phase 2*2*2048
            for i in range(5):
                c = inception_res_c(c, i)
            c = tf.layers.average_pooling2d(c, pool_size=(2,2),
                    strides=(2,2), padding='SAME')
            c = tf.layers.dropout(c, 0.2, training)
            c = tf.contrib.layers.flatten(c)
            c = tf.layers.dense(c, units=200, name="ip1")
        return c

Model = InceptionResV2
