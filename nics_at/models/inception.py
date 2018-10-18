# -*- coding: utf-8 -*-
import tensorflow as tf

from nics_at.models.base import QCNN

class Inception(QCNN):
    TYPE = "inception"
    def __init__(self, namescope, params={}):
        super(Inception, self).__init__(namescope, params)

    def _get_logits(self, inputs):
        def conv_relu(input_, index_, filters_, kernel_size_ = (3, 3), stride_ = 2, name_scope=""):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=kernel_size_,
                strides=(stride_,stride_), padding="same", use_bias=False,
             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name=name_scope+"conv"+str(index_))
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=self.training, scale=True,
                 scope=name_scope+"bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name=name_scope+"relu"+str(index_))
            return relu_
        ###stem
        c = conv_relu(inputs, 1, 32, (3, 3), 2)
        c = conv_relu(c, 2, 64, (3, 3), 1)
        #branch1 32*32*64
        c1 = tf.layers.max_pooling2d(inputs=c, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
        c2 = conv_relu(c, 3, 96, (3, 3), 2)
        c = tf.concat([c1, c2], 3)
        #branch2 16*16*160
        c1 = conv_relu(c, 4, 64, (1, 1), 1)
        c1 = conv_relu(c1, 5, 96, (3, 3), 1)
        c2 = conv_relu(c, 6, 64, (1, 1), 1)
        c2 = conv_relu(c2, 7, 64, (1, 7), 1)
        c2 = conv_relu(c2, 8, 64, (7, 1), 1)
        c2 = conv_relu(c2, 9, 96, (3, 3), 1)
        c = tf.concat([c1, c2], 3)
        #branch3 16*16*192
        c1 = tf.layers.max_pooling2d(inputs=c, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
        c2 = conv_relu(c, 10, 192, (3, 3), 2)
        c = tf.concat([c1, c2], 3)
        c = tf.contrib.layers.batch_norm(c, 
            is_training=self.training, scale=True, 
            scope="branch3-bn1", decay=0.9)
        c = tf.nn.relu(c, name="relu_branch3")
        #8*8*384
        ###Inception-A
        def inception_a(input_, index_):
            scope_ = "inception-a"+str(index_)
            br1 = conv_relu(input_, 1, 96, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 64, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 96, (3, 3), 1, scope_)
            br3 = conv_relu(input_, 4, 64, (1, 1), 1, scope_)
            br3 = conv_relu(br3, 5, 96, (3, 3), 1, scope_)
            br3 = conv_relu(br3, 6, 96, (3, 3), 1, scope_)
            br4 = tf.layers.average_pooling2d(input_, pool_size=(3,3),
                    strides=(1,1), padding='SAME')
            br4 = conv_relu(br4, 7, 96, (1, 1), 1, scope_)
            merge_ = tf.concat([br1, br2, br3, br4], 3)
            return merge_
        ###Inception-B
        def inception_b(input_, index_):
            scope_ = "inception-b"+str(index_)
            br1 = conv_relu(input_, 1, 384, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 192, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 224, (1, 7), 1, scope_)
            br2 = conv_relu(br2, 4, 256, (7, 1), 1, scope_)
            br3 = conv_relu(input_, 5, 192, (1, 1), 1, scope_)
            br3 = conv_relu(br3, 6, 192, (7, 1), 1, scope_)
            br3 = conv_relu(br3, 7, 224, (1, 7), 1, scope_)
            br3 = conv_relu(br3, 8, 224, (7, 1), 1, scope_)
            br3 = conv_relu(br3, 9, 256, (1, 7), 1, scope_)
            br4 = tf.layers.average_pooling2d(input_, pool_size=(3,3),
                    strides=(1,1), padding='SAME')
            br4 = conv_relu(br4, 10, 128, (1, 1), 1, scope_)
            merge_ = tf.concat([br1, br2, br3, br4], 3)
            return merge_
        def inception_c(input_, index_):
            scope_ = "inception-c"+str(index_)
            br1 = conv_relu(input_, 1, 256, (1, 1), 1, scope_)
            br2 = conv_relu(input_, 2, 384, (1, 1), 1, scope_)
            br2_1 = conv_relu(br2, 3, 256, (1, 3), 1, scope_)
            br2_2 = conv_relu(br2, 4, 256, (3, 1), 1, scope_)
            br2 = tf.concat([br2_1, br2_2], 3)
            br3 = conv_relu(input_, 5, 384, (1, 1), 1, scope_)
            br3 = conv_relu(br3, 6, 448, (3, 1), 1, scope_)
            br3 = conv_relu(br3, 7, 512, (1, 3), 1, scope_)
            br3_1 = conv_relu(br3, 8, 256, (1, 3), 1, scope_)
            br3_2 = conv_relu(br3, 9, 256, (3, 1), 1, scope_)
            br3 = tf.concat([br3_1, br3_2], 3)
            br4 = tf.layers.average_pooling2d(input_, pool_size=(3,3),
                    strides=(1,1), padding='SAME')
            br4 = conv_relu(br4, 10, 256, (1, 1), 1, scope_)
            merge_ = tf.concat([br1, br2, br3, br4], 3)
            return merge_
        def reduction_a(input_):
            scope_ = "reduction-a"
            br1 = conv_relu(input_, 1, 384, (3, 3), 2, scope_)
            br2 = conv_relu(input_, 2, 192, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 3, 224, (3, 3), 1, scope_)
            br2 = conv_relu(br2, 4, 256, (3, 3), 2, scope_)
            br3 = tf.layers.max_pooling2d(input_, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
            merge_ = tf.concat([br1, br2, br3], 3)
            return merge_
        def reduction_b(input_):
            scope_ = "reduction-b"
            br1 = conv_relu(input_, 1, 192, (1, 1), 1, scope_)
            br1 = conv_relu(br1, 2, 192, (3, 3), 2, scope_)
            br2 = conv_relu(input_, 3, 256, (1, 1), 1, scope_)
            br2 = conv_relu(br2, 4, 256, (1, 7), 1, scope_)
            br2 = conv_relu(br2, 5, 320, (7, 1), 1, scope_)
            br2 = conv_relu(br2, 6, 320, (3, 3), 2, scope_)
            br3 = tf.layers.max_pooling2d(input_, pool_size=(3,3),
                    strides=(2,2), padding='SAME')
            merge_ = tf.concat([br1, br2, br3], 3)
            return merge_
        #inception-A phase 8*8*384
        for i in range(4):
            c = inception_a(c, i)
        #reduction-A phase 4*4*1024
        c = reduction_a(c)
        #inception-B phase 4*4*1024
        for i in range(7):
            c = inception_b(c, i)
        #reduction-B phase 2*2*1536
        c = reduction_b(c)
        #reduction-C phase 2*2*1536
        for i in range(3):
            c = inception_c(c, i)
        c = tf.layers.average_pooling2d(c, pool_size=(2,2),
                strides=(2,2), padding='SAME')
        c = tf.layers.dropout(c, 0.2, training=self.training)
        c = tf.contrib.layers.flatten(c)
        c = tf.layers.dense(c, units=200, name="ip1",
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay))
        return {"logits": c}
