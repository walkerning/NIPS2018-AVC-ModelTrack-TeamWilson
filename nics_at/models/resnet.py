# -*- coding: utf-8 -*-
import tensorflow as tf

from nics_at.models.base import QCNN

_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2

class Resnet(QCNN):
    TYPE = "resnet18"
    def __init__(self, namescope, params={}):
        super(Resnet, self).__init__(namescope, params)

        self.resnet_size = 18

        self.block_fn = self._building_block_v2
        self.bottleneck = False
        self.data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
        self.num_classes = 200
        self.num_filters = 64
        self.kernel_size = 3
        self.conv_stride = 1
        self.first_pool_size = 0
        self.first_pool_stride = 2
        self.second_pool_size = 7
        self.second_pool_stride = 1
        self.block_sizes = [2, 2, 2, 2]
        self.block_strides = [1, 2, 2, 2]
        self.final_size = 512
        self.more_blocks = params.get("more_blocks", False)
        self.batch_norm_momentum = params.get("batch_norm_momentum", 0.997)
        print("weight decay: {}; batch norm momentum: {}".format(self.weight_decay, self.batch_norm_momentum))

    def batch_norm(self, inputs, training, data_format):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
                inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
                momentum=self.batch_norm_momentum, epsilon=_BATCH_NORM_EPSILON, center=True,
                scale=True, training=training, fused=True)


    def fixed_padding(self, inputs, kernel_size, data_format):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
                [batch, height_in, width_in, channels] depending on data_format.
            kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                                     Should be a positive integer.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            A tensor with the same format as the input with the data either intact
            (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                        [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                        [pad_beg, pad_end], [0, 0]])
        return padded_inputs


    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)

        if self.more_blocks:
            factor_ = 0.5
        else:
            factor_ = 2.0
        return tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(scale=factor_),
                data_format=data_format)


    def _building_block_v2(self, inputs, filters, training, projection_shortcut, strides,
                                                 data_format):
        shortcut = inputs
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        # if self.reuse == False:
        self.relu_list.append(tf.reduce_mean(inputs, [2,3]))

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                data_format=data_format)

        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        # if self.reuse == False:
        self.relu_list.append(tf.reduce_mean(inputs, [2,3]))
        inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=1,
                data_format=data_format)

        return inputs + shortcut

    def block_layer(self, inputs, filters, bottleneck, block_fn, blocks, strides,
                    training, name, data_format, more_blocks=False):

        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = filters * 4 if bottleneck else filters

        def projection_shortcut(inputs):
            return self.conv2d_fixed_padding(
                    inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                    data_format=data_format)

        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                                            data_format)

        for _ in range(1, blocks):
            inputs = block_fn(inputs, filters, training, None, 1, data_format)

        if more_blocks:
            with tf.variable_scope("more_blocks"):
                with tf.variable_scope(name):
                    inputs = block_fn(inputs, filters, training, None, 1, data_format)

        return tf.identity(inputs, name)

    def _get_logits(self, inputs):
        relu_list = self.relu_list = []
        group_list = self.group_list = []
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        inputs = inputs - tf.constant(_CHANNEL_MEANS)
        weight_decay = self.weight_decay
        if self.data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                strides=self.conv_stride, data_format=self.data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        if self.first_pool_size:
            inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_max_pool')

        for i, num_blocks in enumerate(self.block_sizes):
            num_filters = self.num_filters * (2**i)
            inputs = self.block_layer(
                inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                block_fn=self.block_fn, blocks=num_blocks,
                strides=self.block_strides[i], training=self.training,
                name='block_layer{}'.format(i + 1), data_format=self.data_format,
                more_blocks=self.more_blocks)
            # if self.reuse == False:
            group_list.append(inputs)

        inputs = self.batch_norm(inputs, self.training, self.data_format)
        inputs = tf.nn.relu(inputs)
        # if self.reuse == False:
        relu_list.append(tf.reduce_mean(inputs, [2,3]))

        # The current top layer has shape
        # `batch_size x pool_size x pool_size x final_size`.
        # ResNet does an Average Pooling layer over pool_size,
        # but that is the same as doing a reduce_mean. We do a reduce_mean
        # here because it performs better than AveragePooling2D.
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        inputs = tf.reduce_mean(inputs, axes, keep_dims=True)
        inputs = tf.identity(inputs, 'final_reduce_mean')

        inputs = tf.reshape(inputs, [-1, self.final_size])
        self.layer_f1 = inputs
        readout_layer = tf.layers.Dense(
                units=self.num_classes,
                name='readout_layer')
        inputs = readout_layer(inputs)
        inputs = tf.identity(inputs, 'final_dense')

        return {
            "logits": inputs,
            "group_list": group_list,
            "relu_list": relu_list
        }

    def get_saver(self, load_namescope=None, prepend=None):
        if self.more_blocks:
            vars_except_more = [var for var in self.vars if "more_blocks/" not in var.op.name]
        else:
            vars_except_more = self.vars
        var_namescope = self.namescope if not prepend else prepend + "/" + self.namescope
        if load_namescope is None or load_namescope == var_namescope:
            saver = tf.train.Saver(vars_except_more, max_to_keep=20)
        else:
            var_mapping_dct = {var.op.name.replace(var_namescope + "/", (load_namescope + "/") if load_namescope else ""): var for var in vars_except_more}
            saver = tf.train.Saver(var_mapping_dct, max_to_keep=20)
        return saver
