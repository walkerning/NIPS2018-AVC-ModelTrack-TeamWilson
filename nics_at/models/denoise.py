# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import tensorflow as tf

from nics_at.models.base import QCNN
from nics_at import utils

class _Denoiser(QCNN):
    TYPE = "prepend_denoiser"
    def __init__(self, namescope, params):
        super(_Denoiser, self).__init__(namescope, params)
        self.forward = []
        self.backward = []
        self.fwd_out = [64, 128, 256, 256, 256]
        self.num_fwd = [2, 3, 3, 3, 3]
        self.back_out = [64, 128, 256, 256]
        self.num_back = [2, 3, 3, 3]
        assert self.output_name == "denoise_output"

    def _get_logits(self, input_):
        self.forward = []
        self.backward = []
        # this denoiser is channel_last
        def conv_bn_relu(input_, index_, filters_, stride_ = 1):
            conv_ = tf.layers.conv2d(input_, filters=filters_, kernel_size=(3,3),
                                     strides=(stride_,stride_), padding="same", use_bias=False, name="conv"+str(index_),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            bn_ = tf.contrib.layers.batch_norm(conv_, is_training=self.training, scale=True, scope="bn"+str(index_), decay=0.9)
            relu_ = tf.nn.relu(bn_, name="relu"+str(index_))
            return relu_
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
        x = tf.layers.conv2d(x, filters=3, kernel_size=(1,1), padding="same", use_bias=False, name="last_conv",
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay),
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        self.denoise_output = x + input_
        return {
            "denoise_output": self.denoise_output,
            "forward": self.forward,
            "backward": self.backward
        }

class DenoiseNet(QCNN):
    TYPE = "denoise"
    def __init__(self, namescope, params):
        super(DenoiseNet, self).__init__(namescope, params)
        self.denoiser = QCNN.create_model(params["denoiser"])
        self.inner_model = QCNN.create_model(params["model"])

    @property
    def trainable_vars(self):
        return self.denoiser.trainable_vars + self.inner_model.trainable_vars

    def get_training_status(self):
        return self.denoiser.training

    def _get_logits(self, inputs): # FIXME: should cache in denoiser
        denoise_output = self.denoiser.get_logits(inputs)
        self.logits = self.inner_model.get_logits(denoise_output)
        return {
            "logits": self.logits
        }

    def load_checkpoint(self, path, sess, load_namescope=[None, None]):
        assert len(path) == 2 and len(load_namescope) == 2
        utils.log("Load denoiser/inner from ", path)
        if path[0]:
            self.denoiser.load_checkpoint(path[0], sess,
                                          load_namescope[0],
                                          prepend_namescope=self.namescope)
        if path[1]:
            self.inner_model.load_checkpoint(path[1], sess,
                                             load_namescope[1],
                                             prepend_namescope=self.namescope)

    def save_checkpoint(self, path, sess):
        if not self.params["denoiser"].get("model_params", {}).get("test_only", False):
            denoiser_path = path + "_denoiser"
            self.denoiser.save_checkpoint(denoiser_path, sess, prepend_namescope=self.namescope)
            utils.log("Saved denoiser to ", denoiser_path)
        if not self.params["model"].get("model_params", {}).get("test_only", False):
            model_path = path + "_model"
            self.inner_model.save_checkpoint(model_path, sess, prepend_namescope=self.namescope)
            utils.log("Saved denoiser inner model to ", model_path)

