# -*- coding: utf-8 -*-

import tensorflow as tf

def huber_relu(inputs, thresh):
    inputs = tf.maximum(inputs, 0)
    quadratic = tf.minimum(inputs, thresh)
    linear = (inputs - quadratic)
    return 0.5 * quadratic**2 + thresh * linear

def smooth_relu(inputs, thresh):
    inputs = tf.maximum(inputs, 0)
    return tf.where(inputs<thresh, inputs**2/thresh, inputs)

def thresh_relu(inputs, thresh):
    return tf.where(inputs > thresh, inputs, tf.zeros_like(inputs))

@tf.RegisterGradient("backthrough_thresh_gradient")
def _grad(op, output_grad):
    return tf.where(op.inputs[0]>0, tf.ones_like(op.inputs[0]), tf.zeros_like(op.inputs[0])) * output_grad

def backthrough_thresh_relu(inputs, thresh):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Where": "backthrough_thresh_gradient"}):
        return tf.where(inputs > thresh, inputs, tf.zeros_like(inputs))

# def get_adaptive_relu(relu_thresh, back_through=True):
#     if back_through:
#         return lambda inputs: backthrough_thresh_relu(inputs, relu_thresh)
#     else:
#         return lambda inputs: thresh_relu(inputs, relu_thresh)


def _pad(x, window_h, window_w):
    # Pad NCHW orderd images x to [window_h, window_w] size using 0s.
    pad_pre_h = (window_h - tf.shape(x)[2]) / 2
    pad_post_h = window_h - tf.shape(x)[2] - pad_pre_h
    pad_pre_w = (window_w - tf.shape(x)[3]) / 2
    pad_post_w = window_w - tf.shape(x)[3] - pad_pre_w
    paddings = [[pad_pre_h, pad_post_h], [pad_pre_w, pad_post_w]]
    return tf.pad(x, tf.concat([[[0, 0], [0, 0]], paddings], axis=0)), paddings

def _coarse_dropout(x, keep_prob, div_h, div_w): # NCHW
    # MAYBE(i think not necessary, as this situation does not occurs much): 
    #        When div_h >= height of x and div_w >= width of x,
    #        these padding and reshaping steps can be omited, as coarse_dropout will be equivalent to dropout; 
    window_h = tf.cast(tf.ceil(tf.cast(tf.shape(x)[2], tf.float32)/tf.cast(div_h, tf.float32)), tf.int32)
    window_w = tf.cast(tf.ceil(tf.cast(tf.shape(x)[3], tf.float32)/tf.cast(div_w, tf.float32)), tf.int32)
    pad_x, paddings = _pad(x, window_h * div_h, window_w * div_w)
    resize_x_h = tf.reshape(pad_x, tf.stack([tf.shape(x)[0], tf.shape(x)[1], div_h, window_h, tf.shape(pad_x)[3]]))
    resize_x_w = tf.transpose(tf.reshape(tf.transpose(resize_x_h,
                                                      [0, 1, 2, 4, 3]),
                                         tf.stack([tf.shape(x)[0], tf.shape(x)[1], div_h, div_w, window_w, window_h])),
                              [0, 1, 2, 3, 5, 4])
    dropout_resize_x = tf.nn.dropout(resize_x_w, keep_prob=keep_prob, noise_shape=tf.concat([tf.shape(resize_x_w)[:4], [1, 1]], axis=0))
    resize_back_x = tf.reshape(tf.transpose(dropout_resize_x, [0, 1, 2, 4, 3, 5]), tf.shape(pad_x))
    # remember to assert the shape
    return tf.reshape(resize_back_x[:, :, paddings[0][0]:tf.shape(pad_x)[2]-paddings[0][1], paddings[1][0]:tf.shape(pad_x)[3]-paddings[1][1]], tf.shape(x)) 

def coarse_dropout(x, keep_prob, div_h, div_w, training):
    return tf.cond(training, lambda: _coarse_dropout(x, keep_prob, div_h, div_w), lambda: x)
