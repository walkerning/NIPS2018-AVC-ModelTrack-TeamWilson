# -*- coding: utf-8 -*-

import tensorflow as tf

def thresh_relu(inputs, thresh):
    return tf.where(inputs > thresh, inputs, tf.zeros_like(inputs))

@tf.RegisterGradient("backthrough_thresh_gradient")
def _grad(op, output_grad):
    return tf.where(op.inputs[0]>0, tf.ones_like(op.inputs[0]), tf.zeros_like(op.inputs[0])) * output_grad

def backthrough_thresh_relu(inputs, thresh):
    G = tf.get_default_graph()
    with G.gradient_override_map({"Where": "backthrough_thresh_gradient"}):
        return tf.where(inputs > thresh, inputs, tf.zeros_like(inputs))

def get_adaptive_relu(relu_thresh, back_through=True):
    if back_through:
        return lambda inputs: backthrough_thresh_relu(inputs, relu_thresh)
    else:
        return lambda inputs: thresh_relu(inputs, relu_thresh)
