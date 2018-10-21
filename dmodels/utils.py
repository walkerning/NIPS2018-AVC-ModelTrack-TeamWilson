# -*- coding: utf-8 -*-

from __future__ import print_function

from functools import wraps
import tensorflow as  tf

def tf_vars_before_after(func):
    @wraps(func)
    def _func(model, *args, **kwargs):
        if not (hasattr(model, "name_space") and model.name_space):
            model._vars_before_model = tf.global_variables()
        res = func(model, *args, **kwargs)
        if not (hasattr(model, "name_space") and model.name_space):
            model._vars_after_model = tf.global_variables()
            for _var in model._vars_before_model:
                model._vars_after_model.remove(_var)
        return res
    return _func
