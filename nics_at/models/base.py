# -*- coding: utf-8 -*-
import six
import abc

import tensorflow as tf

from cleverhans.model import Model

def RegistryMetaFactory(registry_name):
    class _RegistryMeta(abc.ABCMeta):
        registry = {}
        def create_model(cls, cfg):
            return cls.registry[cfg["type"]](cfg["namescope"], params=cfg.get("model_params", {}))

        def __new__(mcls, name, bases, attrs):
            parents = [b for b in bases if isinstance(b, _RegistryMeta)]
            if not parents:
                # BaseStrategy do not need to be registered.
                cls = super(_RegistryMeta, mcls).__new__(mcls, name, bases, attrs)
                return cls
            reg_name = attrs.get("TYPE", name.lower())
            cls = super(_RegistryMeta, mcls).__new__(mcls, name, bases, attrs)
            _RegistryMeta.registry[reg_name] = cls
            return cls
    _RegistryMeta.__name__ = "_RegistryMeta_{}".format(registry_name)
    return _RegistryMeta

@six.add_metaclass(RegistryMetaFactory("model"))
class QCNN(Model):
    def __init__(self, namescope, params={}):
        super(Model, self).__init__()
        self.cached = {}

        self.params = params
        self.test_only = params.get("test_only", False)
        self.logits = None
        self.reuse = False
        self.namescope = namescope
        if self.test_only:
            self.training = False
        else:
            self.training = tf.placeholder_with_default(False, shape=())
        self.weight_decay = params.get("weight_decay", 0.0001)
        self.output_name = params.get("output_name", "logits")

        self._vars = []
        self._trainable_vars = []
        self._save_saver = None

        # Parse patch_relu config
        patch_relu = params.get("patch_relu", None)
        self.patch_relu = None
        if patch_relu is not None:
            self.relu_thresh = None
            with tf.variable_scope(self.namescope):
                self.relu_thresh = tf.get_variable("relu_thresh", shape=[], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
            # from nics_at.tf_utils import get_adaptive_relu
            # if patch_relu in {"thresh", "backthrough_thresh"}:
            #     self.patch_relu = get_adaptive_relu(self.relu_thresh, back_through=patch_relu=="backthrough_thresh")
            # else:
            from nics_at import tf_utils
            relu_func = getattr(tf_utils, patch_relu + "_relu")
            self.patch_relu = lambda inputs: relu_func(inputs, self.relu_thresh)
            self._vars.append(self.relu_thresh)

    def _all_update_ops(self):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.namescope)

    @property
    def update_ops(self):
        only_updatable = self.params.get("updatable", None)
        update_ops = self._all_update_ops()
        if only_updatable is None:
            # must have namescope here
            return update_ops
        else:
            return [op for op in update_ops if any(pattern in op.name for pattern in only_updatable)]

    @property
    def trainable_vars(self):
        only_trainable = self.params.get("trainable", None)
        if only_trainable is None:
            return self._trainable_vars
        else:
            return [var for var in self._trainable_vars if any(pattern in var.op.name for pattern in only_trainable)]

    @property
    def vars(self):
        return self._vars

    def get_training_status(self):
        return self.training

    def get_logits(self, inputs, output_name=None):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        output_name = output_name or self.output_name
        if inputs in self.cached:
            [setattr(self, n, v) for n, v in self.cached[inputs].iteritems()]
            return self.cached[inputs][output_name]
        if self.cached:
            self.reuse = True
        else:
            self.reuse = False
        if not self.reuse:
            _before_vars = tf.global_variables()
            if not self.test_only:
                _before_t_vars = tf.trainable_variables()
        with tf.variable_scope(self.namescope, reuse=self.reuse):
            if self.patch_relu is not None: # patch tf.nn.relu to another relu func
                _backup_relu = tf.nn.relu
                tf.nn.relu = self.patch_relu
                res = self._get_logits(inputs)
                tf.nn.relu = _backup_relu
            else:
                res = self._get_logits(inputs)
        [setattr(self, n, v) for n, v in res.iteritems()]
        self.cached[inputs] = res
        _after_vars = tf.global_variables()
        if not self.reuse:
            for var_ in _before_vars:
                if var_ in _after_vars:
                    _after_vars.remove(var_)
            self._vars += _after_vars
            if not self.test_only:
                _after_t_vars = tf.trainable_variables()
                for var_ in _before_t_vars:
                    if var_ in _after_t_vars:
                        _after_t_vars.remove(var_)
                self._trainable_vars += _after_t_vars
        return res[output_name]

    def get_probs(self, x):
        return tf.nn.softmax(self.get_logits(x))

    def get_saver(self, load_namescope=None, prepend=None, exclude_pattern=[]):
        _vars = [v for v in self.vars if all(p not in v.op.name for p in exclude_pattern)]
        var_namescope = self.namescope if not prepend else prepend + "/" + self.namescope
        if load_namescope is None or load_namescope == var_namescope:
            saver = tf.train.Saver(_vars, max_to_keep=20)
        else:
            var_mapping_dct = {var.op.name.replace(var_namescope + "/", (load_namescope + "/") if load_namescope else ""): var for var in _vars}
            saver = tf.train.Saver(var_mapping_dct, max_to_keep=20)
        return saver

    def get_save_saver(self, prepend=None):
        if not self._save_saver:
            if prepend is None:
                saver = tf.train.Saver(self.vars, max_to_keep=20)
            else:
                var_mapping_dct = {var.op.name.replace(prepend + "/", ""): var for var in self.vars}
                saver = tf.train.Saver(var_mapping_dct, max_to_keep=20)
            self._save_saver = saver
        return self._save_saver

    def load_checkpoint(self, path, sess, load_namescope=None, prepend_namescope=None, exclude_pattern=[]):
        self.saver = self.get_saver(load_namescope, prepend=prepend_namescope, exclude_pattern=exclude_pattern)
        print("Load model from", path)
        self.saver.restore(sess, path)

    def save_checkpoint(self, path, sess, prepend_namescope=None):
        self.get_save_saver(prepend=prepend_namescope).save(sess, path)

class QCNNProxy(Model): # Patch get_logits func, and proxy all other attribute to proxy_model
    def __init__(self, proxy_model, patch_get_logits):
        super(Model, self).__init__()
        self.proxy_model = proxy_model
        self.get_logits = patch_get_logits.__get__(self)

    def __getattr__(self, name):
        return getattr(self.proxy_model, name)
