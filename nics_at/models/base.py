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
        self.test_only = params.get("test_only", False)
        self.logits = None
        self.reuse = False
        self.namescope = namescope
        if self.test_only:
            self.training = False
        else:
            self.training = tf.placeholder_with_default(False, shape=())
        self.weight_decay = params.get("weight_decay", 0.0001)

    def get_training_status(self):
        return self.training

    def get_logits(self, inputs):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        if inputs in self.cached:
            [setattr(self, n, v) for n, v in self.cached[inputs].iteritems()]
            return self.cached[inputs]["logits"]
        if self.cached:
            self.reuse = True
        else:
            self.reuse = False
        with tf.variable_scope(self.namescope, reuse=self.reuse):
            res = self._get_logits(inputs)
        [setattr(self, n, v) for n, v in res.iteritems()]
        self.cached[inputs] = res
        return res["logits"]

    def get_probs(self, x):
        return tf.nn.softmax(self.get_logits(x))
