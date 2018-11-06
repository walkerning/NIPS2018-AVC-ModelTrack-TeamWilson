# -*- coding: utf-8 -*-
import os
import importlib

import tensorflow as tf
from ..tf_base import BaseTFModel
from ..utils import tf_vars_before_after

class SeqTFModel(BaseTFModel):
    def __init__(self, cfg_list):
        super(SeqTFModel, self).__init__()
        self.cfg_list = cfg_list
        self.models = []
        for cfg in self.cfg_list:
            mod = importlib.import_module("dmodels." + cfg["type"])
            model = mod.Model(**cfg["cfg"])
            self.models.append(model)
    
    def load_checkpoint(self, paths, load_name_spaces):
        paths = paths or [cfg["checkpoint"] for cfg in self.cfg_list]
        load_name_spaces = load_name_spaces or [cfg.get("load_name_space", None) for cfg in self.cfg_list]
        savers = [tf.train.Saver(model.saver_mapping(lns)) for model, lns in zip(self.models, load_name_spaces)]
        sess = tf.Session()
        for saver, path in zip(savers, paths):
            if os.path.isdir(path):
                path = tf.train.latest_checkpoint(path)
            saver.restore(sess, path)
        return sess

    @tf_vars_before_after
    def __call__(self, inputs, training):
        for model in self.models:
            inputs = model(inputs, training)
        return inputs
        

Model = SeqTFModel
