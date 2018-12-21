# -*- coding: utf-8 -*-
import os
import tensorflow as tf
slim = tf.contrib.slim

from foolbox.models import TensorFlowModel

from .base import BaseModel

class BaseTFModel(BaseModel):
    @classmethod
    def create_fmodel(cls, cfg):
        model = cls.create_model(cfg)
        model.sess = model.load_checkpoint(cfg["checkpoint"], cfg.get("load_name_space", None))
        with model.sess.as_default():
            fmodel = TensorFlowModel(model._images, model._logits, bounds=(0, 255))
        fmodel.model = model
        return fmodel
    
    @classmethod
    def create_model(cls, cfg):
        # graph = tf.Graph()
        graph = tf.get_default_graph()
        with graph.as_default():
            image_shape = cfg["cfg"].pop("image_shape", [64, 64, 3])
            images = tf.placeholder(tf.float32, tuple([None] + list(image_shape)))
            model = cls(**cfg["cfg"])
            logits = model(images, training=False)
        model.graph = graph
        model._logits = logits
        model._images = images
        return model

    def model_vars(self):
        if not hasattr(self, "_model_vars"):
            if self.name_space:
                self._model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_space + "/")
            else:
                self._model_vars = self._vars_after_model
        return self._model_vars

    def _subname(self, name_space, l_name_space, name):
        if not name_space:
            return l_name_space + "/" + name
        else:
            return name.replace(name_space + "/", l_name_space + ("/" if l_name_space else ""))

    def saver_mapping(self, load_name_space):
        if load_name_space is not None and load_name_space != self.name_space:
            return {self._subname(self.name_space, load_name_space, v.op.name): v for v in self.model_vars()}
        else:
            return self.model_vars()

    def load_checkpoint(self, path, load_name_space=None):
        with self.graph.as_default():
            with tf.variable_scope("utilities"):
                self.saver = tf.train.Saver(self.saver_mapping(load_name_space))
        # sess = tf.Session(graph=self.graph)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.get_default_session() or tf.Session(graph=self.graph, config=config)
        print("load checkpoint from path: ", path)
        if os.path.isdir(path):
            path = tf.train.latest_checkpoint(path)
        self.saver.restore(sess, path)
        return sess

class TFSlimModel(BaseTFModel):
    pass
    # def load_checkpoint(self, path):
    #     with self.graph.as_default():
    #         self.saver = tf.train.Saver(slim.get_model_variables())
    #     # session_creator = tf.train.ChiefSessionCreator(
    #     #     scaffold=tf.train.Scaffold(saver=self.saver),
    #     #     checkpoint_filename_with_path=path)

    #     # sess = tf.train.MonitoredSession(session_creator=session_creator)
    #     return sess

    def model_vars(self):
        return slim.get_model_variables()

    # def load_checkpoint(self, path, load_name_space=None):
    #     with self.graph.as_default():
    #         with tf.variable_scope("utilities"):
    #             self.saver = tf.train.Saver(slim.get_model_variables())
    #     sess = tf.Session(graph=self.graph)
    #     self.saver.restore(sess, path)
    #     return sess
