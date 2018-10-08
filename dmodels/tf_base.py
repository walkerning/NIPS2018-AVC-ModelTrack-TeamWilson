# -*- coding: utf-8 -*-

import tensorflow as tf
slim = tf.contrib.slim

from foolbox.models import TensorFlowModel

from .base import BaseModel

class BaseTFModel(BaseModel):
    @classmethod
    def create_fmodel(cls, cfg):
        model = cls.create_model(cfg)
        sess = model.load_checkpoint(cfg["checkpoint"])
        with sess.as_default():
            fmodel = TensorFlowModel(model._images, model._logits, bounds=(0, 255))
        return fmodel
    
    @classmethod
    def create_model(cls, cfg):
        # graph = tf.Graph()
        graph = tf.get_default_graph()
        with graph.as_default():
            images = tf.placeholder(tf.float32, (None, 64, 64, 3))
            model = cls(**cfg["cfg"])
            logits = model(images, training=False)
        model.graph = graph
        model._logits = logits
        model._images = images
        return model

    def model_vars(self):
        if not hasattr(self, "_model_vars"):
            if self.name_space:
                self._model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_space)
            else:
                self._model_vars = self._vars_after_model
        return self._model_vars

    def load_checkpoint(self, path):
        with self.graph.as_default():
            with tf.variable_scope("utilities"):
                # self.saver = tf.train.Saver()
                self.saver = tf.train.Saver(self.model_vars())
        # sess = tf.Session(graph=self.graph)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.get_default_session() or tf.Session(graph=self.graph, config=config)
        print("load checkpoint from path: ", path)
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

    def load_checkpoint(self, path):
        with self.graph.as_default():
            with tf.variable_scope("utilities"):
                self.saver = tf.train.Saver(slim.get_model_variables())
        sess = tf.Session(graph=self.graph)
        self.saver.restore(sess, path)
        return sess
