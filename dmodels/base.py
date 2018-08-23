# -*- coding: utf-8 -*-

class BaseModel(object):
    def load_checkpoint(self, path):
        pass

    def __call__(self, inputs, training):
        pass

    @classmethod
    def create_fmodel(cls, cfg):
        pass
