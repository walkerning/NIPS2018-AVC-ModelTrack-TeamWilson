# -*- coding: utf-8 -*-
from __future__ import print_function

import copy
import os

from datasets import Dataset

class settings(object):
    default_cfg = {}
    def __init__(self, config, args):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.update(config)
        config = cfg
        print("Configuration:\n" + "\n".join(["{:10}: {:10}".format(n, v) for n, v in config.items()]) + "\n")

        self.args = args
        self.dct = config

    def __getitem__(self, name):
        return getattr(self, name)

    def __getattr__(self, name):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        elif name in self.dct:
            return self.dct[name]
        else:
            raise KeyError("no attribute named {}".format(name))

class Trainer(object):
    def __init__(self, args, cfg):
        self.sess = None
        self.FLAGS = self._settings(cfg, args)
        self.dataset = Dataset(self.FLAGS.batch_size, self.FLAGS.epochs, self.FLAGS.aug_saltpepper, self.FLAGS.aug_gaussian,
                               generated_adv=self.FLAGS.generated_adv, num_threads=self.FLAGS.num_threads)

    @classmethod
    def populate_arguments(cls, parser):
        pass
