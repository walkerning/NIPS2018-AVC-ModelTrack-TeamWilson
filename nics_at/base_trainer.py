# -*- coding: utf-8 -*-
from __future__ import print_function

import copy

from datasets import get_dataset_cls

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
        if self.FLAGS.test_only:
            if self.FLAGS.dataset.startswith("gray_"):
                print("WARNINING: will not use gray dataset in test-only mode")
                self.FLAGS.dataset = self.FLAGS.dataset[5:]
        self.dataset = get_dataset_cls(self.FLAGS.dataset)(self.FLAGS)

    @classmethod
    def populate_arguments(cls, parser):
        pass
