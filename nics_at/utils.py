# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import time
import random
from functools import wraps
import numpy as np

class AvailModels(object):
    # Singleton that store the references to all models
    registry = {}

    @classmethod
    def add(cls, model, x, y):
        cls.registry[model.namescope] = (model, x, y)

    @classmethod
    def get_model(cls, mid):
        return cls.registry[mid][0]

    @classmethod
    def get_model_io(cls, mid):
        return cls.registry[mid][1:]

class LrAdjuster(object):
    @classmethod
    def create_adjuster(cls, cfg):
        return globals()[cfg.get("type", "ExpDecay") + "Adjuster"](cfg)

    def get_lr(self):
        return self.lr

class ExpDecayAdjuster(LrAdjuster):
    def __init__(self, cfg):
        self.decay_every = cfg.get("decay_every", None)
        self.boundaries = cfg.get("boundaries", [])
        assert self.decay_every or self.boundaries
        self.lr = cfg["start_lr"]
        self.decay = cfg["decay"]
        self.epoch = 0

    def add_multiple_acc(self, *accs):
        self.add()

    def add(self, *accs):
        self.epoch += 1
        if (self.decay_every and self.epoch % self.decay_every == 0) or (self.boundaries and self.epoch in self.boundaries):
            self.lr *= self.decay
            log("will decaying lr to {}".format(self.lr))

class AccLrAdjuster(LrAdjuster):
    def __init__(self, cfg):
        self.decay_epoch_thre = cfg["decay_epoch_threshold"]
        self.end_epoch_thre = cfg["end_epoch_threshold"]
        self.lr = cfg["start_lr"]
        self.decay = cfg["decay"]

        self.num_epoch = 0
        self.best_acc_epoch = 0
        self.best_acc = None
        self.accs = []

    def add_multiple_acc(self, *acc):
        self.num_epoch += 1
        acc = np.array(acc)
        self.accs.append(acc)
        # if np.all(acc > self.best_acc):
        if self.best_acc is None or np.any(acc > self.best_acc):
            if self.best_acc is None:
                self.best_acc = np.zeros(acc.shape)
            self.best_acc_epoch = self.num_epoch
            self.best_acc = np.maximum(acc, self.best_acc)
        # if or not to end training
        if self.num_epoch - self.best_acc_epoch > self.end_epoch_thre:
            self.lr = None
        elif self.num_epoch - self.best_acc_epoch > self.decay_epoch_thre:
            self.lr *= self.decay
            log("will decaying lr to {}".format(self.lr))

    def add(self, acc):
        self.num_epoch += 1
        self.accs.append(acc)
        if self.best_acc is None or acc > self.best_acc:
            self.best_acc_epoch = self.num_epoch
            self.best_acc = acc
        # if or not to end training
        if self.num_epoch - self.best_acc_epoch > self.end_epoch_thre:
            self.lr = None
        elif self.num_epoch - self.best_acc_epoch > self.decay_epoch_thre:
            self.lr *= self.decay
            log("will decaying lr to {}".format(self.lr))

def get_log_func(log_file):
    def log(*args, **kwargs):
        flush = kwargs.pop("flush", None)
        if log_file is not None:
            print(*args, file=log_file, **kwargs)
            if flush:
                log_file.flush()
        print(*args, **kwargs)
        if flush:
            sys.stdout.flush()
    return log

PROFILING = os.environ.get("NICS_AT_PROFILING", False)
all_profiled = {}
def profiling(func):
    if not PROFILING:
        return func
    name = func.__name__
    while name in all_profiled:
        name = func.__name__ + "_" + str(random.randint(0, 100))
    all_profiled[name] = [0, 0]
    @wraps(func)
    def _func(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        all_profiled[name][0] += 1
        all_profiled[name][1] += time.time() - start
        return res
    return _func
