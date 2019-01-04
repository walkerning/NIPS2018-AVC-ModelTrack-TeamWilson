# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import time
import random
from functools import wraps
import numpy as np

log = None

class AvailModels(object):
    # Singleton that store the references to all models
    registries = {None: {}}

    @classmethod
    def add(cls, model, x, y, tag=None, name=None):
        name = name or model.namescope
        cls.registries.setdefault(tag, {})[name] = (model, x, y)

    @classmethod
    def get_model(cls, mid, tag=None):
        try_default = cls.registries[tag].get(mid, None)
        if try_default is None and tag is None:
            # try all registries
            all_registries = reduce(lambda x, y: x.update(y) or x , cls.registries.values(), {})
            return all_registries[mid][0]
        else:
            return try_default[0]

    @classmethod
    def get_model_io(cls, mid, tag=None):
        return cls.registries[tag][mid][1:]

class LrAdjuster(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.adjust_type = cfg.get("adjust_type", "mult")
        assert self.adjust_type in {"mult", "add"}

        self.num_epoch = 0
        self.best_acc_epoch = 0
        self.best_acc = None
        self.accs = []
        self.improve_criterion = cfg.get("improve_criterion", "any")
        log("improve criterion: {}".format(self.improve_criterion))

    def add_and_check_improve(self, acc):
        self.num_epoch += 1
        acc = np.array(acc)
        self.accs.append(acc)
        if self.best_acc is None or self._is_improve(acc):
            print("improve at epoch {}".format(self.num_epoch))
            is_best = True
            if self.best_acc is None:
                self.best_acc = np.zeros(acc.shape)
            self.best_acc_epoch = self.num_epoch
            self.best_acc = np.maximum(acc, self.best_acc)
        else:
            is_best = False
            log("accs do not have improvements for {} epochs".format(self.num_epoch - self.best_acc_epoch))
        return is_best

    def _is_improve(self, acc):
        if self.improve_criterion == "any":
            return np.any(acc > self.best_acc)
        elif self.improve_criterion == "all":
            return np.all(acc > self.best_acc)
        else:
            assert isinstance(self.improve_criterion, (float, int))
            return np.mean(acc - self.best_acc) > self.improve_criterion

    @classmethod
    def create_adjuster(cls, cfg, name="learning_rate"):
        ins = globals()[cfg.get("type", "ExpDecay") + "Adjuster"](cfg)
        ins.name = name        
        return ins

    def set_status(self, best_acc=None, best_epoch=None, lr=None):
        if best_acc is not None:
            self.best_acc = best_acc
        if best_epoch is not None:
            self.best_acc_epoch = best_epoch
        if lr is not None:
            self.lr = lr
            
    def get_lr(self):
        return self.lr

    def adjust(self):
        if self.adjust_type == "mult":
            self.lr *= self.decay
        elif self.adjust_type == "add":
            self.lr += self.decay
        self.lr = min(max(self.lr, self.cfg.get("min", -np.inf)), self.cfg.get("max", np.inf))
        log("will decaying {} to {}".format(self.name, self.lr))

class ExpDecayAdjuster(LrAdjuster):
    def __init__(self, cfg):
        super(ExpDecayAdjuster, self).__init__(cfg)

        self.decay_every = cfg.get("decay_every", None)
        self.boundaries = cfg.get("boundaries", [])
        assert self.decay_every or self.boundaries
        self.lr = cfg["start_lr"]
        self.decay = cfg["decay"]

    def add_multiple_acc(self, *accs):
        return self.add(accs)

    def add(self, accs):
        is_best = self.add_and_check_improve(accs)
        if (self.decay_every and self.epoch % self.decay_every == 0) or (self.boundaries and self.epoch in self.boundaries):
            self.adjust()

        return is_best

class CosineLrAdjuster(LrAdjuster):
    # CosineLrAdjuster will not call LrAdjuster.adjust, as it's a different adjust method
    def __init__(self, cfg):
        super(CosineLrAdjuster, self).__init__(cfg)
        self.T_mult = cfg["T_mult"]
        self.lr_mult = cfg["lr_mult"]
        self.restart_every = cfg["restart_every"]
        self.eta_min = cfg["eta_min"]
        self.start_lr = self.base_lr = cfg["start_lr"]
        self.epoch = 0
        self.restarted_at = 0

    def add_multiple_acc(self, *accs):
        return self.add()

    def add(self, *accs):
        self.epoch += 1
        if self.epoch - self.restarted_at >= self.restart_every:
            self.restart()

    def restart(self):
        log("Restart at epoch: ", self.epoch)
        self.restart_every = int(self.restart_every * self.T_mult)
        self.base_lr = self.base_lr * self.lr_mult
        self.restarted_at = self.epoch

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.epoch - self.restarted_at) / self.restart_every)) / 2

class AccLrAdjuster(LrAdjuster):
    def __init__(self, cfg):
        super(AccLrAdjuster, self).__init__(cfg)
        self.decay_epoch_thre = cfg["decay_epoch_threshold"]
        self.end_epoch_thre = cfg["end_epoch_threshold"]
        self.lr = cfg["start_lr"]
        self.decay = cfg["decay"]

    def add_multiple_acc(self, *acc):
        is_best = self.add_and_check_improve(acc)
        # if or not to end training
        if self.end_epoch_thre and self.num_epoch - self.best_acc_epoch >= self.end_epoch_thre:
            self.lr = None
        elif self.num_epoch - self.best_acc_epoch >= self.decay_epoch_thre:
            self.adjust()
        return is_best

    # def add(self, acc):
    #     self.num_epoch += 1
    #     self.accs.append(acc)
    #     if self.best_acc is None or acc > self.best_acc:
    #         self.best_acc_epoch = self.num_epoch
    #         self.best_acc = acc
    #     # if or not to end training
    #     if self.num_epoch - self.best_acc_epoch > self.end_epoch_thre:
    #         self.lr = None
    #     elif self.num_epoch - self.best_acc_epoch > self.decay_epoch_thre:
    #         self.adjust()

class AccLrWithRestartAdjuster(AccLrAdjuster):
    def __init__(self, cfg):
        super(AccLrWithRestartAdjuster, self).__init__(cfg)
        self.restart_every = cfg.get("restart_every", None) # None means restart when reach max_lr

    def restart(self):
        self.lr = self.cfg["start_lr"]
        log("Restart {} to {}".format(self.name, self.lr))

    def adjust(self):
        if self.num_epoch:
            if self.restart_every is None:
                if self.lr == self.get("max", np.inf) or self.lr == self.get("min", -np.inf):
                    self.restart()
                    return
            elif self.num_epoch % self.restart_every == 0:
                self.restart()
                return
        super(AccLrWithRestartAdjuster, self).adjust()

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

def get_tensor_dependencies(tensor):
    dependencies = set()
    dependencies.update(tensor.op.inputs)
    for sub_op in tensor.op.inputs:
        dependencies.update(get_tensor_dependencies(sub_op))
    return dependencies

def get_schedule_value(schedule, epoch, step, steps_per_epoch):
    if isinstance(schedule, (float, int)):
        return schedule
    if schedule.get("type") == "add":
        v = schedule["start"] + epoch // schedule["every"] * schedule["step"]
    elif schedule.get("type") == "mult":
        v = schedule["start"] * schedule["step"] ** (epoch // schedule["every"])
    return min(max(v, schedule.get("min", np.inf)), schedule.get("max", np.inf))
