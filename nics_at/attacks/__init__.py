# -*- coding: utf-8 -*-

import contextlib

import numpy as np

from cleverhans.model import Model
import cleverhans.attacks
import foolbox
import foolbox.distances
import foolbox.attacks
from foolbox.models import TensorFlowModel

from nics_at.utils import AvailModels, profiling
from pgd_variants import MadryEtAl_L2, MadryEtAl_transfer
cleverhans.attacks.MadryEtAl_L2 = MadryEtAl_L2
cleverhans.attacks.MadryEtAl_transfer = MadryEtAl_transfer

@contextlib.contextmanager
def substitute_argscope(_callable, dct):
    if isinstance(_callable, type): # class
        _callable.old_init = _callable.__init__
        def new_init(self, *args, **kwargs):
            kwargs.update(dct)
            return self.old_init(*args, **kwargs)
        _callable.__init__ = new_init
        yield
        _callable.__init__ = _callable.old_init
    else: # function/methods
        raise Exception("not implemented")

class AttackGenerator(object):
    def __init__(self, generate_cfg, merge=False, split_adv=False, use_cache=False):
        self.cfg = generate_cfg
        self.merge = merge # whether or not to merge all adv into 1 array
        self.split_adv = split_adv # whether or not to split adv into multiple batch
        self.use_cache = use_cache
        self.batch_cache = {}
        self.epoch = 0
        self.batch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def new_batch(self):
        self.batch += 1
        self.batch_cache.clear()

    def new_epoch(self):
        self.epoch += 1

    def get_key(self, acfg):
        if "gid" in acfg:
            return acfg["gid"]
        else:
            params = acfg.get("attack_params", {})
            key = "-".join(["{}_{}".format(k, v) for k, v in sorted(params.items(), key=lambda pair: pair[0])])
            return acfg["id"] + ":" + key

    @profiling
    def generate_for_model(self, x, y, mid, pre_adv_x=None):
        cfg = self.cfg[mid]
        attacks = self.get_attacks(cfg)
        generated = []
        keys = []
        for a in attacks:
            if a["id"] is None:
                adv_x = x
                generated.append(adv_x)
                keys.append(a.get("gid", "normal"))
                continue
            key = a.get("gid", a["id"] + "-".join(["{}_{}".format(k, v) for k, v in sorted(a.get("attack_params", {}).items(), key=lambda pair: pair[0])]))
            keys.append(key)
            if self.use_cache:
                if key in self.batch_cache:
                    generated.append(self.batch_cache[key])
                else:
                    # "__generated__" is the magical key for pre-generated adv examples stored in filesystem(the configuration is in FLAGS["generated_adv"])
                    if "__generated__" in key:
                        adv_x = pre_adv_x
                    else:
                        adv_x = Attack.get_attack(a["id"]).generate(x, y)
                    self.batch_cache[key] = adv_x
                    generated.append(adv_x)
            else:
                if "__generated__" in key:
                    adv_x = pre_adv_x
                    if self.split_adv:
                        adv_x = list(adv_x.transpose((1, 0, 2, 3, 4)))
                        generated += adv_x
                        last_key = keys[-1]
                        keys = keys[:-1] + ["{}_split_{}".format(last_key, i) for i in range(len(adv_x))]
                    else:
                        generated.append(adv_x)
                else:
                    adv_x = Attack.get_attack(a["id"]).generate(x, y)
                    generated.append(adv_x)
        if self.merge:
            generated = [np.expand_dims(g, 1) if len(g.shape) == 4 else g for g in generated]
            generated = [np.concatenate(generated, axis=1)]
            keys = ["merge-" + "-".join(keys)]
        # reshape into [-1, 64, 64, 3]
        generated = [g.reshape([-1] + list(g.shape[-3:])) for g in generated]
        return keys, generated

    def epoch_mod(self, modn, leftn):
        return self.epoch % modn == leftn

    def batch_mod(self, modn, leftn):
        return self.batch % modn == leftn

    @profiling
    def meet_conds(self, conds):
        for cond in conds:
            if not eval("self." + cond):
                return False
        return True

    @profiling
    def get_attacks(self, cfg, epoch=None):
        choosed = []
        for acfg in cfg:
            if isinstance(acfg, (list, tuple)):
                avail_cfgs = [sacfg for sacfg in acfg if self.meet_conds(sacfg.get("conds", []))]
                if not avail_cfgs:
                    continue
                ratios = [sacfg.get("rel_ratio", 1.0) for sacfg in avail_cfgs]
                ratios = ratios / np.sum(ratios)
                randn = np.random.rand()
                idx = list(ratios > randn).index(True)
                choosed.append(avail_cfgs[idx])
            elif self.meet_conds(acfg.get("conds", [])):
                choosed.append(acfg)
        return choosed
        
class Attack(object):
    registry = {}

    def __init__(self, sess, cfg):
        self.cfg = cfg
        self.default_params = self.cfg["attack_params"]
        self.sess = sess

    @classmethod
    def get_attack(cls, aid):
        return cls.registry[aid]

    @classmethod
    def create_attack(cls, sess, cfg):
        _type = cfg.get("type", "cleverhans")
        # atk = cls.cls_registry[_type](sess, cfg)
        atk = globals()[_type.capitalize() + "Attack"](sess, cfg)
        cls.registry[atk.cfg["id"]] = atk
        return atk

    def generate(self, x, y, params={}):
        t_params = {k: v for k, v in self.default_params.iteritems()}
        t_params.update(params)
        return self._generate(x, y, t_params)

class FoolboxAttack(Attack):
    attack_methods = {
        "pgd": "PGD",
        "l2_pgd": "L2BasicIterativeAttack"
    }
    def __init__(self, sess, cfg):
        super(FoolboxAttack, self).__init__(sess, cfg)
        images, logits = AvailModels.get_model_io(self.cfg["model"])
        with sess.as_default():
            # model = AvailModels.get_model(self.cfg["model"])
            # images, logits = model.last_x, model.logits
            fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
        transfer_model = cfg.get("transfer", None)
        images, logits = AvailModels.get_model_io(transfer_model)
        with sess.as_default():
            transfer_fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
        if transfer_model is not None:
            # (predictions from blackbox, gradients from substitute)
            self.attack_model = foolbox.models.CompositeModel(
                forward_model=fmodel,
                backward_model=transfer_fmodel)
        else:
            self.attack_model = fmodel
        criterion = foolbox.criteria.Misclassification()
        self.attack = getattr(foolbox.attacks, self.attack_methods[cfg["method"]])(self.attack_model, criterion)

    def _generate(self, x_v, y_v, params):
        with substitute_argscope(foolbox.Adversarial, {"distance": foolbox.distances.Linf}):
            advs = [self.attack(sx_v, np.argmax(sy_v), binary_search=False, **params) for sx_v, sy_v in zip(x_v, y_v)]
        advs = [adv if adv is not None else sx_v for sx_v, adv in zip(x_v, advs)]
        return np.array(advs)

class CleverhansAttack(Attack):
    attack_methods = {
        "fgsm": "FastGradientMethod",
        "bim": "BasicIterativeMethod",
        "jsma": "SaliencyMapMethod",
        "cw": "CarliniWagnerL2",
        "pgd": "MadryEtAl",
        "transfer_pgd": "MadryEtAl_transfer",
        "l2_pgd": "MadryEtAl_L2",
        "momentum_pgd": "MomentumIterativeMethod"
    }
    def __init__(self, sess, cfg):
        super(CleverhansAttack, self).__init__(sess, cfg)
        if "transfer" in cfg:
            self.attack = getattr(cleverhans.attacks, self.attack_methods[self.cfg["method"]])(AvailModels.get_model(self.cfg["model"]), AvailModels.get_model(self.cfg["transfer"]), sess=sess)
        else:
            self.attack = getattr(cleverhans.attacks, self.attack_methods[self.cfg["method"]])(AvailModels.get_model(self.cfg["model"]), sess=sess)

    @profiling
    def _generate(self, x_v, y_v, params):
        attack_with_y = params.pop("attack_with_y", True)
        if attack_with_y:
            return self.attack.generate_np(x_v, y=y_v, **params)
        else:
            return self.attack.generate_np(x_v, **params)
