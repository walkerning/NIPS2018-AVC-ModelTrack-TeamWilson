# -*- coding: utf-8 -*-

import contextlib

import numpy as np

import cleverhans.attacks
import foolbox
import foolbox.distances
import foolbox.attacks
from foolbox.models import TensorFlowModel

from nics_at import utils
from nics_at.utils import AvailModels, profiling
from pgd_variants import MadryEtAl_L2, MadryEtAl_transfer, MadryEtAl_transfer_re, MadryEtAl_KLloss, MadryEtAl_L2_transfer_re
cleverhans.attacks.MadryEtAl_L2 = MadryEtAl_L2
cleverhans.attacks.MadryEtAl_transfer = MadryEtAl_transfer
cleverhans.attacks.MadryEtAl_transfer_re = MadryEtAl_transfer_re
cleverhans.attacks.MadryEtAl_L2_transfer_re = MadryEtAl_L2_transfer_re
cleverhans.attacks.MadryEtAl_KLloss = MadryEtAl_KLloss

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
    def __init__(self, generate_cfg, merge=False, split_adv=False, random_split_adv=False,
                 random_interp=None, random_interp_adv=None, use_cache=False,
                 mixup_alpha=1.0, name=""):
        self.name = name
        self.cfg = generate_cfg
        self.merge = merge # whether or not to merge all adv into 1 array
        self.split_adv = split_adv # whether or not to split adv into multiple batch
        self.random_split_adv = random_split_adv
        self.random_interp = random_interp # random interpolation between adv examples and normal examples
        self.random_interp_adv = random_interp_adv # random interpolation between adv examples
        self.mixup_alpha = mixup_alpha
        self.use_cache = use_cache
        self.batch_cache = {}
        self.epoch = 0
        self.batch = 0
        utils.log("AttackGenerator {}: split_adv: {}; random_split_adv: {}; random_interp: {}; random_interp_adv: {}; use_cache: {}".
                  format(self.name, self.split_adv, self.random_split_adv, self.random_interp, self.random_interp_adv, self.use_cache))

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
        cfg = self.cfg.get(mid, []) or []
        attacks = self.get_attacks(cfg)
        generated = []
        ys = []
        keys = []
        batch_size = x.shape[0]
        mixup_x = None
        mixup_y = None
        for a in attacks:
            normal_x = x
            normal_y = y
            if a.get("mixup", False): # pre-mixup normal x,y before attack
                # FIXME: for now, only mixup auged normal examples, not generated black-box adv examples;
                # NOTE: now, sample-level interpolation, can try batch-level too; only pre-mixup, not post-pixup now
                if mixup_x is None and mixup_y is None: # not cached mixup_x, mixup_y
                    # calculate mixup_x and mixup_y
                    weights = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size)
                    x_weights = weights.reshape((batch_size, 1, 1, 1))
                    y_weights = weights.reshape((batch_size, 1))
                    mixup_inds = np.random.permutation(batch_size)
                    x_1, x_2 = normal_x, normal_x[mixup_inds]
                    mixup_x = x_1 * x_weights + x_2 * (1 - x_weights)
                    y_1, y_2 = normal_y, normal_y[mixup_inds]
                    mixup_y = y_1 * y_weights + y_2 * (1 - y_weights)
                normal_x = mixup_x
                normal_y = mixup_y
            if a["id"] is None: # normal
                adv_x = normal_x
                keys.append(a.get("gid", "normal"))
                generated.append(adv_x)
                ys.append(y)
                continue
            key = a.get("gid", a["id"] + "-".join(["{}_{}".format(k, v) for k, v in sorted(a.get("attack_params", {}).items(), key=lambda pair: pair[0])]))
            keys.append(key)
            if "__generated__" in key:
                adv_x = pre_adv_x
                if self.random_interp is not None:
                    min_, max_ = self.random_interp
                    mult = min_ + np.random.rand(adv_x.shape[0], adv_x.shape[1], 1, 1, 1) * (max_ - min_)
                    adv_x = np.clip(np.expand_dims(x, 1) * (1-mult) + adv_x * mult, 0, 255)
                if self.split_adv and not self.random_split_adv:
                    adv_x = list(adv_x.transpose((1, 0, 2, 3, 4)))
                    generated += adv_x
                    last_key = keys[-1]
                    keys = keys[:-1] + ["{}_split_{}".format(last_key, i) for i in range(len(adv_x))]
                    ys = ys+ [y] * len(adv_x)
                else:
                    generated.append(adv_x)
                    ys.append(np.tile(np.expand_dims(y, 1), (1, adv_x.shape[1], 1)))
                if self.random_interp_adv is not None:
                    # NOTE: now use sample-level interpolation, can try batch-level too, might be more stable?
                    min_, max_ = self.random_interp_adv
                    breaks = np.vstack((np.random.rand(pre_adv_x.shape[1]-1, pre_adv_x.shape[0]), np.ones((1, pre_adv_x.shape[0]))))
                    weights = []
                    tmp_max = 1
                    for i in range(pre_adv_x.shape[1]):
                        w = min_ + breaks[i] * (tmp_max - min_)
                        weights.append(w)
                        tmp_max = tmp_max - w
                    np.random.shuffle(weights) # here weights is of size [len_adv, batch_size]
                    weights = np.transpose(weights).reshape((pre_adv_x.shape[0], pre_adv_x.shape[1], 1, 1, 1))
                    additional_adv_x = np.clip(np.sum(weights * pre_adv_x, axis=1), 0, 255)
                    generated.append(additional_adv_x)
                    ys.append(y)
                    keys.append("random_interp_advs")
            else: # if __generated__ not in key, generate white-box adversarials
                # white-box attack is the bottleneck of adversarial generation, use cache when needed
                if self.use_cache and key in self.batch_cache:
                    adv_x = self.batch_cache[key]
                else:
                    adv_x = Attack.get_attack(a["id"]).generate(normal_x, normal_y)
                    if self.use_cache:
                        self.batch_cache[key] = adv_x # cached
                generated.append(adv_x)
                ys.append(normal_y)

        if self.random_split_adv:
            generated = [np.expand_dims(g, 1) if len(g.shape) == 4 else g for g in generated]
            total = np.concatenate(generated, axis=1)
            ys = [np.expand_dims(s_y, 1) if len(s_y.shape) == 2 else s_y for s_y in ys]
            total_y = np.concatenate(ys, axis=1)
            num_split = total.shape[1]
            index = np.tile(np.arange(num_split)[np.newaxis], (x.shape[0], 1))
            [np.random.shuffle(_ind) for _ind in index]
            generated = [total[range(x.shape[0]), index[:,i]] for i in range(num_split)]
            ys = [total_y[range(x.shape[0]), index[:,i]] for i in range(num_split)]
            keys = ["random-split-{}".format(i) for i in range(num_split)]
        if self.merge:
            generated = [np.expand_dims(g, 1) if len(g.shape) == 4 else g for g in generated]
            generated = [np.concatenate(generated, axis=1)]
            ys = [np.expand_dims(s_y, 1) if len(s_y.shape) == 2 else s_y for s_y in ys]
            ys = [np.concatenate(ys, axis=1)]
            keys = ["merge-" + "-".join(keys)]
        # reshape into [-1, 64, 64, 3]
        generated = [g.reshape([-1] + list(g.shape[-3:])) for g in generated]
        ys = [s_y.reshape([-1, y.shape[-1]]) for s_y in ys]
        return keys, generated, ys

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

    def generate_tensor(self, x, y, params={}):
        t_params = {k: v for k, v in self.default_params.iteritems()}
        t_params.update(params)
        return self._generate_tensor(x, y, t_params)

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

    def _generate_tensor(self, x, y, params):
        # not implemented
        raise Exception("Not implemented")

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
        "re_transfer_pgd": "MadryEtAl_transfer_re",
        "l2_re_transfer_pgd": "MadryEtAl_L2_transfer_re",
        "l2_pgd": "MadryEtAl_L2",
        "momentum_pgd": "MomentumIterativeMethod",
        "kl_vat": "MadryEtAl_KLloss" # https://github.com/takerum/vat
    }
    def __init__(self, sess, cfg):
        super(CleverhansAttack, self).__init__(sess, cfg)
        if "transfer" in cfg:
            self.attack = getattr(cleverhans.attacks, self.attack_methods[self.cfg["method"]])(AvailModels.get_model(self.cfg["model"]), AvailModels.get_model(self.cfg["transfer"]), sess=sess)
        else:
            self.attack = getattr(cleverhans.attacks, self.attack_methods[self.cfg["method"]])(AvailModels.get_model(self.cfg["model"]), sess=sess)

    def _generate_tensor(self, x, y, params):
        return self.attack.generate(x, y=y, **params)

    @profiling
    def _generate(self, x_v, y_v, params):
        attack_with_y = params.pop("attack_with_y", True)
        if attack_with_y:
            return self.attack.generate_np(x_v, y=y_v, **params)
        else:
            return self.attack.generate_np(x_v, **params)
