# -*- coding: utf-8 -*-

import torch
from foolbox.models import PyTorchModel

from .base import Model

class BasePyTorchModel(Model):
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

    @classmethod
    def create_fmodel(cls, cfg):
        model = cls(**cfg["cfg"])
        fmodel = PyTorchModel(model, bounds=(0, 255), num_classes=cfg["num_classes"])
        return fmodel

