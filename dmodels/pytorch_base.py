# -*- coding: utf-8 -*-

import torch
from foolbox.models import PyTorchModel

from .base import BaseModel

class BasePyTorchModel(BaseModel):
    def load_checkpoint(self, path, load_name_space=None, dict_key="state_dict"):
        checkpoint = torch.load(path)
        if isinstance(checkpoint, dict) and dict_key in checkpoint:
            state_dct = checkpoint[dict_key]
        else:
            state_dct = checkpoint
        if load_name_space is not None:
            load_name_space += "."
            state_dct = {k[len(load_name_space):] if k.startswith(load_name_space) else k: v for k, v in state_dct.items()}
        self.model.load_state_dict(state_dct)

    @classmethod
    def create_fmodel(cls, cfg):
        model = cls(**cfg["cfg"])
        model.load_checkpoint(cfg["checkpoint"], load_name_space=cfg.get("load_name_space", None))
        # pytorch models usually receive input of bounds (0,1) due to tansform.ToTensor
        fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=cfg["num_classes"])
        return fmodel

