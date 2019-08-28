# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ..pytorch_base import BasePyTorchModel
from aw_nas.final.base import FinalModel
from aw_nas.common import get_search_space

class Model(BasePyTorchModel, nn.Module):
    def load_checkpoint(self, path, load_name_space=None):
        if self.model is None:
            self.model = torch.load(path)
        else:
            super(Model, self).load_checkpoint(path, load_name_space, dict_key=None)
            self.model.to(self.model.device)

    def __init__(self, model_type, # NOTE: model_type is not used
                 search_space_type=None, search_space_cfg=None,
                 final_model_type=None, final_model_cfg=None,
                 substract_mean=[125.307, 122.961, 113.8575], div=[62.993, 62.089, 66.705],
                 data_format="channels_first"):
        BasePyTorchModel.__init__(self)
        nn.Module.__init__(self)

        if final_model_type:
            self.search_space = get_search_space(search_space_type, **search_space_cfg)
            assert final_model_cfg is not None
            self.model = FinalModel.get_class_(final_model_type)(
                self.search_space,
                torch.device("cuda"),
                **final_model_cfg
            )
        self.data_format = data_format
        self.substract_mean = torch.tensor(substract_mean, requires_grad=False)
        self.div = torch.tensor(div, requires_grad=False)

    def __call__(self, inputs):
        assert self.model is not None
        inputs = inputs.float()
        inputs = (inputs - self.substract_mean.to(inputs.device).type_as(inputs)) / self.div.to(inputs.device).type_as(inputs)
        if self.data_format == "channels_first":
            inputs = inputs.permute(0, 3, 1, 2)
        self.model.eval()
        return self.model(inputs)
