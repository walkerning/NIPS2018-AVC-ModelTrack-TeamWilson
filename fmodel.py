# -*- coding: utf-8 -*-
import os
import importlib

import yaml
import numpy as np

from foolbox.models import Model
from foolbox.models import DifferentiableModel

class EnsembleModel(DifferentiableModel):
    def num_classes(self):
        pass

    def __init__(self, models, cfgs, bounds=(0, 255), channel_axis=3, preprocessing=(0, 1)):
        super(EnsembleModel, self).__init__(bounds, channel_axis, preprocessing)
        self.models = models
        self.cfgs = cfgs
        self.num_classes = models[0].num_classes
        self.names = [cfg.get("name", cfg["type"]) for cfg in cfgs]
        self.weights = np.array([cfg.get("weight", 1.0) for cfg in cfgs])
        self.weights = self.weights / np.sum(self.weights)

    def batch_predictions(self, images):
        predictions = [m.batch_predictions(images) for m in self.models]
        return np.sum(self.weights.reshape(len(self.models), 1, 1) * predictions, axis=0)

    def predictions_and_gradient(self, image, label):
        predictions, gradients = zip(*[m.predictions_and_gradient(image, label) for m in self.models])
        predictions = np.array(predictions); gradients = np.array(gradients)
        assert predictions.ndim == 2
        assert gradients.shape[1:] == image.shape
        predictions = np.sum(self.weights[:, np.newaxis] * predictions, axis=0)
        gradients = np.sum(self.weights[:, np.newaxis, np.newaxis, np.newaxis] * gradients, axis=0)

        return predictions, gradients

def create_fmodel():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_fname = os.environ.get("FMODEL_MODEL_CFG", "model.yaml")
    print("Load fmodel cfg from {}".format(cfg_fname))
    with open(cfg_fname, "r") as f:
        model_cfg = yaml.load(f)
    models = []
    for m in model_cfg["models"]:
        mod = importlib.import_module("dmodels." + m["type"])
        assert hasattr(mod, "Model"), "model package must have an attribute named Model"
        models.append(mod.Model.create_fmodel(m))
    fmodel = EnsembleModel(models, model_cfg["models"])
    return fmodel

if __name__ ==  "__main__":
    # executable for debuggin and testing
    print(create_fmodel())
