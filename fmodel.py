# -*- coding: utf-8 -*-
import os
import importlib

import yaml
import numpy as np
import scipy
import scipy.ndimage
from scipy.misc import imresize

from foolbox.models import Model
from foolbox.models import DifferentiableModel

class EnsembleModel(DifferentiableModel):
    def num_classes(self):
        pass

    def __init__(self, models, cfgs, ensemble_type="weight", multi_angle=[], use_resolution=[], multi_crop=[], multi_type="mean", bounds=(0, 255), channel_axis=3, preprocessing=(0, 1)):
        super(EnsembleModel, self).__init__(bounds, channel_axis, preprocessing)
        self.models = models
        self.cfgs = cfgs
        self.ensemble_type = ensemble_type
        self.multi_angle = multi_angle
        self.multi_crop = multi_crop
        self.multi_type = multi_type
        self.use_resolution = use_resolution
        self.num_classes = models[0].num_classes
        self.num_class = models[0].num_classes()
        self.names = [cfg.get("name", cfg["type"]) for cfg in cfgs]
        self.weights = np.array([cfg.get("weight", 1.0) for cfg in cfgs])
        self.weights = self.weights / np.sum(self.weights)
        assert self.ensemble_type in {"vote", "weight"}
        assert self.multi_type in {"vote", "mean"}
        print("ensemble type : {};  number of model: {} ".format(self.ensemble_type, len(self.models)))

    def backward(self, gradient, image): # for foolbox 1.7.0 compatability
        return np.sum(self.weights.reshape((len(self.models), 1, 1, 1)) * [model.backward(gradient, image) for model in self.models], axis=0)

    def batch_predictions(self, images):
        preds = []
        if self.use_resolution is not None:
            for res in self.use_resolution:
                if res == 1:
                    pred = self._batch_predictions(images)
                else:
                    ims = np.stack([imresize(imresize(image, res, interp="nearest"),
                                             (64,64,3), interp="nearest") for image in images])
                    pred = self._batch_predictions(ims)
                preds.append(pred)
            return np.mean(preds, axis=0)
        if self.multi_angle is not None:
            preds = [self._batch_predictions(images)]
            preds += [self._batch_predictions(scipy.ndimage.rotate(images, angle=ang, axes=(1, 2), reshape=False)) for ang in self.multi_angle]
            # flip lr
            preds += [self._batch_predictions(np.flip(images, 2))]
            if self.multi_type == "vote":
                preds = [np.argmax(pred, -1) for pred in preds]
                tmp = np.eye(self.num_class)
                preds = [tmp[p] for p in preds]
            return np.mean(preds, axis=0)
        if self.multi_crop is not None:
            preds = [self._batch_predictions(images)]
            def resize_and_crop_image(img, aug):
                img = np.reshape(img, [64, 64, 3])
                img = np.pad(img, ((aug[1], 16-aug[1]), (aug[2], 16-aug[2]), (0, 0)), 'constant')
                img = img[8:72, 8:72, :]
                img = np.reshape(img, [1, 64, 64, 3])
                return img
            preds += [self._batch_predictions(resize_and_crop_image(images, aug)) for aug in self.multi_crop]
            return np.mean(preds, axis=0)
        #else:
        return self._batch_predictions(images)

    def _batch_predictions(self, images):
        predictions = [m.batch_predictions(images) for m in self.models]
        if self.ensemble_type == "vote":
            predictions = [np.argmax(pred, -1) for pred in predictions]
            tmp = np.eye(self.num_class)
            predictions = [tmp[p] for p in predictions]
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
    fmodel = EnsembleModel(models, model_cfg["models"], ensemble_type=model_cfg.get("ensemble_type", "weight"),
                           use_resolution=model_cfg.get("use_resolution", None),
                           multi_angle=model_cfg.get("multi_angle", None),
                           multi_crop=model_cfg.get("multi_crop", None),
                           multi_type=model_cfg.get("multi_type", "mean"))
    return fmodel

if __name__ ==  "__main__":
    # executable for debuggin and testing
    print(create_fmodel())
