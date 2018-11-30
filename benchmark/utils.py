import os
import yaml
import logging
import contextlib

import numpy as np
from scipy.misc import imread

from fmodel import create_fmodel

__all__ = ["substitute_argscope", "ImageReader", "distance", "worst_case_distance", "create_fmodel_cfg"]

def create_fmodel_cfg(cfg_path):
    _old = os.environ.get("FMODEL_MODEL_CFG", None)
    os.environ["FMODEL_MODEL_CFG"] = cfg_path
    fmodel = create_fmodel()
    if _old is None:
        os.environ.pop("FMODEL_MODEL_CFG")
    else:
        os.environ["FMODEL_MODEL_CFG"] = _old
    return fmodel

def distance(X, Y):
    X = X.astype(np.float64) / 255
    Y = Y.astype(np.float64) / 255
    return np.linalg.norm(X - Y)

def worst_case_distance(X):
    worst_case = np.zeros_like(X)
    worst_case[X < 128] = 255
    return distance(X, worst_case)

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

dataset_dct = {
    "tiny-imagenet": {"shape": (64, 64, 3), "channel_axis": -1},
    "cifar10": {"shape": (32, 32, 3), "channel_axis": -1}
}
class ImageReader(object):
    available_methods = ["npy", "img", "bin"]
    def __init__(self, tp, dataset="tiny-imagenet"):
        assert tp in self.available_methods
        self.tp = tp
        self.dataset = dataset
        self.shape, self.channel_axis = dataset_dct[dataset]["shape"], dataset_dct[dataset]["channel_axis"]

        assert self.channel_axis in {-1, 0}
        if self.channel_axis == 0:
            self.sc_shape = self.shape[1:] # single color shape
        else:
            self.sc_shape = self.shape[:-1]
        self.tile_shape = [1 for _ in self.shape]
        self.tile_shape[self.channel_axis] = self.shape[self.channel_axis]
        self.tile_shape = tuple(self.tile_shape)

    def _read_image(self, key):
        input_folder = os.getenv('INPUT_IMG_PATH')
        img_path = os.path.join(input_folder, key)
        if self.tp == "npy":
            image = np.load(img_path)
        elif self.tp == "bin":
            image = np.fromfile(img_path, dtype=np.uint8).reshape(self.shape)
        else:
            image = imread(img_path)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        return image

    def read_images(self):
        filepath = os.getenv('INPUT_YML_PATH')
        with open(filepath, 'r') as ymlfile:
            data = yaml.load(ymlfile)
        for key in data.keys():
            im = self._read_image(key)
            if im.shape != self.shape:
                if im.shape != self.sc_shape:
                    logging.warning("shape of image read from file {} is not {} or {}. ignore.".format(key, self.shape, self.sc_shape))
                    continue
                im = np.tile(np.expand_dims(im, self.channel_axis), self.tile_shape)
                # continue
            yield (key, im, data[key])
