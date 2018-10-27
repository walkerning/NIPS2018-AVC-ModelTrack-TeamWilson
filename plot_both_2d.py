# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from fmodel import create_fmodel

def create_fmodel_(cfg_path):
    _old = os.environ.get("FMODEL_MODEL_CFG", None)
    os.environ["FMODEL_MODEL_CFG"] = cfg_path
    fmodel = create_fmodel()
    if _old is None:
        os.environ.pop("FMODEL_MODEL_CFG")
    else:
        os.environ["FMODEL_MODEL_CFG"] = _old
    return fmodel

parser = argparse.ArgumentParser()
parser.add_argument("test_name", help="result_dir, 应该反映有哪些模型在里面")
# parser.add_argument("img_key", help="image key")
parser.add_argument("-i", "--img-key", action="append", default=[], help="image keys")
parser.add_argument("--model", action="append", default=[])
parser.add_argument("--adv-path", action="append", default=[])
parser.add_argument("--use-grad", action="append", default=[])
parser.add_argument("--gpu", default="0")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_name = args.test_name
img_key = args.img_key

# load labels
yml_fpath = "/home/foxfi/yml_files/labels.yml"
image_dir = "/home/foxfi/test_images/"
with open(yml_fpath, 'r') as yml_f:
    label_dct = yaml.load(yml_f)

# load models
model_cfgs = args.model

model_names = [os.path.basename(cfg).split(".")[0] for cfg in model_cfgs]
_model_names_dct = {n:i for i, n in enumerate(model_names)}
models = [create_fmodel_(cfg) for cfg in model_cfgs]

_name_sub = [
    ("iterative_transfer", "it"),
    ("transfer", "t")
]
def _santitize(path):
    path = os.path.normpath(path)
    path = path.strip("/").strip("./").replace("/", "-")
    for k, v in _name_sub:
        path = path.replace(k, v)
    return path

adv_paths = args.adv_path
def find_orth(direct):
    rand_vector = np.random.rand(*img.shape).reshape(-1)
    direct_ort = rand_vector - np.sum(rand_vector * direct) * direct
    direct_ort = (direct_ort / np.linalg.norm(direct_ort))
    assert np.abs(np.sum(direct * direct_ort)) < 1e-6
    return direct_ort

for img_key in args.img_key:
    print("handle image: ", img_key)
    # prepare directions
    direct_names = []
    directs = []
    img = np.load(os.path.join(image_dir, img_key)).astype(np.float32)
    label = label_dct[img_key]
    for adv_path in adv_paths:
        adv_img = np.load(os.path.join(adv_path, img_key)).astype(np.float32)
        unn_dist = np.linalg.norm(adv_img - img)
        if np.abs(unn_dist) < 1e-6:
            continue
        direct = ((adv_img - img) / unn_dist).reshape(-1)
        direct_names.append("adv-" + _santitize(adv_path))
        directs.append((direct, find_orth(direct)))
    
    use_grad_names = [os.path.basename(cfg).split(".")[0] for cfg in args.use_grad]
    for use_grad in use_grad_names:
        adv_direct = models[_model_names_dct[use_grad]].predictions_and_gradient(img, label)[1]
        unn_dist = np.linalg.norm(adv_direct)
        direct = (adv_direct / unn_dist).reshape(-1)
        direct_names.append("grad-" + use_grad)
        directs.append((direct, find_orth(direct)))
    
    # make result dir
    res_dir = os.path.join("./boundaries/new_2dplots", img_key)
    os.makedirs(res_dir, exist_ok=True)
    
    # for now, all direction use the same scale to plot
    dist = np.array([8000, 8000]) # 29.35 * 255
    # dist = np.array([2000, 2000])
    step = 300
    dist = step * (dist // step)
    im_size = (2 * (dist // step) + 1).astype(np.int)
    print("bounds: ", "+-{}; +-{}".format(*dist), "image size: ", im_size)
    
    # start calculating
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        x = np.array(x)
        x = np.exp(x)
        x.astype('float32')
        if x.ndim == 1:
            sumcol = sum(x)
            for i in range(x.size):
                x[i] = x[i]/float(sumcol)
        if x.ndim > 1:
            sumcol = x.sum(axis = 0)
            for row in x:
                for i in range(row.size):
                    row[i] = row[i]/float(sumcol[i])
        return x
    
    def _to_hex(nums):
        hexes = "#"
        for num in nums:
            hex_ = hex(num)[2:]
            hexes += "0" * (2 - len(hex_)) + hex_
        return hexes
    
    print("Evaluate on directions: ", direct_names)
    
    for direct_n, (direct, direct_ort) in zip(direct_names, directs):
        boundary_maps = [np.zeros(im_size, dtype=np.uint8) for _ in range(len(models))]
        iter_loss = [np.zeros(im_size, dtype=np.float32) for _ in range(len(models))]
        print("Calculating for direction: ", direct_n)
        direct_ort = direct_ort.reshape(img.shape)
        direct = direct.reshape(img.shape)
    
        for i, step_x in enumerate(np.arange(-dist[0], dist[0]+step, step)):
            for j, step_y in enumerate(np.arange(-dist[1], dist[1]+step, step)):
                iter_img = img + step_x * direct + step_y * direct_ort
                for mi, model in enumerate(models):
                    pre_ = model.predictions(iter_img)
                    label_plus1 = np.argmax(pre_) + 1
                    iter_loss[mi][i, j] = np.log(softmax(pre_))[label]
                    boundary_maps[mi][i, j] = label_plus1
        s_res_dir = os.path.join(res_dir, direct_n, test_name)
        os.makedirs(s_res_dir, exist_ok=True)
        np.save(os.path.join(s_res_dir, "dist_{}_{}-step_{}.npz".format(dist[0], dist[1], step)), {"boundary_maps": boundary_maps, "directs": [direct, direct_ort]})
    
        available_classes = list(np.unique(boundary_maps))
        # assert label + 1 in available_classes
        if label + 1 not in available_classes:
            print("maybe the step is set too large... this picture will be skipped by default.")
            continue
        print("avail classes num: ", len(available_classes))
        available_classes[available_classes.index(label+1)] = available_classes[0]
        available_classes[0] = label+1
        colors = (np.random.rand(len(available_classes)+2, 3) * 255).astype(np.uint8)
        
        bm_pic_maps = []
        for bm in boundary_maps:
            bm_pic = np.zeros(list(bm.shape) + [3], dtype=np.uint8)
            for i, cls in enumerate(available_classes):
                bm_pic[bm == cls] = colors[i]
            bm_pic_maps.append(bm_pic)
        
        
        fig = plt.figure(figsize=(1.5*len(model_names), 2.5))
        for i, (mn, pic) in enumerate(zip(model_names, bm_pic_maps)):
            ax = fig.add_subplot(1, len(models), i+1)
            ax.imshow(pic)
            circle = plt.Circle([im_size[1]//2, im_size[0]//2], 1, color="black", fill=False)
            ax.add_artist(circle)
            ax.set_title(mn, fontsize=10)
        
        cbaxes = fig.add_axes([0.2, 0.01, 0.6, 0.03])  # left bottom width height
        cmap = mpl.colors.ListedColormap([_to_hex(c) for c in colors])
        
        bounds = range(len(colors))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap,
                                        norm=norm,
                                        boundaries=list(bounds) + [len(colors)],
                                        ticks=bounds,
                                        spacing='uniform',
                                        orientation='horizontal')
        plt.suptitle("img {} - direct {} - {} - step {}".format(img_key, direct_n, test_name, step), fontsize=12)
        plt.savefig(os.path.join(s_res_dir, "dist_{}_{}-step_{}.png".format(dist[0], dist[1], step)))
        
        X = np.arange(-dist[0], dist[0] + step, step)
        Y = np.arange(-dist[1], dist[1] + step, step)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(1.5*len(model_names), 2.5))
        for i in range(len(models)):
            ax = fig.add_subplot(1, len(models), i+1, projection='3d')
            ax.plot_surface(X, Y, iter_loss[i])
            ax.set_title(model_names[i], fontsize=10)
        plt.suptitle("img {} - direct {} - {} - step {} grad".format(img_key, direct_n, test_name, step), fontsize=12)
        plt.savefig(os.path.join(s_res_dir, "dist_{}_{}-step_{}-grad.png".format(dist[0], dist[1], step)))
        print("Save pictures and results to: ", s_res_dir)
