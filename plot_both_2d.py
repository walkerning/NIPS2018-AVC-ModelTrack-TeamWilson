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

'''
[[0.1, 4121.2, 7191.2], [1649.1, 99.1, 458.1], [2971.2, 1971.2, 0.1], [831.1, 110.1, 831.1], [89.1, 2257.1, 556.1]]
'''

#test_name = "resnet_grad"
test_name = "inception_grad"
yml_fpath = "/home/foxfi/yml_files/labels.yml"
image_dir = "/home/foxfi/test_images/"
with open(yml_fpath, 'r') as yml_f:
    label_dct = yaml.load(yml_f)
img_key = "26643040003_eb983fb434_o.npy"
#img_key = "26768991671_e3585315e8_o.npy"
img = np.load(os.path.join(image_dir, img_key)).astype(np.float32)
label = label_dct[img_key]

# model_cfgs = ["cfgs/resnet18.yaml", "cfgs/resnet_advtrained.yaml"]
# model_cfgs = ["cfgs/resnet18.yaml", "cfgs/resnet_advtrained.yaml"]
model_cfgs = [ "cfgs/resnet18.yaml", "cfgs/denoise_resnet18.yaml", "cfgs/resnet_advtrained.yaml"]
model_names = [os.path.basename(cfg).split(".")[0] for cfg in model_cfgs]
models = [create_fmodel_(cfg) for cfg in model_cfgs]

adv_path = "results/resnet/genbyinception/transfer"
adv_img = np.load(os.path.join(adv_path, img_key)).astype(np.float32)
unn_dist = np.linalg.norm(adv_img - img)
direct = ((adv_img - img) / unn_dist).reshape(-1)
# adv_direct = models[0].predictions_and_gradient(img, label)[1]
# unn_dist = np.linalg.norm(adv_direct)
# direct = (adv_direct / unn_dist).reshape(-1)

# TODO:
rand_vector = np.random.rand(*img.shape).reshape(-1)
direct_ort = rand_vector - np.sum(rand_vector * direct) * direct
direct_ort = (direct_ort / np.linalg.norm(direct_ort)).reshape(img.shape)
direct = direct.reshape(img.shape)
#directions = [direct.reshape(img.shape), direct_ort.reshape(img.shape)]
assert np.abs(np.sum(direct * direct_ort)) < 1e-6

res_dir = os.path.join("./boundaries/2dplots", test_name)
os.makedirs(res_dir, exist_ok=True)

# dist = np.array([1600, 1000])
dist = np.array([3000, 3000])
#dist = np.array([2, 1])
#step = 0.05
step = 100
#step = 100

im_size = (2 * (dist / step) + 1).astype(np.int)
print("bounds: ", "+-{}; +-{}".format(*dist), "image size: ", im_size)
boundary_maps = [np.zeros(im_size, dtype=np.uint8) for _ in range(len(models))]
iter_loss = [np.zeros(im_size, dtype=np.float32) for _ in range(len(models))]
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
for i, step_x in enumerate(np.arange(-dist[0], dist[0]+step, step)):
    for j, step_y in enumerate(np.arange(-dist[1], dist[1]+step, step)):
        iter_img = img + step_x * direct + step_y * direct_ort
        for mi, model in enumerate(models):
            pre_ = model.predictions(iter_img)
            label_plus1 = np.argmax(pre_) + 1
            iter_loss[mi][i, j] = np.log(softmax(pre_))[label]
            boundary_maps[mi][i, j] = label_plus1

np.save(os.path.join(res_dir, img_key + "-step_{}.npz".format(step)), {"boundary_maps": boundary_maps, "directs": [direct, direct_ort]})

def _to_hex(nums):
    hexes = "#"
    for num in nums:
        hex_ = hex(num)[2:]
        hexes += "0" * (2 - len(hex_)) + hex_
    return hexes

available_classes = list(np.unique(boundary_maps))
assert label + 1 in available_classes
available_classes[available_classes.index(label+1)] = available_classes[0]
available_classes[0] = label+1
colors = (np.random.rand(len(available_classes), 3) * 255).astype(np.uint8)

bm_pic_maps = []
for bm in boundary_maps:
    bm_pic = np.zeros(list(bm.shape) + [3], dtype=np.uint8)
    for i, cls in enumerate(available_classes):
        bm_pic[bm == cls] = colors[i]
    bm_pic_maps.append(bm_pic)


fig = plt.figure()
for i, (mn, pic) in enumerate(zip(model_names, bm_pic_maps)):
    ax = fig.add_subplot(1, len(models), i+1)
    ax.imshow(pic)
    circle = plt.Circle([im_size[1]//2, im_size[0]//2], 1, color="black", fill=False)
    ax.add_artist(circle)
    ax.set_title(mn)

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
plt.suptitle(test_name + " " + img_key + " step {}".format(step))
plt.savefig(os.path.join(res_dir, img_key + "-step_{}.png".format(step)))

X = np.arange(-dist[0], dist[0] + step, step)
Y = np.arange(-dist[1], dist[1] + step, step)
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
for i in range(len(models)):
    ax = fig.add_subplot(1, len(models), i+1, projection='3d')
    import pdb
    pdb.set_trace()
    ax.plot_surface(X, Y, iter_loss[i])
    ax.set_title(model_names[i])
plt.suptitle(test_name + " " + img_key + " step {} grad".format(step))
plt.savefig(os.path.join(res_dir, img_key + "-step_{}_grad.png".format(step)))
