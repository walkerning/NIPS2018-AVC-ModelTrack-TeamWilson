# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import argparse
import numpy as np
from scipy.misc import imread
import matplotlib
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from utils import create_fmodel_cfg


parser = argparse.ArgumentParser()
parser.add_argument("test_name", help="result_dir, 应该反映有哪些模型在里面")
parser.add_argument("--cfg", default=None, help="plot cfg file")
# parser.add_argument("img_key", help="image key")
parser.add_argument("-i", "--img-key", action="append", default=[], help="image keys")
parser.add_argument("--all", action="store_true", default=False)
parser.add_argument("--model", action="append", default=[])
parser.add_argument("--adv-path", action="append", default=[])
parser.add_argument("--use-grad", action="append", default=[])
parser.add_argument("--gpu", default="0")

parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--val", action="store_true", default=False)
parser.add_argument("--label-file", default=None) # "/home/foxfi/yml_files/labels.yml")
parser.add_argument("--image-path", default=None) # "/home/foxfi/test_images/")
parser.add_argument("--image-type", default=None, choices=[None, "img", "npy"])
parser.add_argument("--other-type", default=None, choices=[None, "npy", "bin"])
parser.add_argument("--plot-loss", action="store_true", default=False)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

test_name = args.test_name

# load plot config
if args.cfg is not None:
    plot_cfg = yaml.load(open(args.cfg, "r"))
else:
    plot_cfg = {}

# load labels
if args.train:
    print("use train imgdir and labels")
    yml_fpath = "/home/foxfi/tiny-imagenet-200/train_labels.yml"
    image_dir = "/home/foxfi/tiny-imagenet-200/"
    image_type = "img"
    other_type = "bin"
elif args.val:
    print("use val imgdir and labels")
    yml_fpath = "/home/foxfi/tiny-imagenet-200/val/val_labels.yml"
    image_dir = "/home/foxfi/tiny-imagenet-200/val/images"
    image_type = "img"
    other_type = "bin"
else:
    yml_fpath = args.label_file or plot_cfg.get("label_file", None)
    image_dir = args.image_path or plot_cfg.get("image_path", None)
    image_type = args.image_type or plot_cfg.get("image_type", None)
    other_type = args.other_type or plot_cfg.get("other_type", None)

# priority-last default; first priority is the cmdline, second priority is the configuration
image_dir = image_dir or "/home/foxfi/test_images/" 
yml_fpath = yml_fpath or "/home/foxfi/yml_files/labels.yml"
image_type = image_type or "npy"
other_type = other_type or "npy"

print("label fpath: {}; image dir: {}; image type: {}; adv files type: {}".format(yml_fpath, image_dir, image_type, other_type))
with open(yml_fpath, 'r') as yml_f:
    label_dct = yaml.load(yml_f)

# load models
model_cfgs = args.model

model_names = [os.path.basename(cfg).split(".")[0] for cfg in model_cfgs]
_model_names_dct = {n:i for i, n in enumerate(model_names)}
models = [create_fmodel_cfg(cfg) for cfg in model_cfgs]

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

def load_example(path, img_key):
    fpath = os.path.join(path, img_key + "." + other_type)
    if other_type == "npy":
        return np.load(fpath)
    else: # bin
        return np.fromfile(fpath, dtype=np.uint8).reshape((64, 64, 3))

def parse_dist_cfg(size, step, im_size_max, max_ratio, coors, adjust_ratio=1.1):
    # handle size(in l2 dist)
    if size == "adjust":
        size = np.max(np.abs(coors), axis=0) * adjust_ratio
        if size[1] == 0:
            size[1] = size[0]
        if size[0] / size[1] > max_ratio:
            size[1] = size[0] / max_ratio
        elif size[1] / size[0] > max_ratio:
            size[0] = size[1] / max_ratio
    elif size == "adjust_max":
        size = np.max(np.abs(coors)) * 1.1
        size = [size, size]
    else:
        assert isinstance(size, (list, tuple))
    size = np.array(size).astype(np.int)
    # handle step
    if isinstance(step, int):
        step = [step, step]
    elif step == "auto":
        step = int(np.floor(2 * max(size) / im_size_max))
        step = [step, step]
    else:
        assert isinstance(step, (list, tuple))
    size = step * (size // step)
    im_size = (2 * (size // step) + 1).astype(np.int)
    return size, step, im_size

adv_paths = args.adv_path
def find_orth(direct, another=None):
    if another is None:
        rand_vector = np.random.rand(*img.shape).reshape(-1)
    else:
        rand_vector = another.reshape(-1)
    weight_dire = np.sum(rand_vector * direct)
    direct_ort = rand_vector - weight_dire * direct
    weight_ort = np.linalg.norm(direct_ort)
    direct_ort = direct_ort / weight_ort
    assert np.abs(np.sum(direct * direct_ort)) < 1e-6
    if another is None:
        return direct_ort
    else:
        return direct_ort, (weight_dire, weight_ort)

if args.all: # test all images listed in the label file
    args.img_key = list(label_dct.keys())

for ori_img_key in args.img_key:
    print("handle image: ", ori_img_key)
    # prepare directions
    directn_point_map = {}
    directn_cfg_map = {}
    direct_names = []
    directs = []
    if image_type == "npy":
        img = np.load(os.path.join(image_dir, ori_img_key)).astype(np.float32)
    else: # image_type == "img"
        img = imread(os.path.join(image_dir, ori_img_key)).astype(np.float32)
    label = label_dct[ori_img_key]
    img_key = os.path.basename(ori_img_key).split(".")[0]
    for plot_scfg in plot_cfg["directs"]:
        vertical_img = load_example(plot_scfg["vertical"], img_key).astype(np.float32)
        unn_dist = np.linalg.norm(vertical_img - img)
        if np.abs(unn_dist) < 1e-6:
            continue
        direct = ((vertical_img - img) / unn_dist).reshape(-1)
        another_f = plot_scfg.get("another", None)
        direct_n = plot_scfg["direct_name"]
        directn_point_map[direct_n] = [(plot_scfg["vertical_name"], (unn_dist, 0))]
        if another_f is None:
            direct_orth = find_orth(direct)
        else:
            another_img = load_example(another_f, img_key).astype(np.float32)
            direct_orth, another_coor = find_orth(direct, another_img - img)
            directn_point_map[direct_n].append((plot_scfg["another_name"], another_coor))
        directn_cfg_map[direct_n] = plot_scfg
        direct_names.append(direct_n)
        directs.append((direct, direct_orth))

    for adv_path in adv_paths:
        adv_img = load_example(adv_path, img_key).astype(np.float32)
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
    if not direct_names:
        continue

    # make result dir
    res_dir = os.path.join("./boundaries/new_2dplots", img_key)
    os.makedirs(res_dir, exist_ok=True)
    
    for direct_n, (direct, direct_ort) in zip(direct_names, directs):
        if args.cfg:
            scfg = directn_cfg_map[direct_n]
            dist, step, im_size = parse_dist_cfg(size=scfg.get("size", "adjust"), step=scfg.get("step", "auto"),
                                                 im_size_max=scfg.get("im_size_max", [50,50]),
                                                 max_ratio=scfg.get("max_ratio", 2), coors=[d[1] for d in directn_point_map[direct_n]],
                                                 adjust_ratio=scfg.get("adjust_ratio", 1.1))
            step0, step1 = step
            print("bounds: ", "+-{}; +-{}".format(*dist), "image size: ", im_size, "step: {}, {}".format(step0, step1))
        else:    
            # for now, all direction use the same scale to plot
            dist = np.array([8000, 8000]) # 29.35 * 255
            # dist = np.array([2000, 2000])
            step0 = step1 = 300
            dist = step * (dist // step)
            im_size = (2 * (dist // step) + 1).astype(np.int)
            print("bounds: ", "+-{}; +-{}".format(*dist), "image size: ", im_size, "step: {}, {}".format(step0, step1))

        boundary_maps = [np.zeros(im_size, dtype=np.uint8) for _ in range(len(models))]
        iter_loss = [np.zeros(im_size, dtype=np.float32) for _ in range(len(models))]
        print("Calculating for direction: ", direct_n)
        direct_ort = direct_ort.reshape(img.shape)
        direct = direct.reshape(img.shape)
    
        for i, step_x in enumerate(np.arange(-dist[0], dist[0]+step0, step0)):
            for j, step_y in enumerate(np.arange(-dist[1], dist[1]+step1, step1)):
                iter_img = img + step_x * direct + step_y * direct_ort
                for mi, model in enumerate(models):
                    pre_ = model.predictions(iter_img)
                    label_plus1 = np.argmax(pre_) + 1
                    iter_loss[mi][i, j] = np.log(softmax(pre_))[label]
                    boundary_maps[mi][i, j] = label_plus1
        s_res_dir = os.path.join(res_dir, direct_n, test_name)
        os.makedirs(s_res_dir, exist_ok=True)
        np.save(os.path.join(s_res_dir, "dist_{}_{}-step_{}_{}.npz".format(dist[0], dist[1], step0, step1)), {"boundary_maps": boundary_maps, "directs": [direct, direct_ort]})
    
        available_classes = list(np.unique(boundary_maps))
        # assert label + 1 in available_classes
        if label + 1 not in available_classes:
            # print("maybe the step is set too large... this picture will be skipped by default.")
            # continue
            pass
        print("avail classes num: ", len(available_classes))
        available_classes[available_classes.index(label+1)] = available_classes[0]
        available_classes[0] = label+1
        colors = (np.random.rand(len(available_classes)+2, 3) * 255).astype(np.uint8)
        other_point_colors = ["r", "g"]

        bm_pic_maps = []
        for bm in boundary_maps:
            bm_pic = np.zeros(list(bm.shape) + [3], dtype=np.uint8)
            for i, cls in enumerate(available_classes):
                bm_pic[bm == cls] = colors[i]
            bm_pic_maps.append(bm_pic)

        
        fig = plt.figure(figsize=(1.5*len(model_names), 3))
        for i, (mn, pic) in enumerate(zip(model_names, bm_pic_maps)):
            ax = fig.add_subplot(1, len(models), i+1)
            ax.imshow(pic)
            circle = plt.Circle([im_size[1]//2, im_size[0]//2], 1, color="black", fill=False)
            ax.add_artist(circle)
            _ns = []
            _cs = []
            for (n, coor), c in zip(directn_point_map[direct_n], other_point_colors):
                if i == 0:
                    print("coordinate: {}; distance: {}".format(coor, np.linalg.norm(coor)/255))
                circle = plt.Circle([im_size[1]//2+coor[1]//step1, im_size[0]//2+coor[0]//step0], 1, color=c, fill=False)
                # text = plt.Text(im_size[1]//2+coor[1]//step1, im_size[0]//2+coor[0]//step0+0.5, text=n, color="red", fontsize="small", size="xx-small")
                # ax.add_artist(text)
                ax.add_artist(circle)
                _ns.append(n)
                _cs.append(circle)
            if i == 0:
                ax.legend(_cs, _ns)
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
        # plt.suptitle("img {} - direct {} - {} - step {} {}".format(img_key, direct_n, test_name, step0, step1), fontsize=12)
        plt.savefig(os.path.join(s_res_dir, "dist_{}_{}-step_{}_{}.png".format(dist[0], dist[1], step0, step1)))

        if args.plot_loss:
            X = np.arange(-dist[0], dist[0] + step0, step0)
            Y = np.arange(-dist[1], dist[1] + step1, step1)
            X, Y = np.meshgrid(X, Y)
            fig = plt.figure(figsize=(1.5*len(model_names), 2.5))
            for i in range(len(models)):
                ax = fig.add_subplot(1, len(models), i+1, projection='3d')
                ax.plot_surface(X, Y, iter_loss[i])
                ax.set_title(model_names[i], fontsize=10)
            # plt.suptitle("img {} - direct {} - {} - step {} {} grad".format(img_key, direct_n, test_name, step0, step1), fontsize=12)
            plt.savefig(os.path.join(s_res_dir, "dist_{}_{}-step_{}_{}-grad.png".format(dist[0], dist[1], step0, step1)))
        print("Save pictures and results to: ", s_res_dir)
