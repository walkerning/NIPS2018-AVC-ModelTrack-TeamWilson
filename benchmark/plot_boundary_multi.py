# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import argparse
import numpy as np

from utils import create_fmodel_cfg

parser = argparse.ArgumentParser()
parser.add_argument("label_path")
parser.add_argument("--adv-path", action="append", default=[])
parser.add_argument("--model", action="append", default=[])
parser.add_argument("--use-grad", action="append", default=[])
parser.add_argument("--save", required=True)
parser.add_argument("--save-log", required=True)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
log_file = open(args.save_log, "a")

models = []
names = []
use_grad = []
args.model = [os.path.abspath(m) for m in args.model]
args.use_grad = [os.path.abspath(m) for m in args.use_grad]
assert len(set(args.use_grad).difference(args.model)) == 0

for cfg in args.model:
    models.append(create_fmodel_cfg(cfg))
    name = os.path.basename(cfg).split(".")[0]
    names.append(name)
    if cfg in args.use_grad:
        use_grad.append(name)
use_grad = set(use_grad)
yml_fpath = args.label_path
image_dir = "/home/foxfi/test_images/"
with open(yml_fpath, 'r') as yml_f:
    label_dct = yaml.load(yml_f)

all_grad_norms = [{} for _ in range(len(models))]
all_adv_dists = {adv_path: {} for adv_path in args.adv_path}
all_dtb_dct = {}
total_pic_num = len(label_dct)

direction_names = ["grad:{}".format(name) for name in names if name in use_grad]
direction_names += args.adv_path
print("models_names: ", names, file=log_file)
print("direction_names: ", direction_names, file=log_file)
log_file.flush()

for imi, (img_key, label) in enumerate(label_dct.items()):
    img, label = np.load(os.path.join(image_dir, img_key)).astype(np.float32), label_dct[img_key]
    directions = []
    direction_names = []
    not_correct = []
    for i, (model, name) in enumerate(zip(models, names)):
        if np.argmax(model.predictions(img)) != label:
            not_correct.append(name)
            continue
        grad = model.predictions_and_gradient(img, label)[1].reshape(-1)
        grad_norm = np.linalg.norm(grad)
        all_grad_norms[i][img_key] = grad_norm
        if name in use_grad:
            directions.append(grad / grad_norm)
        # norma_grad = grad / grad_norm
    if not_correct:
        print(img_key, "models that are not correct on this image: ", not_correct, file=log_file)
        continue
    for i, imdir in enumerate(args.adv_path):
        adv_img = np.load(os.path.join(imdir, img_key)).astype(np.float32)
        unn_dist = np.linalg.norm(adv_img - img)
        n_dist = np.linalg.norm((adv_img - img)/255)
        all_adv_dists[imdir] = n_dist
        directions.append((adv_img - img) / unn_dist)

    dtb_lst = []
    eps_boundary = 500
    eps_multiply = 10
    for di, direct in enumerate(directions):
        print("Searching direction {}".format(di))
        eps = 0.1
        d_img = direct.reshape(img.shape)
        iter_img = img
        dist_to_boundaries = [None for _ in range(len(models))]
        num = 0
        left_num = len(models)
        last_find_num = 0
        dist_base = 0
        t_num = 0
        while 1:
            t_num += 1
            num += 1
            last_find_num += 1
            last_iter_img = iter_img
            iter_img = np.clip(iter_img + eps * d_img, 0, 255)
            if np.all(iter_img == last_iter_img):
                break
            print("\rStep {}: Searching {}".format(t_num, dist_base+num*eps), end="")
            for i, (model, model_name) in enumerate(zip(models, names)):
                if dist_to_boundaries[i] is None and np.argmax(model.predictions(iter_img)) != label: # non-targeted boundary
                    # TODO: 如果eps太大是不是可以往回binary search, 精确一点..
                    dist_to_boundaries[i] = dist_base + num * eps
                    left_num -= 1
                    last_find_num = 0
                    # print("Find boundary for model {}: {}".format(model_name, dist_base+num*eps))
            if last_find_num > eps_boundary:
                dist_base = dist_base + num * eps
                eps *= eps_multiply
                num = 0
                last_find_num = 0
                # print("Step {}: using eps: {}".format(t_num, eps))
            if left_num == 0:
                break
        dtb_lst.append(dist_to_boundaries)
    all_dtb_dct[img_key] = dtb_lst
    print(img_key, dtb_lst, file=log_file)
    log_file.flush()
    print("\r{}/{}: Finish searching for {}".format(imi+1, total_pic_num, img_key))
np.savez(open(args.save, "wb"), **{
    #"adv_path": args.adv_path,
    "model_names": names,
    "dtb_dct": all_dtb_dct,
    "direction_names": direction_names,
    "grad_norms": all_grad_norms,
    "all_adv_dists": all_adv_dists
})

