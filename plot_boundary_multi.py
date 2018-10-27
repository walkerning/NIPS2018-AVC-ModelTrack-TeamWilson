# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import argparse
import numpy as np

from fmodel import create_fmodel
# adv training带来的变化
# TODO(1): ~14:00 **观察梯度的变化**. 统计一下在test500 inception, vgg11, resnet, resnet_adv的gradient方向是不是接近正交? adv training之后, 提督空间会比较扭曲吗? (画一下梯度场?)
# TODO(2): ~14:30和之后 找一下代码, 没有代码自己写一下 **基于正交adversarial 方向集合的减小做分析** 用GAAS找一下正交方向集. 看看这个正交adversarial directions里经过adv training有哪些没有了, 其中的一部分方向的boundary是不是通过adv training推的很远? 其中这些方向上可以transfer的方向的boundary被推进的比例大不大)
# TODO(3): **在gradient方向做boundary搜索, 看adv trained导致的各个关键方向的boundary距离变化**, 已经试了一下, 发现adv training还是有点用的...即使是在inception的grad方向;
#    - [x] ~13:00 用inception做resnet baseline的transfer攻击的adv方向是不是没有推boundary, 这个adv方向和inception自己的grad以及resnet baseline／resnet adv差距有多大?inception做renset adv trained的transfer攻击的adv方向比起做resnet trained的adv方向是不是没啥改变?  改变比想象中大, 还是推进了一些的，虽然没有推进的resnet gradient方向那么多，主要是因为adv训练之后normal acc降低了，所以看起来总体分数没变，其实在保持正确的样本上确实推动了...  
#    - [ ] ~13:20 找一个其他类别的正常样本看看benigned决策面有没有发生比较大的变化...
#    - [x] 试一下对多个数据点跑一下看看
#    - [ ] Tramer(1704.003453) GAAS说对抗训练没有displace boundary significantly... 因为其对模型boundary的推进太小了, 平均来说比一般adv 扰动的norm要小很多? (这个指的应该是transferable的扰动吧? 白盒扰动感觉可太小了...也不transferable). 也来统计看看transfer攻击生成的samples的距离...以及在transfer生成的adv samples方向上的决策面情况
# TODO(4): **白盒扰动是不是在hole里?** 画一下白盒adv扰动direction + 一个random正交方向的的点图, 看看决策面情况?


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

def create_fmodel_(cfg_path):
    _old = os.environ.get("FMODEL_MODEL_CFG", None)
    os.environ["FMODEL_MODEL_CFG"] = cfg_path
    fmodel = create_fmodel()
    if _old is None:
        os.environ.pop("FMODEL_MODEL_CFG")
    else:
        os.environ["FMODEL_MODEL_CFG"] = _old
    return fmodel

models = []
names = []
use_grad = []
args.model = [os.path.abspath(m) for m in args.model]
args.use_grad = [os.path.abspath(m) for m in args.use_grad]
assert len(set(args.use_grad).difference(args.model)) == 0

for cfg in args.model:
    models.append(create_fmodel_(cfg))
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


"""
在resnet baseline和resnetadv错误的图片上, 讲道理直接
是不是因为resnet adv 在normal examples的正确率变低了, 所以导致median/mean看起来差不多... 感觉最后还是没有错分的正确样本上的攻击好像距离确实变好了.

在inception, resnet, resnetadv都正确的222张图片
inception iterative transfer attack on resnet baseline:
median pixel distance: 4.169938195661555, mean pixel distance: 5.243404504250328

inception iterative transfer attack on resnet adv:
median pixel distance: 7.245576947226279, mean pixel distance: 7.762126723783608
"""
