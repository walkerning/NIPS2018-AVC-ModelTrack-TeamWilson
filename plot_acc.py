# -*- coding: utf-8 -*-
from __future__ import print_function

import re
import os
import math
import argparse

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--save", "-s", required=True, help="save to path")
parser.add_argument("--iter-train", action="store_true", default=False)
parser.add_argument("--mutual-num", default=None, type=int)
parser.add_argument("--plot-train-acc", default=False, action="store_true")
args, fnames = parser.parse_known_args()

exists_test = ["normal"]
if args.plot_train_acc:
    exists_test.append("train")
flabels = []
adv_acc_dct_lst = []
for fname in fnames:
    if fname.endswith("train.log"):
        dirname = os.path.dirname(os.path.abspath(fname))
    else:
        assert os.path.isdir(fname)
        dirname = os.path.abspath(fname)
    fname = os.path.join(dirname, "train.log")
    label = os.path.basename(dirname)
    print("handling {}".format(label))
    content = open(fname, "r").read().strip().split("Epoch", 1)[1]
    # normal_accs = re.findall("Test normal_adv:[\n\t ]+?loss: [.0-9e]+; accuracy: ([.0-9e]+) %; Mean pixel distance:", content)
    # normal_accs = re.findall("loss: [.0-9e]+; accuracy: ([.0-9e]+) %; Mean pixel distance:", content)
    if args.plot_train_acc:
        # FIXME: mutual train acc is not supported now
        train_accs = re.findall("student accuracy: ([.0-9e]+) %;", content)
    normal_accs = re.findall("loss: [.0-9e]+; accuracy: ([.0-9e]+) %; ", content)
    adv_accs = re.findall("test (.+): acc: ([.0-9e]+);", content)
    adv_acc_dct = {}
    for k, v in adv_accs:
        adv_acc_dct.setdefault(k, []).append(float(v))
        if k not in exists_test:
            exists_test.append(k)
    if args.mutual_num:
        # mean/var
        normal_accs = np.array([float(v) for v in normal_accs]).reshape([-1, args.mutual_num])
        # mean_normal_acc = np.mean(normal_accs, axis=-1)
        # width_normal_acc = 4 * np.std(normal_accs, axis=-1)
        # adv_acc_dct["normal"] = zip(mean_normal_acc, width_normal_acc)
        adv_acc_dct["normal"] = normal_accs
    else:
        adv_acc_dct["normal"] = [float(v) for v in normal_accs]
        if args.plot_train_acc:
            adv_acc_dct["train"] = [float(v) for v in train_accs]
    flabels.append(label)
    adv_acc_dct_lst.append(adv_acc_dct)

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
num_plot = len(exists_test)
fig = plt.figure(figsize=(2*num_plot, 6))

if args.iter_train:
    true_accs = [{k: [sv for i, sv in enumerate(v) if i % 2 == 0] for k, v in acc_dct.iteritems()} for acc_dct in adv_acc_dct_lst]
    false_accs = [{k: [sv for i, sv in enumerate(v) if i % 2 == 1] for k, v in acc_dct.iteritems()} for acc_dct in adv_acc_dct_lst]
    true_accs = zip(*[[acc_dct.get(test_name, []) for test_name in exists_test] for acc_dct in true_accs])
    false_accs = zip(*[[acc_dct.get(test_name, []) for test_name in exists_test] for acc_dct in false_accs])
else:
    accs = zip(*[[acc_dct.get(test_name, []) for test_name in exists_test] for acc_dct in adv_acc_dct_lst])

if args.iter_train:
    for i, (name, t_acc, f_acc) in enumerate(zip(exists_test, true_accs, false_accs)):
        ax = fig.add_subplot(2, math.ceil(num_plot/2.), i+1)
        for j, (l, st_acc, sf_acc) in enumerate(zip(flabels, t_acc, f_acc)):
            if st_acc:
                ax.plot(st_acc, colors[j%len(colors)], label=l)
            if sf_acc:
                ax.plot(sf_acc, colors[j%len(colors)]+"--", label=l)
        ax.legend(loc=4, prop={"size": 4})
        ax.set_title(name)
else:
    for i, (name, acc) in enumerate(zip(exists_test, accs)):
        ax = fig.add_subplot(2, math.ceil(num_plot/2.), i+1)
        for j, (l, s_acc) in enumerate(zip(flabels, acc)):
            if args.mutual_num and name == "normal": # only mutual trainer logs multiple normal accuracies per epoch
                plt.fill_between(x=range(s_acc.shape[0]), y1=np.min(s_acc, axis=-1), y2=np.max(s_acc, axis=-1), color=colors[j%len(colors)], alpha="0.2")
                ax.plot(np.mean(s_acc, axis=-1), colors[j%len(colors)], label=l)
            elif s_acc:
                ax.plot(s_acc, colors[j%len(colors)], label=l)

        ax.legend(loc=4, prop={"size": 4})
        ax.set_title(name)

plt.savefig(args.save)
