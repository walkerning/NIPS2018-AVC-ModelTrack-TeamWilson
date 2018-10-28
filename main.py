# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import yaml
import shutil
import argparse
import subprocess

from nics_at import utils
from nics_at import MutualTrainer, DistillTrainer
trainers = {
    "mutual": MutualTrainer,
    "distill": DistillTrainer
}

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Common arguments
parser.add_argument("--train-dir", type=str, default="",
                    help="Directory for storing snapshots")
parser.add_argument("--gpu", type=str, default="0",
                    help="GPU used for training/validation")
parser.add_argument("--config", type=str, default="./config.json",
                    help="Config files")
parser.add_argument("--test-only", action="store_true", default=False, help="Only run test")
parser.add_argument("--no-init-test", action="store_true", default=False, help="Do not run test before training")
parser.add_argument("--save-every", default=5, type=int)
parser.add_argument("--print-every", default=10, type=int, help="print every PRINT_EVERY step")

subparsers = parser.add_subparsers(dest="trainer_type")
for t_tp, t_cls in trainers.iteritems():
    # Trainer specific arguments
    sub_parser = subparsers.add_parser(t_tp)
    t_cls.populate_arguments(sub_parser)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
with open(args.config) as config_file:
    config = yaml.load(config_file)

if not args.test_only:
    if not os.path.exists(args.train_dir):
        subprocess.check_call("mkdir -p {}".format(args.train_dir),
                              shell=True)
    args.log_file = os.path.join(args.train_dir, "train.log")
    args.log_file = open(args.log_file, "w")
    # shutil.copyfile(sys.argv[0], os.path.join(args.train_dir, "train.py"))
    def _onlycopy_py(src, names):
        return [name for name in names if not name.endswith(".py")]
    if os.path.exists(os.path.join(args.train_dir, "nics_at")):
        shutil.rmtree(os.path.join(args.train_dir, "nics_at"))
    shutil.copytree("nics_at",  os.path.join(args.train_dir, "nics_at"))
    shutil.copyfile(args.config, os.path.join(args.train_dir, "config.yaml"))
else:
    args.log_file = None
utils.log = utils.get_log_func(args.log_file)
utils.log("CMD: ", " ".join(sys.argv))
if not args.train_dir:
    log("WARNING: model will not be saved if `--train_dir` option is not given.")
trainer = trainers[args.trainer_type](args, config)
trainer.init()
trainer.start()
trainer.sess.close()
