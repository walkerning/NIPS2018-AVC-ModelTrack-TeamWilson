# -*- coding: utf-8 -*-
"""
This is an example of training cifar10.

Hyper-parameters for training VGG11 follows https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py

**NOTE**: To run this script, you should install keras, as the data handling utils is from keras.
"""

from __future__ import division
from __future__ import print_function

import os
import re
import cv2
import pdb
import sys
import copy
import json
import yaml
import time
import glob
import random
import shutil
import argparse
import subprocess
import multiprocessing
import numpy as np
from datetime import datetime

from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod, SaliencyMapMethod, BasicIterativeMethod, CarliniWagnerL2, MadryEtAl, MomentumIterativeMethod
from l2_attack import MadryEtAl_L2

import tensorflow as tf

class settings:
    default_cfg = {
	"test_frequency": 1,
	"epochs": 180,
	"batch_size": 100,
	"start_lr": 0.01,
	"lr_decay": 0.1,
	"weight_decay": 1e-4,
	
	"attack": "pgd",
	"eps": 4.0,
	"eps_iter": 1.0,
	"iter_train": 1,
	"combine_train": 0,
	"change_eps": 0,
	"eps_step": 2.0,
	"eps_iter_step": 0.5,

	"alpha": 0.1,
	"beta": 0,
	"theta": 0.5,
	"temperature": 1,
	"at_mode": "attention",

	"median_blur": 0,
	"bits": 8,

	"debug": 0,

        # sample eps
        "sample_eps_method": None,
        "sample_eps": None,
        "sample_eps_iter": None,

        # augmentaion
        "aug_saltpepper": None,
        "aug_gaussian": None,

        # test
        "test_saltpepper": None,
        "aug_mode": "pre",
        "test_eps": None,
        "test_eps_iter": None,
        "test_eps_pair": None,

        "denoiser_cfg": None,

        # net type
        "net_type": "resnet18",

        # namescope
        "tea_namescope": "",
        "stu_namescope": "stu_",

        "more_blocks": False,
        "attack_with_y": True,
        "boundaries": []
    }

    def __init__(self, config, args):
        cfg = copy.deepcopy(self.default_cfg)
        cfg.update(config)
        config = cfg
        print("Configurattion:\n" + "\n".join(["{:10}: {:10}".format(n, v) for n, v in config.items()]) + "\n")
        self.log_file = os.path.join(args.train_dir, "train.log")
        self.args = args
        self.dct = config

    def __getattr__(self, name):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        elif name in self.dct:
            return self.dct[name]
        else:
            raise KeyError("no attribute named {}".format(name))

FLAGS = None
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2

def load_filenames_labels(mode):
    """Gets filenames and labels

    Args:
        mode: 'train' or 'val'
            (Directory structure and file naming different for
            train and val datasets)

    Returns:
        list of tuples: (jpeg filename with path, label)
    """
    label_dict, class_description = build_label_dicts()
    filenames_labels = []
    if mode == 'train':
        filenames = glob.glob('./tiny-imagenet-200/train/*/images/*.JPEG')
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = str(label_dict[match.group()])
            filenames_labels.append((filename, label))
    elif mode == 'val':
        with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = './tiny-imagenet-200/val/images/' + split_line[0]
                label = str(label_dict[split_line[1]])
                filenames_labels.append((filename, label))

    return filenames_labels


def build_label_dicts():
    """Build look-up dictionaries for class label, and class description

    Class labels are 0 to 199 in the same order as 
        tiny-imagenet-200/wnids.txt. Class text descriptions are from 
        tiny-imagenet-200/words.txt

    Returns:
        tuple of dicts
            label_dict: 
                keys = synset (e.g. "n01944390")
                values = class integer {0 .. 199}
            class_desc:
                keys = class integer {0 .. 199}
                values = text description from words.txt
    """
    label_dict, class_description = {}, {}
    with open('./tiny-imagenet-200/wnids.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1]  # remove \n
            label_dict[synset] = i
    with open('./tiny-imagenet-200/words.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset, desc = line.split('\t')
            desc = desc[:-1]  # remove \n
            if synset in label_dict:
                class_description[label_dict[synset]] = desc

    return label_dict, class_description


def read_image(filename_q, mode, non_sp=None):
    """Load next jpeg file from filename / label queue
    Randomly applies distortions if mode == 'train' (including a 
    random crop to [56, 56, 3]). Standardizes all images.

    Args:
        filename_q: Queue with 2 columns: filename string and label string.
         filename string is relative path to jpeg file. label string is text-
         formatted integer between '0' and '199'
        mode: 'train' or 'val'

    Returns:
        [img, label]: 
            img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
            label = tf.unit8 target class label: {0 .. 199}
    """
    item = filename_q.dequeue()
    filename = item[0]
    label = item[1]
    file = tf.read_file(filename)
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.cast(img, tf.float32)
    if mode == "train":
        if FLAGS.aug_mode == "pre":
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.25)
            img = tf.image.random_hue(img, 0.1)
            img = tf.image.random_saturation(img, 0.8, 1.25)
            # img = tf.random_crop(img, [64, 64, 3])

    # image distortions: left/right, random hue, random color saturation
    if mode == 'train' and FLAGS.aug_mode == "post":
        # img = tf.image.random_flip_left_right(img)
        # # val accuracy improved without random hue
        # # img = tf.image.random_hue(img, 0.05)
        # img = tf.image.random_saturation(img, 0.5, 2.0)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.8, 1.25)
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_saturation(img, 0.8, 1.25)

    img = tf.pad(img, tf.constant([[4, 4], [4, 4], [0, 0]]))
    img = tf.random_crop(img, [64, 64, 3])
    label = tf.string_to_number(label, tf.int32)
    label = tf.cast(label, tf.uint8)

    return [img, label]


def batch_q(mode, non_sp=None):
    """Return batch of images using filename Queue

    Args:
        mode: 'train' or 'val'
        config: training configuration object

    Returns:
        imgs: tf.uint8 tensor [batch_size, height, width, channels]
        labels: tf.uint8 tensor [batch_size,]

    """
    filenames_labels = load_filenames_labels(mode)
    random.shuffle(filenames_labels)
    filename_q = tf.train.input_producer(filenames_labels,
                                num_epochs=FLAGS.epochs*10,
                                shuffle=True)

    # 2 read_image threads to keep batch_join queue full:
    return tf.train.batch_join([read_image(filename_q, mode, non_sp) for i in range(2)],
                            FLAGS.batch_size, shapes=[(64, 64, 3), ()],
                            capacity=2048)


################################################################################
# Convenience functions for building the ResNet model.
#######################################################
def get_at_loss(at1, at2):
    assert(len(at1) == len(at2))
    loss = 0
    if FLAGS.at_mode == "attention":
        def at(at1):
            return tf.nn.l2_normalize(tf.reduce_mean(tf.square(at1), axis=1), dim=(1,2))
        for ind_ in range(len(at1)):
            at1_ = at1[ind_]
            at2_ = at2[ind_]
            loss += tf.reduce_mean(tf.square(at(at2_) - at(at1_)), axis=(0,1,2))
        return loss
    elif FLAGS.at_mode == "path":
        def at(at1):
            return tf.nn.l2_normalize(tf.reduce_mean(tf.square(at1), axis=(2,3)), dim=(1))
        for ind_ in range(len(at1)):
            at1_ = at1[ind_]
            at2_ = at2[ind_]
            loss += tf.reduce_mean(tf.square(at(at2_) - at(at1_)), axis=(0,1))
        return loss
    else:
        for ind_ in range(len(at1)):
            at1_ = at1[ind_]
            at2_ = at2[ind_]
            loss += tf.reduce_mean(tf.square(at2_ - at1_), axis=(0,1,2,3))
        return loss


def main(_):

        adversarial_relu_list = []
        clean_relu_list = []
        if FLAGS.output_path:
            if not os.path.exists(FLAGS.output_path):
                subprocess.check_call("mkdir -p {}".format(FLAGS.output_path),
                                      shell=True)
            f_adv_correct = open(os.path.join(FLAGS.output_path, "adv_correct.txt"), "w")
            f_adv_wrong = open(os.path.join(FLAGS.output_path, "adv_wrong.txt"), "w")
            f_normal_correct = open(os.path.join(FLAGS.output_path, "normal_correct.txt"), "w")
            f_normal_wrong = open(os.path.join(FLAGS.output_path, "normal_wrong.txt"), "w")

        with tf.Session(config=config) as sess:
            if FLAGS.attack == "fgsm":
                attack_method = FastGradientMethod(model_stu, sess=sess)
                attack_params = {'eps': FLAGS.eps}
            elif FLAGS.attack == "bim":
                attack_method = BasicIterativeMethod(model_stu, sess=sess)
                attack_params = {'eps': FLAGS.eps,
                           'eps_iter': FLAGS.eps_iter}
            elif FLAGS.attack == "jsma":
                attack_method = SaliencyMapMethod(model_stu, sess=sess)
                attack_params = {'theta': FLAGS.eps, 'gamma': FLAGS.eps_iter}
            elif FLAGS.attack == "cw":
                attack_params = {'binary_search_steps': 1,
                     #'y': None,
                     'max_iterations': 100,
                     'learning_rate': 0.1,
                     'batch_size': 1,
                     'initial_const': 1}
                attack_method = CarliniWagnerL2(model_stu, sess=sess)
            elif FLAGS.attack == "pgd":
                attack_params = {'nb_iter': 10, "eps": FLAGS.eps, "eps_iter": FLAGS.eps_iter}
                attack_method = MadryEtAl(model_stu, sess=sess)
            elif FLAGS.attack == "momentum_pgd":
                attack_params = {'nb_iter': 10, "eps": FLAGS.eps, "eps_iter": FLAGS.eps_iter}
                attack_method = MomentumIterativeMethod(model_stu, sess=sess)
            elif FLAGS.attack == "pgd_l2":
                attack_params = {'nb_iter': 10, "eps": FLAGS.eps, "eps_iter": FLAGS.eps_iter}
                attack_method = MadryEtAl_L2(model_stu, sess=sess)
            test_attack_params = copy.deepcopy(attack_params)
            saver_save = tf.train.Saver(save_vars, max_to_keep=20)
            saver_res = tf.train.Saver(restore_vars, max_to_keep=20)
            if FLAGS.train_dir:
                train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train',
                                              sess.graph)
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            def test_net(adversary=False, saltpepper=None, name=""):
                steps_per_epoch = 10000 // batch_size
                loss_test = 0
                acc_1_test = 0
                acc_5_test = 0
                image_disturb = 0
                for step in range(1, steps_per_epoch+1):
                    x_v, y_v = sess.run([imgs_v, labels_v])
                    if adversary == True:
                        adv_x = attack_method.generate_np(x_v, y=y_v, **test_attack_params)
                        if FLAGS.attack_with_y:
                            adv_x = attack_method.generate_np(x_v, y=y_v, **test_attack_params)
                        else:
                            adv_x = attack_method.generate_np(x_v, **test_attack_params)
                    else:
                        adv_x = x_v
                    if saltpepper is not None:
                        u = np.random.uniform(size=list(adv_x.shape[:3]) + [1])
                        salt = (u >= 1 - saltpepper/2).astype(adv_x.dtype) * 256
                        pepper = - (u < saltpepper/2).astype(adv_x.dtype) * 256
                        adv_x = np.clip(adv_x + salt + pepper, 0, 255)
                    if FLAGS.bits != 8:
                        adv_x = quantize_input(adv_x, FLAGS.bits)
                    if FLAGS.median_blur != 0:
                        adv_x = median_blur(adv_x, FLAGS.median_blur)
                    image_disturb += np.abs(adv_x - x_v).sum()
                    if FLAGS.net_type == "resnet18":
                        relu_list_, logits_, logits_stu_, loss_v, acc_1, acc_5 = \
                                sess.run([relu_list_stu, logits, logits_stu, loss, accuracy, tea_accuracy],
                                                feed_dict={
                                                    x: adv_x,
                                                    stu_x: adv_x,
                                                    labels: y_v,
                                                    training: False,
                                                    training_stu: False
                                                })
                    else:
                        logits_, logits_stu_, loss_v, acc_1, acc_5 = \
                                sess.run([logits, logits_stu, loss, accuracy, tea_accuracy],
                                                feed_dict={
                                                    x: adv_x,
                                                    stu_x: adv_x,
                                                    labels: y_v,
                                                    training: False,
                                                    training_stu: False
                                                })
                    if FLAGS.output_path:
                        for logits_iter in range(logits_stu_.shape[0]):
                            f_write = None
                            if adversary == True:
                                if logits_stu_[logits_iter].argmax() == y_v[logits_iter].argmax():
                                    f_write = f_adv_correct
                                else:
                                    f_write = f_adv_wrong
                            else:
                                if logits_stu_[logits_iter].argmax() == y_v[logits_iter].argmax():
                                    f_write = f_normal_correct
                                else:
                                    f_write = f_normal_wrong
                            for relu_ in relu_list_:
                                f_write.write(str(relu_[logits_iter].argmax()) + ",")
                    print("\r\ttest steps: {}/{}".format(step, steps_per_epoch), end="")
                    loss_test += loss_v
                    acc_1_test += acc_1
                    acc_5_test += acc_5
                image_disturb /= (10000 * 64 * 64 * 3)
                loss_test /= steps_per_epoch
                acc_1_test /= steps_per_epoch
                acc_5_test /= steps_per_epoch
                log("\r\tTest {}: loss: {}; student accuracy: {:.2f} %; teacher accuracy: {:2f} %; Mean pixel distance: {:2f}. adversary: {}"\
                        .format(name, loss_test, acc_1_test * 100, acc_5_test * 100, image_disturb, adversary), flush=True)
                return acc_1_test, adversary

            def train_net(adversary=True, iter_=False, combine_=False):
                global_acc = [0, 0]
                for epoch in range(1, FLAGS.epochs+1):
                    start_time = time.time()
                    steps_per_epoch = 100000 // batch_size
                    loss_v_epoch = 0
                    acc_1_epoch = 0
                    acc_5_epoch = 0
                    if FLAGS.change_eps > 0 and FLAGS.attack != "none" and epoch % FLAGS.change_eps == 0:
                        attack_params["eps"] = attack_params["eps"] + FLAGS.eps_step
                        attack_params["eps_iter"] = attack_params["eps_iter"] + FLAGS.eps_iter_step
                    if FLAGS.sample_eps_method == "epoch": # uniformly sample eps from an interval
                        # FIXME: epoch-wise or step-wise?
                        if FLAGS.sample_eps:
                            attack_params["eps"] = np.random.uniform(FLAGS.sample_eps[0], FLAGS.sample_eps[1])
                        if FLAGS.sample_eps_iter:
                            attack_params["eps_iter"] = np.random.uniform(FLAGS.sample_eps_iter[0], FLAGS.sample_eps_iter[1])
                    # Train batches
                    for step in range(1, steps_per_epoch+1):
                        if FLAGS.sample_eps_method == "step": # uniformly sample eps from an interval
                            # FIXME: epoch-wise or step-wise?
                            if FLAGS.sample_eps:
                                attack_params["eps"] = np.random.uniform(FLAGS.sample_eps[0], FLAGS.sample_eps[1])
                            if FLAGS.sample_eps_iter:
                                attack_params["eps_iter"] = np.random.uniform(FLAGS.sample_eps_iter[0], FLAGS.sample_eps_iter[1])
                        x_v, y_v = sess.run([imgs_t, labels_t], feed_dict={adv_placeholder: adversary})
                        if FLAGS.aug_saltpepper is not None:
                            saltpepper = np.random.uniform(size=[FLAGS.batch_size, 1, 1, 1], low=FLAGS.aug_saltpepper[0], high=FLAGS.aug_saltpepper[1])
                            u = np.random.uniform(size=list(x_v.shape[:3]) + [1])
                            salt = (u >= 1 - saltpepper/2).astype(x_v.dtype) * 256
                            pepper = - (u < saltpepper/2).astype(x_v.dtype) * 256
                            stu_x_ = np.clip(x_v + salt + pepper, 0, 255)
                        else:
                            stu_x_ = x_v
                        if FLAGS.aug_gaussian is not None:
                            if isinstance(FLAGS.aug_gaussian, (tuple, list)):
                                eps = np.random.uniform(size=[FLAGS.batch_size, 1, 1, 1], low=FLAGS.aug_gaussian[0], high=FLAGS.aug_gaussian[1])
                            else:
                                eps = FLAGS.aug_gaussian
                            noise = np.random.normal(scale=eps/np.sqrt(3)*256, size=list(stu_x_.shape[:3]) + [1])
                            stu_x_ = np.clip(stu_x_ + noise, 0, 255)
                        if adversary == True:
                            if FLAGS.attack_with_y:
                                adv_x = attack_method.generate_np(stu_x_, y=y_v, **attack_params)
                            else:
                                adv_x = attack_method.generate_np(stu_x_, **attack_params)
                            if combine_:
                                adv_x = np.concatenate([adv_x, x_v])
                                y_v = np.concatenate([y_v, y_v])
                                x_v = np.concatenate([x_v, x_v])
                        else:
                            adv_x = stu_x_
                        if FLAGS.bits != 8:
                            x_v = quantize_input(x_v, FLAGS.bits)
                            adv_x = quantize_input(adv_x, FLAGS.bits)
                        if FLAGS.median_blur != 0:
                            x_v = median_blur(x_v, FLAGS.median_blur)
                            adv_x = median_blur(adv_x, FLAGS.median_blur)
                        if FLAGS.net_type == "resnet18":
                            glt, gls, soft_label_, soft_logits_, ce_, _, loss_v, dist, at, acc_1, acc_5, learning_rate_\
                                 = sess.run([group_list_teacher, group_list_student, soft_label, soft_logits, ce,\
                                  train_step, loss, distillation, at_loss, accuracy, tea_accuracy, learning_rate],
                                                               feed_dict={
                                                                   x: x_v,
                                                                   stu_x: adv_x,
                                                                   labels: y_v,
                                                                   training_stu: True,
                                                                   training: False
                                                               })
                        else:
                            soft_label_, soft_logits_, ce_, _, loss_v, dist, at, acc_1, acc_5, learning_rate_\
                                 = sess.run([soft_label, soft_logits, ce,\
                                  train_step, loss, distillation, at_loss, accuracy, tea_accuracy, learning_rate],
                                                               feed_dict={
                                                                   x: x_v,
                                                                   stu_x: adv_x,
                                                                   labels: y_v,
                                                                   training_stu: True,
                                                                   training: False
                                                               })
                        if FLAGS.debug == 1:
                            pdb.set_trace()
                        print("\rEpoch {}: steps {}/{} loss: {}/{}/{} adv: {} adv_eps: {}/{}".format(epoch, step, steps_per_epoch,\
                             loss_v, dist, at, adversary, attack_params["eps"], attack_params["eps_iter"]), end="")
                        loss_v_epoch += loss_v
                        acc_1_epoch += acc_1
                        acc_5_epoch += acc_5

                    loss_v_epoch /= steps_per_epoch
                    acc_1_epoch /= steps_per_epoch
                    acc_5_epoch /= steps_per_epoch

                    duration = time.time() - start_time
                    sec_per_batch = duration / (steps_per_epoch * batch_size)
                    log("\r{}: Epoch {}; (average) loss: {:.3f}; (average) student accuracy: {:.2f} %; (average) teacher accuracy: {:.2f} %. learning rate: {:.4f}. {:.3f} sec/batch; adv: {}"\
                          .format(datetime.now(), epoch, loss_v_epoch, acc_1_epoch * 100, acc_5_epoch * 100, learning_rate_, sec_per_batch, adversary), flush=True)
                    # End training batches

                    # Test on the validation set
                    if epoch % FLAGS.test_frequency == 0:
                        acc_1_test_nor = test_net(False, name="normal")
                        #if FLAGS.test_eps is None:
                        test_attack_params["eps"] = FLAGS.eps
                        test_attack_params["eps_iter"] = FLAGS.eps_iter
                        #acc_1_test_adv = test_net(True, name="1_adv_eps_{}_iter_{}".format(FLAGS.eps, FLAGS.eps_iter))
                        acc_1_test_adv = test_net(True, name="adv")
                        if FLAGS.test_eps is not None:
                            assert isinstance(FLAGS.test_eps, (tuple, list)) and isinstance(FLAGS.test_eps_iter, (tuple, list))
                            for _eps in FLAGS.test_eps:
                                for _eps_iter in FLAGS.test_eps_iter:
                                    test_attack_params["eps"] = _eps
                                    test_attack_params["eps_iter"] = _eps_iter
                                    test_net(True, name="adv_eps_{}_iter_{}".format(_eps, _eps_iter))
                        if FLAGS.test_saltpepper is not None:
                            if isinstance(FLAGS.test_saltpepper, (tuple, list)):
                                for sp in FLAGS.test_saltpepper:
                                    test_net(False, sp, "saltpepper_{}".format(sp))
                            else:
                                test_net(False, FLAGS.test_saltpepper, "saltpepper_{}".format(FLAGS.test_saltpepper))

                        if FLAGS.train_dir:
                            if global_acc[0] < acc_1_test_adv or global_acc[1] < acc_1_test_nor:
                                if global_acc[0] < acc_1_test_adv:
                                    global_acc[0] = acc_1_test_adv
                                if global_acc[1] < acc_1_test_nor:
                                    global_acc[0] = acc_1_test_nor
                                log("Saved model to: ", saver_save.save(sess, os.path.join(FLAGS.train_dir, str(acc_1_test_adv) + "_" + str(acc_1_test_nor))
                                     , global_step=global_step))

                    if iter_:
                        adversary = not adversary

                if FLAGS.train_dir:
                    if not os.path.exists(FLAGS.train_dir):
                        subprocess.check_call("mkdir -p {}".format(FLAGS.train_dir),
                                              shell=True)
                    log("Saved model to: ", saver_save.save(sess, FLAGS.train_dir))
            #log(tf.get_default_graph().get_collection("global_variables"))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)    

            if FLAGS.test_only:
                if FLAGS.load_file_tea and not FLAGS.load_file_stu:
                    saver_res.restore(sess, FLAGS.load_file_tea)
                    if FLAGS.more_blocks:
                        remove_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                            FLAGS.stu_namescope+"/more_blocks" if FLAGS.stu_namescope else "more_blocks")
                        for var_ in remove_vars:
                            if var_ in save_vars:
                                save_vars.remove(var_)
                    var_mapping_dct = {var.op.name.replace(FLAGS.stu_namescope + "/",\
                        ""): var for var in save_vars}
                    load_stu = tf.train.Saver(var_mapping_dct)
                    load_stu.restore(sess, FLAGS.load_file_tea)
                else:
                    saver_save.restore(sess, FLAGS.load_file_stu)
                    if FLAGS.load_file_tea:
                        saver_res.restore(sess, FLAGS.load_file_tea)
                if FLAGS.test_eps is not None or FLAGS.test_eps_pair is not None:
                    if isinstance(FLAGS.test_eps, (tuple, list)) and isinstance(FLAGS.test_eps_iter, (tuple, list)):
                        for _eps in FLAGS.test_eps:
                            for _eps_iter in FLAGS.test_eps_iter:
                                test_attack_params["eps"] = _eps
                                test_attack_params["eps_iter"] = _eps_iter
                                test_net(True, name="adv_eps_{}_iter_{}".format(_eps, _eps_iter))
                    elif isinstance(FLAGS.test_eps_pair, (tuple, list)):
                        for _eps, _eps_iter in FLAGS.test_eps_pair:
                            test_attack_params["eps"] = _eps
                            test_attack_params["eps_iter"] = _eps_iter
                            test_net(True, name="adv_eps_{}_iter_{}".format(_eps, _eps_iter))                  
                if FLAGS.test_saltpepper is not None:
                    if isinstance(FLAGS.test_saltpepper, (tuple, list)):
                        for sp in FLAGS.test_saltpepper:
                            test_net(False, sp, "saltpepper_{}".format(sp))
                    else:
                        test_net(False, FLAGS.test_saltpepper, "saltpepper_{}".format(FLAGS.test_saltpepper))
                coord.request_stop()
                coord.join(threads)
                sys.exit(0)

            # if FLAGS.load_denoiser:
            #     saver_denoiser.restore(sess, FLAGS.load_denoiser)
            if FLAGS.load_file_tea and not FLAGS.load_file_stu:
                saver_res.restore(sess, FLAGS.load_file_tea)
                if FLAGS.more_blocks:
                    remove_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                        FLAGS.stu_namescope+"/more_blocks" if FLAGS.stu_namescope else "more_blocks")
                    for var_ in remove_vars:
                        if var_ in save_vars:
                            save_vars.remove(var_)
                var_mapping_dct = {var.op.name.replace(FLAGS.stu_namescope + "/",\
                    ""): var for var in save_vars}
                load_stu = tf.train.Saver(var_mapping_dct)
                load_stu.restore(sess, FLAGS.load_file_tea)
                test_net(name="loaded_teacher_copy")
                log("Start training...")
                train_net(iter_=FLAGS.iter_train, combine_=FLAGS.combine_train)
            # Training
            elif FLAGS.load_file_stu:
                saver_save.restore(sess, FLAGS.load_file_stu)
                if FLAGS.load_file_tea:
                    saver_res.restore(sess, FLAGS.load_file_tea)
                    test_net(False, name="loaded_stu_tea")
                    test_net(True, name="adv loaded_stu")
                    if FLAGS.test_saltpepper is not None:
                        if isinstance(FLAGS.test_saltpepper, (tuple, list)):
                            for sp in FLAGS.test_saltpepper:
                                test_net(False, sp, name="loaded saltpepper_{}".format(sp))
                        else:
                            test_net(False, FLAGS.test_saltpepper, name="loaded saltpepper_{}".format(FLAGS.test_saltpepper))
                else:
                    test_net(False, name="loaded_stu")
                    test_net(True, name="adv loaded_stu")
                log("Start training...")
                train_net(iter_=FLAGS.iter_train, combine_=FLAGS.combine_train)
            else:
                print("error: no input file.")

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, default="",
                        help="Directory for storing snapshots")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU used for training/validation")
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Config files")
    parser.add_argument("--load_file_stu", type=str, default="",
                        help="Load student model")
    parser.add_argument("--load_file_tea", type=str, default="",
                        help="Load teacher model")
    parser.add_argument("--load_denoiser", type=str, default="",
                        help="load denoiser")
    parser.add_argument("--output_path", type=str, default="",
                        help="Output path information")
    parser.add_argument("--test-only", action="store_true", default=False, help="Only run test")

    # args, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with open(args.config) as config_file:
        # config = json.load(config_file)
        config = yaml.load(config_file)
    FLAGS = settings(config, args)
    if not args.test_only:
        if not os.path.exists(FLAGS.train_dir):
            subprocess.check_call("mkdir -p {}".format(FLAGS.train_dir),
                                  shell=True)
        FLAGS.log_file = open(FLAGS.log_file, "w")
        shutil.copyfile(sys.argv[0], os.path.join(FLAGS.train_dir, "train.py"))
        # shutil.copyfile(args.config, os.path.join(FLAGS.train_dir, "config.json"))
        shutil.copyfile(args.config, os.path.join(FLAGS.train_dir, "config.yaml"))
    if FLAGS.load_file_tea:
        log("Load teacher model: {}".format(FLAGS.load_file_tea))
    if FLAGS.load_file_stu:
        log("Load student model: {}".format(FLAGS.load_file_stu))
    if not FLAGS.train_dir:
        log("WARNING: model will not be saved if `--train_dir` option is not given.")
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    tf.app.run(main=main, argv=[sys.argv[0]])

