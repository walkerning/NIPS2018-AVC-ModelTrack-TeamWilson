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
import glob
import random
import numpy as np

from imgaug import augmenters as iaa
import tensorflow as tf

from nics_at import utils

class Dataset(object):
    def __init__(self, batch_size, epochs, aug_saltpepper, aug_gaussian, generated_adv=[], num_threads=2, capacity=1024, more_augs=False):
        self.batch_size = batch_size
        if isinstance(num_threads, int):
            self.num_threads = {"train": num_threads, "val": num_threads}
        else:
            assert isinstance(num_threads, dict)
            self.num_threads = num_threads
        self.gen_epochs = epochs
        self.aug_saltpepper = aug_saltpepper
        self.aug_gaussian = aug_gaussian
        self.capacity = capacity
        self.generated_adv = generated_adv
        self.generated_adv_num = len(generated_adv)
        self.more_augs = more_augs
        if self.more_augs:
            rarely = lambda aug: iaa.Sometimes(0.1, aug)
            sometimes = lambda aug: iaa.Sometimes(0.25, aug)
            often = lambda aug: iaa.Sometimes(0.5, aug)
            self.seq = iaa.Sequential([
                # iaa.Fliplr(0.5),
                often(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.12, 0)},
                    rotate=(-10, 10),
                    shear=(-8, 8),
                    order=[0, 1],
                    cval=(0, 255),
                )),
                iaa.SomeOf((0, 4), [
                    rarely(
                        iaa.Superpixels(
                            p_replace=(0, 0.3),
                            n_segments=(20, 200)
                        )
                    ),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 2.0)),
                        iaa.AverageBlur(k=(2, 4)),
                        iaa.MedianBlur(k=(3, 5)),
                    ]),
                    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),
                    rarely(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.3)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.06 * 255), per_channel=0.5
                    ),
                    iaa.OneOf([
                        iaa.Dropout((0.0, 0.05), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.05), size_percent=(0.01, 0.05),
                            per_channel=0.2
                        ),
                    ]),
                    rarely(iaa.Invert(0.05, per_channel=True)),
                    often(iaa.Add((-40, 40), per_channel=0.5)),
                    iaa.Multiply((0.7, 1.3), per_channel=0.5),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)
                    ),
        
                ], random_order=True),
                iaa.Fliplr(0.5),
                iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ], random_order=True)  # apply augmenters in random order
            def aug(input_):
                output_ = self.seq.augment_image(input_).astype(np.float32)
                return output_
            self.more_aug = aug
        self.label_dict, self.class_description = self.build_label_dicts()
        self._gen = False

    def load_filenames_labels(self, mode):
        filenames_labels = []
        if mode == 'train':
            filenames = glob.glob('./tiny-imagenet-200/train/*/images/*.JPEG')
            for filename in filenames:
                match = re.search(r'n\d+', filename)
                label = str(self.label_dict[match.group()])
                filename_label = [filename, label]
                filename_label += [os.path.join(adv_cfg["path"], mode, os.path.basename(filename).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv]
                filenames_labels.append(tuple(filename_label))
        elif mode == 'val':
            with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    split_line = line.split('\t')
                    filename = './tiny-imagenet-200/val/images/' + split_line[0]
                    label = str(self.label_dict[split_line[1]])
                    filename_label = [filename, label]
                    filename_label += [os.path.join(adv_cfg["path"], mode, os.path.basename(filename).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv]
                    filenames_labels.append(tuple(filename_label))
        setattr(self, mode + "_num", len(filenames_labels))
        return filenames_labels

    def build_label_dicts(self):
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
    
    def read_image(self, filename_q, mode):
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
            auged_img = tf.image.random_flip_left_right(img)
            auged_img = tf.image.random_brightness(auged_img, 0.15)
            auged_img = tf.image.random_contrast(auged_img, 0.8, 1.25)
            auged_img = tf.image.random_hue(auged_img, 0.1)
            auged_img = tf.image.random_saturation(auged_img, 0.8, 1.25)
            if self.aug_saltpepper is not None:
                p = tf.random_uniform([], minval=self.aug_saltpepper[0], maxval=self.aug_saltpepper[1])
                u = tf.expand_dims(tf.random_uniform([64, 64], maxval=1.0), axis=-1)
                salt = tf.cast(u >= 1 - p/2, tf.float32) * 256
                pepper = - tf.cast(u < p/2, tf.float32) * 256
                auged_img = tf.clip_by_value(auged_img + salt + pepper, 0, 255)
            if self.more_augs:
                utils.log("Use more augmentation!")
                auged_img = tf.py_func(self.more_aug, [tf.cast(auged_img, tf.uint8)], tf.float32)
                auged_img = tf.clip_by_value(auged_img, 0, 255) # 需要吗?
            elif self.aug_gaussian is not None:
                if isinstance(self.aug_gaussian, (tuple, list)):
                    eps = tf.random_uniform([], minval=self.aug_gaussian[0], maxval=self.aug_gaussian[1])
                else:
                    eps = self.aug_gaussian
                noise = tf.random_normal([64, 64, 3], stddev=eps/np.sqrt(3)*256)
                auged_img = tf.clip_by_value(auged_img + noise, 0, 255)
        else:
            auged_img = img

        auged_img = tf.pad(auged_img, tf.constant([[4, 4], [4, 4], [0, 0]]))
        auged_img = tf.random_crop(auged_img, [64, 64, 3])
        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)
        adv_imgs = []
        for i in range(self.generated_adv_num):
            adv_filename = item[i+2]
            file = tf.read_file(adv_filename)
            data = tf.decode_raw(file, out_type=tf.uint8)
            data = tf.reshape(data, (64, 64, 3))
            adv_imgs.append(tf.cast(data, tf.float32))
        adv_imgs = tf.reshape(tf.stack(adv_imgs), (self.generated_adv_num, 64, 64, 3))
        return [img, auged_img, label, adv_imgs]
    
    
    def batch_q(self, mode):
        """Return batch of images using filename Queue
    
        Args:
            mode: 'train' or 'val'
            config: training configuration object
    
        Returns:
            imgs: tf.uint8 tensor [batch_size, height, width, channels]
            auged_img: tf.uint8 tensor [batch_size, height, width, channels]
            labels: tf.uint8 tensor [batch_size,]
            adv_imgs: tf.uint8 tensor [batch_size, adv_num, height, width, channels]
        """
        self.filenames_labels = self.load_filenames_labels(mode)
        random.shuffle(self.filenames_labels)
        filename_q = tf.train.input_producer(self.filenames_labels,
                                             num_epochs=self.gen_epochs * 4 if mode == "val" else self.gen_epochs,
                                             shuffle=True,
                                             name="data_producer_" + mode)

        return tf.train.batch_join([self.read_image(filename_q, mode) for i in range(self.num_threads[mode])],
                                   self.batch_size, shapes=[(64, 64, 3), (64, 64, 3), (), (self.generated_adv_num, 64, 64, 3)],
                                   capacity=self.capacity)

    @property
    def data_tensors(self):
        if not self._gen:
            self._gen = True
            with tf.device('/cpu:0'):
                self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t = self.batch_q("train")
                self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v = self.batch_q("val")
    
            self.labels_t = tf.one_hot(self.labels_t, 200)
            self.labels_v = tf.one_hot(self.labels_v, 200)
        return (self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t), (self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v)
