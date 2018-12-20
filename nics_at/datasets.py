# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import re
import glob
import random
import numpy as np

import tensorflow as tf

from nics_at import utils

class Dataset(object):
    def __init__(self, FLAGS):
    # def __init__(self, batch_size, epochs, aug_saltpepper, aug_gaussian, generated_adv=[], num_threads=2, capacity=1024, more_augs=False, test_path=None, dataset_info={}):
        self.FLAGS = FLAGS
        batch_size = FLAGS.batch_size
        epochs = FLAGS.epochs
        aug_saltpepper = FLAGS.aug_saltpepper
        aug_gaussian = FLAGS.aug_gaussian
        generated_adv = FLAGS.generated_adv
        num_threads = FLAGS.num_threads
        capacity = FLAGS.capacity
        more_augs = FLAGS.more_augs
        test_path = FLAGS.test_path
        dataset_info = FLAGS.dataset_info

        self.dataset_info = dataset_info
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
        if self.generated_adv is None:
            self.generated_adv = []
        self.generated_adv_num = len(self.generated_adv)
        self.test_path = test_path
        self.more_augs = more_augs
        if self.more_augs:
            from imgaug import augmenters as iaa
            rarely = lambda aug: iaa.Sometimes(0.1, aug)
            sometimes = lambda aug: iaa.Sometimes(0.25, aug)
            often = lambda aug: iaa.Sometimes(0.5, aug)
            if self.more_augs == "v3":
                utils.log("v3: only add affine transformation")
                often = lambda aug: iaa.Sometimes(0.8, aug)
                self.seq = iaa.Sequential([
                    often(iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # zoom in/out
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.05, 0.05)}, # translation
                        rotate=(-15, 15),
                        shear=(-8, 8),
                        order=[0, 1], # interpolation order values
                        cval=(0, 255), # fill value
                    ))], random_order=True)
            elif self.more_augs == "v4":
                utils.log("v4: add affine transformation, dropout, grayscale, blur")
                often = lambda aug: iaa.Sometimes(0.8, aug)
                self.seq = iaa.Sequential([
                    often(iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # zoom in/out
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.05, 0.05)}, # translation
                        rotate=(-15, 15),
                        shear=(-8, 8),
                        order=[0, 1], # interpolation order values
                        cval=(0, 255), # fill value
                    )),
                    sometimes(iaa.OneOf([
                        iaa.Dropout((0.0, 0.05), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.05), size_percent=(0.03, 0.07),
                            per_channel=0.2
                        )])),
                    sometimes(iaa.Grayscale(alpha=(0.0, 1.0))),
                    sometimes(iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)),
                            iaa.AverageBlur(k=3)
                    ])),
                ], random_order=True)
            elif self.more_augs == "v5":
                utils.log("v5: only add rotation")
                often = lambda aug: iaa.Sometimes(0.8, aug)
                self.seq = iaa.Sequential([
                    often(iaa.Affine(
                        # scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # zoom in/out
                        # translate_percent={"x": (-0.1, 0.1), "y": (-0.05, 0.05)}, # translation
                        rotate=(-15, 15),
                        order=[0, 1], # interpolation order values
                        # cval=(0, 255), # fill value
                        cval=0
                    ))], random_order=True)
            else:
                self.seq = iaa.Sequential([
                    # iaa.Fliplr(0.5),
                    often(iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # zoom in/out
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.05, 0.05)}, # translation
                        rotate=(-10, 10),
                        shear=(-8, 8),
                        order=[0, 1], # interpolation order values
                        cval=(0, 255), # fill value
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
                            iaa.AverageBlur(k=3),
                            iaa.MedianBlur(k=3),
                        ]),
                        iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),
                        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),
                        # rarely(iaa.OneOf([
                        #     iaa.EdgeDetect(alpha=(0, 0.3)),
                        #     iaa.DirectedEdgeDetect(
                        #         alpha=(0, 0.7), direction=(0.0, 1.0)
                        #     ),
                        # ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.06 * 255), per_channel=0.5
                        ),
                        iaa.OneOf([
                            iaa.Dropout((0.0, 0.05), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.05), size_percent=(0.05, 0.05),
                                per_channel=0.2
                            ),
                        ]),
                        # rarely(iaa.Invert(0.05, per_channel=True)),
                        # often(iaa.Add((-40, 40), per_channel=0.5)),
                        # iaa.Multiply((0.7, 1.3), per_channel=0.5),
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

    def start(self, sess):
        self.sess = sess
        # Start the queue runner!
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def sync_epoch(self, epoch):
        pass

    def end(self):
        self.coord.request_stop()
        self.coord.join(self.threads)

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
                                             shuffle=mode=="train",
                                             name="data_producer_" + mode)

        return tf.train.batch_join([self.read_image(filename_q, mode) for i in range(self.num_threads[mode])],
                                   self.batch_size, shapes=[tuple(self.image_shape), tuple(self.image_shape), (),
                                                            tuple([self.generated_adv_num] + list(self.image_shape))],
                                   capacity=self.capacity)

    @property
    def data_tensors(self):
        if not self._gen:
            self._gen = True
            with tf.device('/cpu:0'):
                self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t = self.batch_q("train")
                self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v = self.batch_q("val")

            self.labels_t = tf.one_hot(self.labels_t, self.num_labels)
            self.labels_v = tf.one_hot(self.labels_v, self.num_labels)
        return (self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t), (self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v)

class TinyImageNetDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(TinyImageNetDataset, self).__init__(*args, **kwargs)
        # dataset_info is the specific dataset configs
        self.use_imgnet1k = self.dataset_info.get("use_imgnet1k", False)
        self.image_shape = [64,64,3]
        self.num_labels = 200

    def load_filenames_labels(self, mode):
        filenames_labels = []
        if mode == 'train':
            filenames = glob.glob('./tiny-imagenet-200/train/*/images/*.JPEG')
            for filename in filenames:
                match = re.search(r'n\d+', filename)
                label = str(self.label_dict[match.group()])
                filename_label = [filename, label, "0"]
                filename_label += [os.path.join(adv_cfg["path"], mode, os.path.basename(filename).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv]
                filenames_labels.append(tuple(filename_label))
            if self.use_imgnet1k:
                filenames = glob.glob(os.path.abspath('./imagenet1k-tinysubset/train/*/*.JPEG'))
                for filename in filenames:
                    match = re.search(r'n\d+', filename)
                    label = str(self.label_dict[match.group()])
                    filename_label = [filename, label, "1"]
                    # for imgnet1k examples, we do not have time to generate blackbox adversarials
                    filename_label += [os.path.join(adv_cfg["path"], mode, os.path.basename(filename).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv] # will not use
                    filenames_labels.append(tuple(filename_label))
        elif mode == 'val':
            with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
                for line in f.readlines():
                    split_line = line.split('\t')
                    filename = './tiny-imagenet-200/val/images/' + split_line[0]
                    label = str(self.label_dict[split_line[1]])
                    filename_label = [filename, label, "0"]
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
        is_imgnet1k = tf.cast(tf.string_to_number(item[2]), tf.bool)
        file = tf.read_file(filename)
        img = tf.image.decode_jpeg(file, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.cond(is_imgnet1k,
                      lambda : tf.image.resize_images(img, [64, 64]), lambda: img)

        if mode == "train":
            auged_img = tf.image.random_flip_left_right(img)
            auged_img = tf.image.random_brightness(auged_img, 0.15)
            auged_img = tf.image.random_contrast(auged_img, 0.8, 1.25)
            auged_img = tf.image.random_hue(auged_img, 0.1)
            auged_img = tf.image.random_saturation(auged_img, 0.8, 1.25)
            if self.more_augs:
                utils.log("Use more augmentation!")
                auged_img = tf.py_func(self.more_aug, [tf.cast(auged_img, tf.uint8)], tf.float32)
                auged_img = tf.clip_by_value(auged_img, 0, 255) # 需要吗?
            if self.aug_saltpepper is not None:
                p = tf.random_uniform([], minval=self.aug_saltpepper[0], maxval=self.aug_saltpepper[1])
                u = tf.expand_dims(tf.random_uniform([64, 64], maxval=1.0), axis=-1)
                salt = tf.cast(u >= 1 - p/2, tf.float32) * 256
                pepper = - tf.cast(u < p/2, tf.float32) * 256
                auged_img = tf.clip_by_value(auged_img + salt + pepper, 0, 255)
            if self.aug_gaussian is not None and (not self.more_augs or self.more_augs in {"v3", "v4", "v5"}):
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
        if self.use_imgnet1k:
            def read_adv():
                adv_imgs = []
                for i in range(self.generated_adv_num):
                    adv_filename = item[i+3]
                    file = tf.read_file(adv_filename)
                    data = tf.decode_raw(file, out_type=tf.uint8)
                    data = tf.reshape(data, (64, 64, 3))
                    adv_imgs.append(tf.cast(data, tf.float32))
                adv_imgs = tf.reshape(tf.stack(adv_imgs), (self.generated_adv_num, 64, 64, 3))
                return adv_imgs
            adv_imgs = tf.cond(is_imgnet1k,
                          lambda : tf.reshape(tf.stack([auged_img] * self.generated_adv_num), (self.generated_adv_num, 64, 64, 3)) , read_adv)
        else:
            adv_imgs = []
            for i in range(self.generated_adv_num):
                adv_filename = item[i+3]
                file = tf.read_file(adv_filename)
                data = tf.decode_raw(file, out_type=tf.uint8)
                data = tf.reshape(data, (64, 64, 3))
                adv_imgs.append(tf.cast(data, tf.float32))
            adv_imgs = tf.reshape(tf.stack(adv_imgs), (self.generated_adv_num, 64, 64, 3))
        return [img, auged_img, label, adv_imgs]


class Cifar10Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(Cifar10Dataset, self).__init__(*args, **kwargs)
        # dataset_info is the specific dataset configs
        self.image_shape = [32,32,3]
        self.num_labels = 10

    def load_filenames_labels(self, mode):
        import yaml
        filenames_labels = []
        if self.test_path and mode == "val":
            yaml_fname = self.test_path
        else:
            yaml_fname = "./cifar10_{}.yaml".format(mode)
        with open(yaml_fname, "r") as yaml_f:
            fname_label_dct = yaml.load(yaml_f)
        for fname, label in fname_label_dct.iteritems():
            filenames_label = [fname, str(label)]
            filenames_label += [os.path.join(adv_cfg["path"], mode, os.path.basename(fname).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv]
            filenames_labels.append(tuple(filenames_label))
        setattr(self, mode + "_num", len(filenames_labels))
        return filenames_labels

    def build_label_dicts(self):
        label_dict = {n: i for i, n in enumerate(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])}
        class_description = {}
        return label_dict, class_description

    def read_image(self, filename_q, mode):
        item = filename_q.dequeue()
        filename = item[0]
        label = item[1]
        file = tf.read_file(filename)
        img = tf.reshape(tf.decode_raw(file, out_type=tf.uint8), (32, 32, 3))
        img = tf.cast(img, tf.float32)

        if mode == "train":
            auged_img = tf.image.random_flip_left_right(img)
            # auged_img = tf.image.random_brightness(auged_img, 0.15)
            # auged_img = tf.image.random_contrast(auged_img, 0.8, 1.25)
            # auged_img = tf.image.random_hue(auged_img, 0.1)
            # auged_img = tf.image.random_saturation(auged_img, 0.8, 1.25)
            if self.aug_saltpepper is not None:
                p = tf.random_uniform([], minval=self.aug_saltpepper[0], maxval=self.aug_saltpepper[1])
                u = tf.expand_dims(tf.random_uniform([32, 32], maxval=1.0), axis=-1)
                salt = tf.cast(u >= 1 - p/2, tf.float32) * 256
                pepper = - tf.cast(u < p/2, tf.float32) * 256
                auged_img = tf.clip_by_value(auged_img + salt + pepper, 0, 255)
            if self.aug_gaussian is not None and (not self.more_augs or self.more_augs in {"v3", "v4", "v5"}):
                if isinstance(self.aug_gaussian, (tuple, list)):
                    eps = tf.random_uniform([], minval=self.aug_gaussian[0], maxval=self.aug_gaussian[1])
                else:
                    eps = self.aug_gaussian
                noise = tf.random_normal([32, 32, 3], stddev=eps/np.sqrt(3)*256)
                auged_img = tf.clip_by_value(auged_img + noise, 0, 255)
            auged_img = tf.pad(auged_img, tf.constant([[4, 4], [4, 4], [0, 0]]))
            auged_img = tf.random_crop(auged_img, [32, 32, 3])
        else: # val
            auged_img = img

        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)
        adv_imgs = []
        for i in range(self.generated_adv_num):
            adv_filename = item[i+2]
            file = tf.read_file(adv_filename)
            data = tf.decode_raw(file, out_type=tf.uint8)
            data = tf.cast(tf.reshape(data, (32, 32, 3)), tf.float32)
            adv_imgs.append(data)
        adv_imgs = tf.reshape(tf.stack(adv_imgs), (self.generated_adv_num, 32, 32, 3))
        return [img, auged_img, label, adv_imgs]

class SVHNDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super(SVHNDataset, self).__init__(*args, **kwargs)
        # dataset_info is the specific dataset configs
        self.image_shape = [32,32,3]
        self.num_labels = 10

    def load_filenames_labels(self, mode):
        import yaml
        filenames_labels = []
        if self.test_path and mode == "val":
            yaml_fname = self.test_path
        else:
            if mode == "train" and self.dataset_info.get("with_extra", False):
                yaml_fname = "./svhn_extra.yaml"
            else:
                yaml_fname = "./svhn_{}.yaml".format(mode)
        with open(yaml_fname, "r") as yaml_f:
            fname_label_dct = yaml.load(yaml_f)
        for fname, label in fname_label_dct.iteritems():
            new_mode = "extra" if (self.dataset_info.get("with_extra", False) and mode == "train") else mode
            filenames_label = [os.path.join("svhn", new_mode, fname), str(label)]
            filenames_label += [os.path.join(adv_cfg["path"], new_mode, os.path.basename(fname).split(".")[0] + "." + adv_cfg["suffix"]) for adv_cfg in self.generated_adv]
            filenames_labels.append(tuple(filenames_label))
        setattr(self, mode + "_num", len(filenames_labels))
        return filenames_labels

    def build_label_dicts(self):
        label_dict = {n: i for i, n in enumerate(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])}
        class_description = {}
        return label_dict, class_description

    def read_image(self, filename_q, mode):
        item = filename_q.dequeue()
        filename = item[0]
        label = item[1]
        file = tf.read_file(filename)
        img = tf.reshape(tf.decode_raw(file, out_type=tf.uint8), (32, 32, 3))
        img = tf.cast(img, tf.float32)

        auged_img = img
        if mode == "train":
            # auged_img = tf.image.random_brightness(auged_img, 0.15)
            # auged_img = tf.image.random_contrast(auged_img, 0.8, 1.25)
            # auged_img = tf.image.random_hue(auged_img, 0.1)
            # auged_img = tf.image.random_saturation(auged_img, 0.8, 1.25)
            if self.aug_saltpepper is not None:
                p = tf.random_uniform([], minval=self.aug_saltpepper[0], maxval=self.aug_saltpepper[1])
                u = tf.expand_dims(tf.random_uniform([32, 32], maxval=1.0), axis=-1)
                salt = tf.cast(u >= 1 - p/2, tf.float32) * 256
                pepper = - tf.cast(u < p/2, tf.float32) * 256
                auged_img = tf.clip_by_value(auged_img + salt + pepper, 0, 255)
            if self.aug_gaussian is not None and (not self.more_augs or self.more_augs in {"v3", "v4", "v5"}):
                if isinstance(self.aug_gaussian, (tuple, list)):
                    eps = tf.random_uniform([], minval=self.aug_gaussian[0], maxval=self.aug_gaussian[1])
                else:
                    eps = self.aug_gaussian
                noise = tf.random_normal([32, 32, 3], stddev=eps/np.sqrt(3)*256)
                auged_img = tf.clip_by_value(auged_img + noise, 0, 255)

        label = tf.string_to_number(label, tf.int32)
        label = tf.cast(label, tf.uint8)
        adv_imgs = []
        for i in range(self.generated_adv_num):
            adv_filename = item[i+2]
            file = tf.read_file(adv_filename)
            data = tf.decode_raw(file, out_type=tf.uint8)
            data = tf.cast(tf.reshape(data, (32, 32, 3)), tf.float32)
            adv_imgs.append(data)
        adv_imgs = tf.reshape(tf.stack(adv_imgs), (self.generated_adv_num, 32, 32, 3))
        return [img, auged_img, label, adv_imgs]

from gray_datasets import GrayCifar10Dataset, GrayTIDataset

type_dataset_map = {
    "cifar10": Cifar10Dataset,
    "tinyimagenet": TinyImageNetDataset,
    "gray_cifar10": GrayCifar10Dataset,
    "gray_tinyimagenet": GrayTIDataset,
    "svhn": SVHNDataset
}

def get_dataset_cls(type_):
    return type_dataset_map[type_]
