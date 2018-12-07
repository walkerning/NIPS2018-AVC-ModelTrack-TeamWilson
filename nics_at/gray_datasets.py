# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from nics_at.models import QCNN
from nics_at.attacks import Attack
from nics_at.datasets import Dataset, Cifar10Dataset, TinyImageNetDataset
from nics_at import utils
from utils import AvailModels

class GrayDataset(Dataset):
    def __init__(self, FLAGS):
        super(GrayDataset, self).__init__(FLAGS)
        # 1. We can use the same graph to enable copy ops, and all the ops (both data generation in queue runner threads and model training in main thread) are runned using the same session;
        #     disadvantage: need another namescope for stu_;
        # 2. Or should we manage two session, all queue runners should use another session (session 1) as it will run ops in another graph, when using separate graph and session:
        #     on weights copy  : eval all vars of model_on_device0 using session 0 and assign these vars to model_on_device1 using session 1
        #     on fetching data : just using session1 to fetch data
        #
        # i think solution 1 is more elegant in fact... maybe i'm wrong but let's go with solution 1 for now...
        # **NOTE**: all attacks build here will be directly used in __generated__. so just add __generated__ to the "train_models" configuration

        self.sync_every = FLAGS.sync_every
        self._build_queue_trainval()

        # Construct all the models in this device
        self.device = FLAGS.gray_dataset_device
        with tf.device('/gpu:{}'.format(self.device)):
            # **NOTE**: these models will use a different registry to avoid accidentaly causing too many data movement between devices
            #           for considerations of multiple aspects(especially **efficiency**), do not support foolbox type attack
            self.available_models_cfgs = FLAGS.additional_models_gray
            self.available_models = [QCNN.create_model(m_cfg) for m_cfg in self.available_models_cfgs]
            [AvailModels.add(m, None, None, tag="gray_dataset") for m in self.available_models]

            # Construct all the attacks in this device
            self.available_attacks = [Attack.create_attack(None, a_cfg) for a_cfg in (self.FLAGS.available_attacks_gray or [])]

        self.has_following = any(m_cfg.get("follow", None) is not None for m_cfg in self.available_models_cfgs) # whether following models exists
        self.total_generated_adv_num = self.generated_adv_num + len(self.available_attacks)
        self.started = False

    def _build_queue_trainval(self):
        def _build_queue(mode):
            cont_m2t_queue = tf.FIFOQueue(4, dtypes=[tf.bool], shapes=[()]) # main to thread(master) queue
            cont_t2m_queue = tf.FIFOQueue(4, dtypes=[tf.bool], shapes=[()]) # thread(master) to main queue
            m2s_queue = tf.FIFOQueue((self.num_threads[mode] - 1) * 2, dtypes=[tf.bool], shapes=[()]) # master to slave queue
            s2m_queue = tf.FIFOQueue((self.num_threads[mode] - 1) * 2, dtypes=[tf.bool], shapes=[()]) # slave to master queue
            # calculate
            num_per_epoch = getattr(self, "{}_num".format(mode))
            # Number of datum to process between the barrier-sync(per round) per data/queue-runner thread
            # sync_every = self.sync_every
            # num_per_thread_round = int(np.ceil(num_per_epoch * self.sync_every / self.num_threads[mode]))
            # add 1 initial epoch for val gen when initial test exists
            sync_every = tf.Variable(initial_value=(self.sync_every+1) if mode == "val" and not self.FLAGS.no_init_test else self.sync_every, trainable=False, name="sync_every_{}".format(mode), dtype=tf.int32)
            num_per_thread_round = tf.cast(tf.ceil(tf.cast(sync_every, tf.float32) * num_per_epoch / self.num_threads[mode]), tf.int32)

            return cont_m2t_queue, cont_t2m_queue, m2s_queue, s2m_queue, num_per_thread_round, sync_every
        # Prepare train/val filename queue
        self.filenames_labels_dct = {}
        self.filenames_queues_dct = {}
        for mode in ["train", "val"]:
            filenames_labels = self.load_filenames_labels(mode)
            random.shuffle(filenames_labels)
            filename_q = tf.train.input_producer(filenames_labels,
                                                 num_epochs=self.gen_epochs * 4 if mode == "val" else self.gen_epochs,
                                                 shuffle=mode=="train",
                                                 name="data_producer_"+mode)
            self.filenames_labels_dct[mode] = filenames_labels
            self.filenames_queues_dct[mode] = filename_q

        # Prepare the queues used to synchronize queue-runner threads and the main-training thread (train and val)
        self.queues_dct = {mode: _build_queue(mode) for mode in ["train", "val"]}
        self.processed_per_threads = {
            mode: [
                tf.get_variable("data_producer_{}/data_count/{}".format(mode, i),
                                shape=[], dtype=tf.int32, initializer=tf.zeros_initializer())
                for i in range(self.num_threads[mode])
            ] for mode in ["train", "val"]
        }

    def read_image_without_following(self, mode, ind_, major):
        return self._read_image_with_attack(mode)

    def read_image_with_following(self, mode, ind_, major):
        cont_m2t_queue, cont_t2m_queue, m2s_queue, s2m_queue, per_thread_num = self.queues_dct[mode][:5]
        num = self.processed_per_threads[mode][ind_]
        # Use per-thread num to avolid concurrent access to the same variable in multiple threads, alternatively, one can use use_locking in tf.assign_add...
        # because per-thread counting might hurt load balancing performance...
        other_num = self.num_threads[mode] - 1
        def wait_and_handle(num):
            def _func():
                if major: # "major" data thread will communicate with the training thread, on behalf of all the data threads
                    # 1. <-s2m_queue (waiting for slaves to reach the round end)
                    # 2. ->cont_t2m_queue (tell main thread: this round of generation ended)
                    # 3. <-cont_m2t_queue (waiting for main to sync and start the data generation again)
                    # 4. 0 => num (new round!)
                    # 5. ->m2s_queue (tell slaves to begin)
                    # 6. +1 => num
                    with tf.control_dependencies([s2m_queue.dequeue_many(other_num)]),\
                         tf.control_dependencies([cont_t2m_queue.enqueue(tf.constant(True))]),\
                         tf.control_dependencies([cont_m2t_queue.dequeue()]),\
                         tf.control_dependencies([tf.assign(num, 0)]),\
                         tf.control_dependencies([m2s_queue.enqueue_many([tf.constant([True])] * other_num)]),\
                         tf.control_dependencies([tf.assign_add(num, 1)]):
                        return self._read_image_with_attack(mode)
                else: # if not major
                    with tf.control_dependencies([s2m_queue.enqueue(tf.constant(True))]),\
                         tf.control_dependencies([m2s_queue.dequeue()]),\
                         tf.control_dependencies([tf.assign(num, 0)]),\
                         tf.control_dependencies([tf.assign_add(num, 1)]):
                        return self._read_image_with_attack(mode)
            return _func
        def handle(num):
            def _func():
                with tf.control_dependencies([tf.assign_add(num, 1)]):
                    return self._read_image_with_attack(mode)
            return _func
        # contention can occur. more examples might be produced than per_thread_num; i think this will not harm the performance
        with tf.device('/gpu:{}'.format(self.device)):
            return tf.cond(num>=per_thread_num, wait_and_handle(num), handle(num))

    def _read_image_with_attack(self, mode):
        # Construct the aug/load_generated graph
        img, auged_img, label, loaded_adv_imgs = self.read_image(self.filenames_queues_dct[mode], mode)
        # Construct the attack graph using the already constructed attacks
        with tf.device('/gpu:{}'.format(self.device)):
            advs = tf.concat([loaded_adv_imgs] + [a.generate_tensor(tf.expand_dims(auged_img, 0), tf.one_hot(tf.expand_dims(label, 0), self.num_labels)) for a in self.available_attacks], axis=0)
        return [img, auged_img, label, advs]

    def batch_q(self, mode):
        read_image_func = self.read_image_with_following if self.has_following \
                          else self.read_image_without_following
        return tf.train.batch_join([read_image_func(mode, i, major=i==0) for i in range(self.num_threads[mode])],
                                   self.batch_size, shapes=[tuple(self.image_shape), tuple(self.image_shape), (),
                                                            tuple([self.total_generated_adv_num] + list(self.image_shape))],
                                   capacity=self.capacity)

    def start(self, sess):
        self.sess = sess
        # construct copy_ops for follow models, and run these copy_ops for the first time
        # call load_checkpoint for checkpoint models;
        self.copy_ops_dct = {}
        for m_cfg in self.available_models_cfgs:
            ns = m_cfg["namescope"]
            model = AvailModels.get_model(ns, tag="gray_dataset")
            if m_cfg.get("follow", False):
                target_ns = m_cfg["follow"]
                target_model = AvailModels.get_model(target_ns)
                # Pairing vars of target model and the following model
                name_dct = {var.op.name.replace(target_ns + "/", ""): [var] for var in target_model.vars}
                [name_dct[var.op.name.replace(ns + "/", "")].append(var) for var in model.vars]
                assert all(len(pair) == 2 for pair in name_dct.itervalues())
                self.copy_ops_dct[ns] = tf.group(*[v.assign(t_v.value()) for t_v, v in name_dct.itervalues()])
            else: # load-checkpoint type
                model.load_checkpoint(m_cfg["checkpoint"], self.sess, m_cfg.get("load_namescope", None))

        # Run copy ops for the first time
        self.run_copy_ops()
        self.started = True

        # Start the queue runner!
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def run_copy_ops(self):
        print("Copying into models: ", self.copy_ops_dct.keys())
        self.sess.run(self.copy_ops_dct.values())

    def sync_epoch(self, epoch):
        # Called in the main thread, to perform a synchronization with queue-runner threads
        assert self.started, "Fault: GrayDataset not started"
        if epoch % self.sync_every == 0:
            if not self.FLAGS.no_init_test and epoch / self.sync_every == 1:
                self.sess.run(tf.assign(self.queues_dct["val"][-1], self.sync_every))
                utils.log("Adjust validation gen threshold back to ", self.sync_every)
            # Sync for follow models...
            # 1. wait for all queue runner threads to reach barrier
            # 2. run copy_ops
            # 3. tell all queue runner threads to continue, now with new vars!
            utils.log("Syncing following models at epoch {}".format(epoch))
            train_m2t, train_t2m = self.queues_dct["train"][:2]
            val_m2t, val_t2m = self.queues_dct["val"][:2]
            self.sess.run([train_t2m.dequeue(), val_t2m.dequeue()]) # 1
            self.run_copy_ops() # 2
            self.sess.run([train_m2t.enqueue(tf.constant(True)), val_m2t.enqueue(tf.constant(True))]) # 3
            utils.log("Finished syncing following models at epoch {}".format(epoch))
        return

    @property
    def data_tensors(self):
        if not self._gen:
            self._gen = True
            with tf.device("/cpu:0"):
                self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t = self.batch_q("train")
                self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v = self.batch_q("val")

            self.labels_t = tf.one_hot(self.labels_t, self.num_labels)
            self.labels_v = tf.one_hot(self.labels_v, self.num_labels)
        return (self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t), (self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v)

class GrayCifar10Dataset(GrayDataset, Cifar10Dataset):
    pass

class GrayTIDataset(GrayDataset, TinyImageNetDataset):
    pass
