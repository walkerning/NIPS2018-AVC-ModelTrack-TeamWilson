# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from models import QCNN
import utils
from utils import AvailModels, LrAdjuster
from attacks import Attack, AttackGenerator
from base_trainer import settings, Trainer

class DistillTrainer(Trainer):
    class _settings(settings):
        default_cfg = {
            "model": None,
            "test_frequency": 1,
            "aug_mode": "pre",
            "test_frequency": 1,
            "aug_mode": "pre",

            # Data gen
            "num_threads": 2,

            # Training
            "distill_use_auged": False, # 一个谜一样的bug
            "epochs": 50,
            "batch_size": 100,
            "adjust_lr_acc": None,

            "alpha": 0.1,
            "beta": 0,
            "theta": 0.5,
            "temperature": 1,
            "at_mode": "attention",
            "train_models": {},

            # Testing
            "test_saltpepper": None,
            "test_models": {},

            # Augmentaion
            "aug_saltpepper": None,
            "aug_gaussian": None,
            
            # Adversarial Augmentation
            "available_attacks": [],
            "generated_adv": [],
            "train_merge_adv": False,

            "additional_models": []
        }
    def __init__(self, args, cfg):
        super(DistillTrainer, self).__init__(args, cfg)

    def init(self):
        batch_size = self.FLAGS.batch_size # default to 128

        (self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t), (self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v) = self.dataset.data_tensors
        utils.log("Train number: {}; Validation number: {}".format(self.dataset.train_num, self.dataset.val_num))

        self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x")
        self.stu_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="stu_x")
        self.labels = tf.placeholder(tf.float32, [None, 200], name="labels")

        model_tea = QCNN.create_model(self.FLAGS["teacher"])
        self.logits = model_tea.get_logits(self.x) # FIXME: here should also support input of multiple batch size, as there will be multi pre-generated adv data in the future. (used in test)
        restore_vars = tf.global_variables()
        tea_t_vars = tf.trainable_variables()
        model_stu = QCNN.create_model(self.FLAGS["model"])
        self.logits_stu = model_stu.get_logits(self.stu_x)
        self.save_vars = tf.global_variables()
        AvailModels.add(model_stu, self.stu_x, self.logits_stu)
        stu_t_vars = tf.trainable_variables()
        for var_ in restore_vars:
            self.save_vars.remove(var_)
        for var_ in tea_t_vars:
            stu_t_vars.remove(var_)

        self.saver = tf.train.Saver(self.save_vars, max_to_keep=20)
        self.saver_res = tf.train.Saver(restore_vars, max_to_keep=20)

        tf.get_default_graph().clear_collection("trainable_variables")
        for var in stu_t_vars:
            tf.add_to_collection("trainable_variables", var)

        self.training_stu = model_stu.get_training_status()
        
        # Loss and metrics
        soft_label = tf.nn.softmax(self.logits/self.FLAGS.temperature)
        soft_logits = self.logits_stu/self.FLAGS.temperature
        tile_num = tf.shape(soft_logits)[0]/batch_size
        reshape_soft_label = tf.reshape(tf.tile(tf.expand_dims(soft_label, 1), [1, tile_num, 1]), [-1, 200])
        reshape_labels = tf.reshape(tf.tile(tf.expand_dims(self.labels, 1), [1, tile_num, 1]), [-1, 200])
        ce = tf.nn.softmax_cross_entropy_with_logits(\
                labels=reshape_soft_label,\
                 logits=soft_logits)
        self.distillation = tf.reduce_mean(ce)
        origin_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=reshape_labels, logits=self.logits_stu))

        self.loss = origin_loss * self.FLAGS.theta
        if self.FLAGS.alpha != 0:
            self.loss += self.distillation * self.FLAGS.alpha
        if self.FLAGS.beta != 0:
            self.at_loss = get_at_loss(group_list_teacher, group_list_student)
            self.loss += at_loss * self.FLAGS.beta
        else:
            self.at_loss = tf.constant(0.0)
        self.index_label = tf.argmax(self.labels, -1)
        self.reshape_index_label = tf.argmax(reshape_labels, -1)
        correct = tf.equal(tf.argmax(self.logits_stu, -1), self.reshape_index_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tea_correct = tf.equal(tf.argmax(self.logits, -1), self.index_label)
        self.tea_accuracy = tf.reduce_mean(tf.cast(tea_correct, tf.float32))

        # Initialize the optimizer
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.lr_adjuster = LrAdjuster.create_adjuster(self.FLAGS.adjust_lr_acc)
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.FLAGS.model["namescope"])
        with tf.control_dependencies(update_ops):
            self.grads_and_var = optimizer.compute_gradients(self.loss)
            self.train_step = optimizer.apply_gradients(self.grads_and_var)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        [Attack.create_attack(self.sess, a_cfg) for a_cfg in self.FLAGS["available_attacks"]]
        self.train_attack_gen = AttackGenerator(self.FLAGS["train_models"], merge=self.FLAGS.train_merge_adv)
        self.test_attack_gen = AttackGenerator(self.FLAGS["test_models"])

    def train(self):
        sess = self.sess
        steps_per_epoch = self.dataset.train_num // self.FLAGS.batch_size
        for epoch in range(1, self.FLAGS.epochs+1):
            self.train_attack_gen.new_epoch()
            start_time = time.time()
            info_v_epoch = np.zeros((self.FLAGS.update_per_batch, 5))
            now_lr = self.lr_adjuster.get_lr()
            if now_lr is None:
                utils.log("End training as val acc not decay!!!")
                return
            else:
                utils.log("Lr: ", now_lr)

            # Train batches
            gen_time = 0
            run_time = 0
            for step in range(1, steps_per_epoch+1):
                self.train_attack_gen.new_batch()
                x_v, auged_x_v, y_v, adv_x_v = sess.run([self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t])
                gen_start_time = time.time()
                _, adv_xs = self.train_attack_gen.generate_for_model(auged_x_v, y_v, self.FLAGS.model["namescope"], adv_x_v)
                gen_time += time.time() - gen_start_time
                inner_info_v = []
                run_start_time = time.time()
                for adv_x in adv_xs:
                    feed_dict = {
                        self.x: x_v if not self.FLAGS.distill_use_auged else auged_x_v,
                        self.stu_x: adv_x,
                        self.training_stu: True,
                        self.labels: y_v,
                        self.learning_rate: now_lr
                    }
                    info_v, _ = sess.run([[self.loss, self.distillation, self.at_loss, self.accuracy, self.tea_accuracy], self.train_step], feed_dict=feed_dict)
                    inner_info_v.append(info_v)
                run_time += time.time() - run_start_time
                info_v_epoch += inner_info_v
                if step % self.FLAGS.print_every == 0:
                    print("\rEpoch {}: steps {}/{} loss: {}/{}/{}".format(epoch, step, steps_per_epoch, *np.mean(inner_info_v, axis=0)[:3]), end="")
            gen_time = gen_time / steps_per_epoch
            run_time = run_time / steps_per_epoch
            info_v_epoch /= steps_per_epoch
            duration = time.time() - start_time
            sec_per_batch = duration / (steps_per_epoch * self.FLAGS.batch_size)
            loss_v_epoch, _, _, acc_stu_epoch, acc_tea_epoch = np.mean(inner_info_v, axis=0)
            utils.log("\r{}: Epoch {}; (average) loss: {:.3f}; (average) student accuracy: {:.2f} %; (average) teacher accuracy: {:.2f} %. {:.3f} sec/batch; gen time: {:.3f} sec/batch; run time: {:.3f} sec/batch; {}"
                      .format(datetime.now(), epoch, loss_v_epoch, acc_stu_epoch * 100, acc_tea_epoch * 100, sec_per_batch, gen_time, run_time, "" if not utils.PROFILING else "; ".join(["{}: {:.2f} ({:.3f} average) sec".format(k, t, t/num) for k, (num, t) in utils.all_profiled.iteritems()])), flush=True)
                  
            # End training batches

            # Test on the validation set
            if epoch % self.FLAGS.test_frequency == 0:
                self.test(adv=True, name="normal_adv")
                self.lr_adjuster.add()
                if self.FLAGS.train_dir:
                    if epoch % self.FLAGS.save_every == 0:
                        save_path = os.path.join(self.FLAGS.train_dir, str(epoch))
                        self.saver.save(sess, save_path)
                        utils.log("Saved student model to: ", save_path)

    def test(self, saltpepper=None, adv=False, name=""):
        sess = self.sess
        steps_per_epoch = self.dataset.val_num // self.FLAGS.batch_size
        loss_v_epoch = 0
        acc_v_epoch = 0
        tea_acc_v_epoch = 0
        image_disturb = 0
        test_res = {}
        for step in range(1, steps_per_epoch+1):
            self.test_attack_gen.new_batch()
            x_v, auged_x_v, y_v, adv_x_v = sess.run([self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v])
            print("\rTesting {}/{}".format(step, steps_per_epoch), end="")
            if saltpepper is not None: # during test, saltpepper is added at last, this is a train-test discrepancy, but i don't think it matters
                img = x_v
                u = np.random.uniform(size=list(x_v.shape[:3]) + [1])
                salt = (u >= 1 - saltpepper/2).astype(x_v.dtype) * 256
                pepper = - (u < saltpepper/2).astype(x_v.dtype) * 256
                img = np.clip(img + salt + pepper, 0, 255)
                auged_x = img
            else:
                auged_x = x_v
            acc_v, tea_acc_v, loss_v = sess.run([self.accuracy, self.tea_accuracy, self.loss], feed_dict={
                self.x: auged_x,
                self.stu_x: auged_x,
                self.labels: y_v,
                self.training_stu: False
            })
            image_disturb += np.abs(auged_x - x_v).mean()
            loss_v_epoch += loss_v
            acc_v_epoch += acc_v
            tea_acc_v_epoch += tea_acc_v
            # test adv
            if adv:
                test_ids, adv_xs = self.test_attack_gen.generate_for_model(auged_x_v, y_v, "stu_", adv_x_v)
                for test_id, adv_x in zip(test_ids, adv_xs):
                    acc_v, tea_acc_v, loss_v = sess.run([self.accuracy, self.tea_accuracy, self.loss], feed_dict={
                        self.stu_x: adv_x,
                        self.x: adv_x,
                        self.labels: y_v,
                        self.training_stu: False
                    })
                    if test_id not in test_res:
                        test_res[test_id] = np.zeros(4)
                    test_res[test_id] += [acc_v, tea_acc_v, loss_v, np.mean(np.abs(adv_x - auged_x_v))]
        image_disturb /= steps_per_epoch
        loss_v_epoch /= steps_per_epoch
        acc_v_epoch /= steps_per_epoch
        tea_acc_v_epoch /= steps_per_epoch
        print("\r", end="")
        utils.log("\tTest {}: \n\t\tloss: {}; accuracy: {:.2f} %; teacher accuracy: {:.2f} %; Mean pixel distance: {:.2f}".format(name, loss_v_epoch, acc_v_epoch * 100, tea_acc_v_epoch * 100, image_disturb))
        if adv:
            utils.log("\tAdv:\n\t\t{}".format("\n\t\t".join(["test {}: acc: {:.2f}; tea_acc: {:.2f}; ce_loss: {:.2f}; dist: {:.2f}".format(test_id, *(attack_res/steps_per_epoch)) for test_id, attack_res in test_res.items()])), flush=True)

    def start(self):
        sess = self.sess
        if self.FLAGS.train_dir:
            train_writer = tf.summary.FileWriter(self.FLAGS.train_dir + '/train',
                                                 sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)    

        if self.FLAGS.test_only:
            # FIXME: all checkpoint loading staffs should be moved into methods of model 
            if self.FLAGS.load_file_tea and not self.FLAGS.load_file_stu:
                self.saver_res.restore(sess, self.FLAGS.load_file_tea)
                # if self.FLAGS.model.more_blocks:
                #     remove_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                #                                     self.FLAGS.model["namescope"]+"/more_blocks" if self.FLAGS.model["namescope"] else "more_blocks")
                #     for var_ in remove_vars:
                #         if var_ in self.save_vars:
                #             self.save_vars.remove(var_)
                var_mapping_dct = {var.op.name.replace(self.FLAGS.model["namescope"] + "/", ""): var for var in self.save_vars}
                load_stu = tf.train.Saver(var_mapping_dct)
                load_stu.restore(sess, self.FLAGS.load_file_tea)
            else:
                self.saver.restore(sess, self.FLAGS.load_file_stu)
                if self.FLAGS.load_file_tea:
                    self.saver_res.restore(sess, self.FLAGS.load_file_tea)
            self.test(adv=True)
            if FLAGS.test_saltpepper is not None:
                if isinstance(FLAGS.test_saltpepper, (tuple, list)):
                    for sp in FLAGS.test_saltpepper:
                        self.test(saltpepper=sp, adv=False, name="saltpepper_{}".format(sp))
                else:
                    self.test(saltpepper=FLAGS.test_saltpepper, adv=False, name="saltpepper_{}".format(FLAGS.test_saltpepper))
            coord.request_stop()
            coord.join(threads)
            sys.exit(0)

        if self.FLAGS.load_file_tea and not self.FLAGS.load_file_stu:
            self.saver_res.restore(sess, self.FLAGS.load_file_tea)
            # if self.FLAGS.more_blocks:
            #     remove_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
            #                                     self.FLAGS.model["namescope"]+"/more_blocks" if self.FLAGS.model["namescope"] else "more_blocks")
            #     for var_ in remove_vars:
            #         if var_ in self.save_vars:
            #             self.save_vars.remove(var_)
            var_mapping_dct = {var.op.name.replace(self.FLAGS.model["namescope"] + "/",""): var for var in self.save_vars}
                                                       
            load_stu = tf.train.Saver(var_mapping_dct)
            load_stu.restore(sess, self.FLAGS.load_file_tea)
            if not self.FLAGS.no_init_test:
                self.test(adv=True, name="loaded_teacher_copy")
            utils.log("Start training...")
            self.train()
        elif self.FLAGS.load_file_stu:
            self.saver.restore(sess, self.FLAGS.load_file_stu)
            if self.FLAGS.load_file_tea:
                self.saver_res.restore(sess, self.FLAGS.load_file_tea)
            if not self.FLAGS.no_init_test:
                self.test(adv=True, name="loaded_normal_adv")
            utils.log("Start training...")
            self.train()
        else:
            print("error: no input file.")
        coord.request_stop()
        coord.join(threads)

    @classmethod
    def populate_arguments(cls, parser):
        parser.add_argument("--load-file-stu", type=str, default="",
                            help="Load student model")
        parser.add_argument("--load-file-tea", type=str, default="",
                            help="Load teacher model")
