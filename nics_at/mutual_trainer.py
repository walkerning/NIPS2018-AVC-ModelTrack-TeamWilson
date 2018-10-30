# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import time
from datetime import datetime
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from models import QCNN
import utils
from utils import AvailModels, LrAdjuster
from attacks import Attack, AttackGenerator
from base_trainer import settings, Trainer

class MutualTrainer(Trainer):
    class _settings(settings):
        default_cfg = {
            "models": [],
            "test_frequency": 1,
            "aug_mode": "pre",

            # Training
            "num_threads": 2,
            "more_augs": False,
            "use_imgnet1k": False,

            "distill_use_auged": False,
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
            "train_models": {},
            "train_merge_adv": False,
            "random_interp": None,

            "additional_models": []
        }

    def __init__(self, args, cfg):
        super(MutualTrainer, self).__init__(args, cfg)

    def init(self):
        self.mutual_num = len(self.FLAGS["models"])
        batch_size = self.FLAGS.batch_size

        (self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t), (self.imgs_v, self.auged_imgs_v, self.labels_v, self.adv_imgs_v) = self.dataset.data_tensors

        self.labels = tf.placeholder(tf.float32, [None, 200], name="labels")
        model_lst = [QCNN.create_model(m_cfg) for m_cfg in self.FLAGS["models"]]
        input_holder_lst = []
        saver_lst = []
        ce_loss_lst = []
        model_vars_lst = []
        add_model_vars_lst = []
        logits_lst = []
        prob_lst = []
        prob_placeholder_lst = []
        training_lst = []
        accuracy_lst = []
        namescope_lst = [m_cfg["namescope"] for m_cfg in self.FLAGS["models"]]
        for i in range(self.mutual_num):
            x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x_{}".format(i))
            prob_ph = tf.placeholder(tf.float32, shape=[None, 200], name="prob_placeholder_{}".format(i))
            name_scope = namescope_lst[i]
            model = model_lst[i]
            training = model.get_training_status()
            logits = model.get_logits(x)
            prob = tf.nn.softmax(logits)
            reshape_labels = tf.reshape(tf.tile(tf.expand_dims(self.labels, 1), [1, tf.shape(logits)[0] / batch_size, 1]), [-1, 200])
            ce_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=reshape_labels, logits=logits))

            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name_scope)
            saver = tf.train.Saver(model_vars, max_to_keep=20)
            index_label = tf.argmax(reshape_labels, axis=-1)
            correct = tf.equal(tf.argmax(logits, -1), index_label)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            AvailModels.add(model, x, logits)
            input_holder_lst.append(x)
            # model_lst.append(model)
            training_lst.append(training)
            logits_lst.append(logits)
            prob_lst.append(prob)
            prob_placeholder_lst.append(prob_ph)
            ce_loss_lst.append(ce_loss)
            model_vars_lst.append(model_vars)
            saver_lst.append(saver)
            accuracy_lst.append(accuracy)

        # additional test only models
        for i in range(len(self.FLAGS["additional_models"])):
            m_cfg = self.FLAGS["additional_models"][i]
            x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="x_addi_{}".format(i))
            model = QCNN.create_model(m_cfg)
            logits = model.get_logits(x)
            AvailModels.add(model, x, logits)
            add_model_vars_lst.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, m_cfg["namescope"]))

        loss_lst = []
        kl_loss_lst = []
        train_step_lst = []
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.lr_adjuster = LrAdjuster.create_adjuster(self.FLAGS.adjust_lr_acc)
        for i in range(self.mutual_num):
            name_scope = namescope_lst[i]
            prob = prob_lst[i]
            reshape_prob_placeholders = [tf.reshape(tf.tile(tf.expand_dims(prob_placeholder_lst[j], 1), [1, tf.shape(prob)[0] / batch_size, 1]), [-1, 200]) for j in range(self.mutual_num) if j != i]
            kl_losses = [tf.reduce_mean(tf.reduce_sum(rpph * (tf.log(rpph+1e-10) - tf.log(prob+1e-10)), axis=-1)) for rpph in reshape_prob_placeholders]
            kl_loss = tf.reduce_mean(kl_losses)
            loss = self.FLAGS.theta * ce_loss_lst[i] + self.FLAGS.alpha * kl_loss
            kl_loss_lst.append(kl_loss)
            loss_lst.append(loss)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
            with tf.control_dependencies(update_ops):
                grads_and_var = optimizer.compute_gradients(loss, var_list=model_vars_lst[i])
                train_step = optimizer.apply_gradients(grads_and_var)
                train_step_lst.append(train_step)

        self.input_holder_lst = tuple(input_holder_lst)
        self.model_lst = tuple(model_lst)
        self.saver_lst = tuple(saver_lst)
        self.ce_loss_lst = tuple(ce_loss_lst)
        self.model_vars_lst = tuple(model_vars_lst)
        self.add_model_vars_lst = tuple(add_model_vars_lst)
        self.logits_lst = tuple(logits_lst)
        self.prob_lst = tuple(prob_lst)
        self.prob_placeholder_lst = tuple(prob_placeholder_lst)
        self.training_lst = tuple(training_lst)
        self.accuracy_lst = tuple(accuracy_lst)
        self.kl_loss_lst = tuple(kl_loss_lst)
        self.loss_lst = tuple(loss_lst)
        self.train_step_lst = tuple(train_step_lst)
        self.namescope_lst = namescope_lst

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        [Attack.create_attack(self.sess, a_cfg) for a_cfg in self.FLAGS["available_attacks"]]
        self.train_attack_gen = AttackGenerator(self.FLAGS["train_models"], merge=self.FLAGS.train_merge_adv, random_interp=self.FLAGS.random_interp, name="train")
        self.test_attack_gen = AttackGenerator(self.FLAGS["test_models"], name="test")

    def test(self, saltpepper=None, adv=False, name=""):
        sess = self.sess
        steps_per_epoch = self.dataset.val_num // self.FLAGS.batch_size
        loss_lst_v_test = np.zeros(self.mutual_num)
        acc_lst_v_test = np.zeros(self.mutual_num)
        image_disturb = np.zeros(self.mutual_num)
        test_res = [OrderedDict() for _ in range(self.mutual_num)]
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
                auged_x = [img] * self.mutual_num
            else:
                auged_x = [x_v] * self.mutual_num
            acc_lst_v, ce_loss_lst_v = sess.run([self.accuracy_lst, self.ce_loss_lst], feed_dict={
                self.input_holder_lst: auged_x,
                self.labels: y_v,
                self.training_lst: [False] * self.mutual_num
            })
            image_disturb += [np.abs(y - x_v).mean() for y in auged_x]
            loss_lst_v_test += ce_loss_lst_v
            acc_lst_v_test += acc_lst_v
            # test adv
            if adv:
                for mi in range(self.mutual_num):
                    test_ids, adv_xs = self.test_attack_gen.generate_for_model(auged_x_v, y_v, self.namescope_lst[mi], adv_x_v)
                    for test_id, adv_x in zip(test_ids, adv_xs):
                        acc_v, ce_loss_v = sess.run([self.accuracy_lst[mi], self.ce_loss_lst[mi]], feed_dict={
                            self.input_holder_lst[mi]: adv_x,
                            self.labels: y_v,
                            self.training_lst[mi]: False
                        })
                        if test_id not in test_res[mi]:
                            test_res[mi][test_id] = np.zeros(3)
                        if adv_x.shape != auged_x_v.shape:
                            sp = [auged_x_v.shape[0], adv_x.shape[0] / auged_x_v.shape[0]] + list(auged_x_v.shape[1:])
                            tmp_adv_x = adv_x.reshape(sp)
                            sp[1] = 1
                            mean_dist = np.mean(np.abs(tmp_adv_x - auged_x_v.reshape(sp)))
                        else:
                            mean_dist = np.mean(np.abs(adv_x - auged_x_v)) # L1 dist
                        test_res[mi][test_id] += [acc_v, ce_loss_v, mean_dist]
        image_disturb /= steps_per_epoch
        loss_lst_v_test /= steps_per_epoch
        acc_lst_v_test /= steps_per_epoch
        print("\r", end="")
        utils.log("\tTest {}: \n\t\t{}".format(
            name, "\n\t\t".join(["ce loss: {}; accuracy: {:.2f} %; Mean pixel distance: {:2f}".format(l, a, d) for l, a, d in zip(loss_lst_v_test, acc_lst_v_test*100, image_disturb)])))
        # utils.log("\tTest {}:".format(name))
        if adv:
            utils.log("\tAdv:")
            for mn, model_res in zip(self.namescope_lst, test_res):
                utils.log("\t\tModel {}:\n\t\t\t{}".format(mn, "\n\t\t\t".join(["test {}: acc: {:.3f}; ce_loss: {:.2f}; dist: {:.2f}".format(test_id, *(attack_res/steps_per_epoch)) for test_id, attack_res in model_res.items()])), flush=True)
        return list(acc_lst_v_test) + sum([[v[0]/steps_per_epoch for v in mr.values()] for mr in test_res], [])

    def train(self):
        sess = self.sess
        steps_per_epoch = self.dataset.train_num // self.FLAGS.batch_size
        for epoch in range(1, self.FLAGS.epochs+1):
            self.train_attack_gen.new_epoch()
            start_time = time.time()
            info_v_epoch = np.zeros((self.mutual_num, self.FLAGS.update_per_batch, 4))

            now_lr = self.lr_adjuster.get_lr()
            if now_lr is None:
                utils.log("End training as val acc not decay!!!")
                return
            else:
                utils.log("Lr: ", now_lr)

            # Train batches
            for step in range(1, steps_per_epoch+1):
                self.train_attack_gen.new_batch()
                x_v, auged_x_v, y_v, adv_x_v = sess.run([self.imgs_t, self.auged_imgs_t, self.labels_t, self.adv_imgs_t])
                # forward using normal
                normal_prob_lst_v = sess.run(self.prob_lst, feed_dict={
                    self.input_holder_lst: [x_v if not self.FLAGS.distill_use_auged else auged_x_v] * self.mutual_num,
                    self.training_lst: [True] * self.mutual_num
                })
                info_lst_v = []
                for mi in range(self.mutual_num):
                    _, adv_xs = self.train_attack_gen.generate_for_model(auged_x_v, y_v, self.namescope_lst[mi], adv_x_v)
                    if len(adv_xs) == 0: # no adv is generated
                        adv_xs.append(normal_x)
                    if len(adv_xs) < self.FLAGS.update_per_batch:
                        print("warning: update_per_batch is set to {}; only generated {} examples".format(self.FLAGS.update_per_batch, len(adv_xs)))
                        adv_xs += [normal_x] * (self.FLAGS.update_per_batch - len(adv_xs))
                    inner_info_v = []
                    for adv_x in adv_xs:
                        feed_dict = {
                            self.input_holder_lst[mi]: adv_x,
                            self.prob_placeholder_lst: normal_prob_lst_v,
                            self.labels: y_v,
                            self.training_lst[mi]: True,
                            self.learning_rate: now_lr
                        }
                        info_v, _ = sess.run([[self.ce_loss_lst[mi], self.kl_loss_lst[mi], self.loss_lst[mi], self.accuracy_lst[mi]], self.train_step_lst[mi]], feed_dict=feed_dict)
                        inner_info_v.append(info_v) # append each adv example info
                    info_lst_v.append(inner_info_v) # append each model info
                info_v_epoch += info_lst_v
                if step % self.FLAGS.print_every == 0:
                    print("\rEpoch {}: steps {}/{}".format(epoch, step, steps_per_epoch), end="")

            info_v_epoch /= steps_per_epoch
            duration = time.time() - start_time
            sec_per_batch = duration / steps_per_epoch
            utils.log("{}: Epoch {}; {:.3f} sec/batch; {}:\n\t{}"
                  .format(datetime.now(), epoch, sec_per_batch, "" if not utils.PROFILING else "; ".join(["{}: {:.2f} ({:.3f} average) sec".format(k, t, t/num) for k, (num, t) in utils.all_profiled.iteritems()]), "\n\t".join(["Model {}:\n\t\t{}".format(mn, "\n\t\t".join(["ce_loss: {:.3f}; kl_loss: {:.3f}; loss: {:.3f}; acc: {:.3f}".format(*info_every_attack) for info_every_attack in info])) for mn, info in zip(self.namescope_lst, info_v_epoch)])), flush=True)
            # End training batches

            # Test on the validation set
            if epoch % self.FLAGS.test_frequency == 0:
                test_accs = self.test(adv=True, name="normal_adv")
                self.lr_adjuster.add_multiple_acc(test_accs)
                if self.FLAGS.train_dir:
                    if epoch % self.FLAGS.save_every == 0:
                        save_path = os.path.join(self.FLAGS.train_dir, str(epoch))
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        for i, saver in enumerate(self.saver_lst):
                            saver.save(sess, os.path.join(save_path, "model_{}".format(i)))
                        utils.log("Saved multiple model to: ", save_path)

    def start(self):
        if self.FLAGS.load_file:
            utils.log("Will load models from: {}".format(self.FLAGS.load_file))
        sess = self.sess
        if self.FLAGS.train_dir:
            train_writer = tf.summary.FileWriter(self.FLAGS.train_dir + '/train',
                                                 sess.graph)
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
        if not self.FLAGS.load_namescope:
            load_namescopes = [""] * self.mutual_num
        elif len(self.FLAGS.load_namescope) == 1:
            load_namescopes = self.FLAGS.load_namescope * self.mutual_num
        else:
            load_namescopes = self.FLAGS.load_namescope
        assert len(load_namescopes) == self.mutual_num

        if len(self.FLAGS.load_file) == 1:
            load_files = self.FLAGS.load_file * self.mutual_num
        else:
            load_files = self.FLAGS.load_file
        load_files += [m_cfg["checkpoint"] for m_cfg in self.FLAGS["additional_models"]]
        load_namescopes += [m_cfg["load_namescope"] for m_cfg in self.FLAGS["additional_models"]]
        add_namescope_lst = [m_cfg["namescope"] for m_cfg in self.FLAGS["additional_models"]]
        if load_files:
            assert len(load_files) == self.mutual_num + len(self.FLAGS.additional_models)
            for m, l_namescope, l_file in zip(self.model_lst, load_namescopes, load_files):
                m.load_checkpoint(l_file, self.sess, l_namescope)
            # for model_vars, v_namescope, l_namescope, l_file in zip(self.model_vars_lst + self.add_model_vars_lst, self.namescope_lst + add_namescope_lst, load_namescopes, load_files):
            #     var_mapping_dct = {var.op.name.replace(v_namescope + "/", l_namescope + ("/" if l_namescope else "")): var for var in model_vars}
            #     # from tensorflow.python import pywrap_tensorflow
            #     # reader = pywrap_tensorflow.NewCheckpointReader(l_file)
            #     # var_keys = reader.get_variable_to_shape_map().keys()
            #     saver = tf.train.Saver(var_mapping_dct)
            #     saver.restore(sess, l_file)
        if self.FLAGS.test_only:
            if self.FLAGS.test_saltpepper is not None:
                if isinstance(self.FLAGS.test_saltpepper, (tuple, list)):
                    for sp in self.FLAGS.test_saltpepper:
                        self.test(saltpepper=sp, name="saltpepper_{}".format(sp))
                else:
                    self.test(saltpepper=self.FLAGS.test_saltpepper, name="saltpepper_{}".format(self.FLAGS.test_saltpepper))
            self.test(adv=True, name="test_normal_adv")

            coord.request_stop()
            coord.join(threads)
            return

        if not self.FLAGS.no_init_test:
            if self.FLAGS.test_saltpepper is not None:
                if isinstance(self.FLAGS.test_saltpepper, (tuple, list)):
                    for sp in self.FLAGS.test_saltpepper:
                        self.test(saltpepper=sp, name="loaded saltpepper_{}".format(sp))
                else:
                    self.test(saltpepper=self.FLAGS.test_saltpepper, name="loaded saltpepper_{}".format(self.FLAGS.test_saltpepper))
            self.test(adv=True, name="test_normal_adv")
        utils.log("Start training...")
        self.train()

        coord.request_stop()
        coord.join(threads)
        
    @classmethod
    def populate_arguments(cls, parser):
        parser.add_argument("--load-file", type=str, default=[], action="append",
                    help="Load  model")
        parser.add_argument("--load-namescope", type=str, default=[], action="append", help="The namescope in the to-load checkpoint.")
