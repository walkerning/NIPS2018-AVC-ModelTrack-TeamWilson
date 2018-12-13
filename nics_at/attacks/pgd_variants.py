import cleverhans
from cleverhans.model import Model, CallableModelWrapper

class MadryEtAl_KLloss(cleverhans.attacks.MadryEtAl):
    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(MadryEtAl_KLloss, self).__init__(model, back, sess, dtypestr)
        self.feedable_kwargs = {'eps': self.np_dtype,
                                'clip_min': self.np_dtype,
                                'clip_max': self.np_dtype}
        self.structural_kwargs = ["loss_type", "nb_iter"]

    def parse_params(self, **kwargs):
        super(MadryEtAl_KLloss, self).parse_params(**kwargs)
        self.loss_type = kwargs.get("loss_type", "multinomial")
        assert self.loss_type in {"gaussian", "multinomial"}
        print("kl loss type: ", self.loss_type)
        return True

    def KL(self, p, q):
        import tensorflow as tf
        if self.loss_type == "gaussian":
            # the output distribution is treated as multi-dimensional gaussian with same std
            return tf.reduce_sum((p - q) ** 2, axis=-1)
        else: # KL of multinomial
            # the output distribution is treated as a single multinomial dist
            return tf.reduce_sum(p * (tf.log(p + 1e-10) - tf.log(q + 1e-10)), axis=-1)

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        import tensorflow as tf

        eta = tf.random_normal(tf.shape(x), 0, 1, dtype=self.tf_dtype)
        eta = eta / tf.norm(eta, ord=2) * self.eps

        x_p = tf.stop_gradient(tf.nn.softmax(self.model.get_logits(x)))
        for i in range(self.nb_iter):
            eta = self.attack_single_step(x, eta, x_p) # do not need y
        adv_x = x + eta
        return adv_x

    def attack_single_step(self, x, eta, x_p):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf

        adv_x = x + eta
        logits = self.model.get_logits(adv_x)
        loss = self.KL(x_p, tf.nn.softmax(logits))
        grad, = tf.gradients(loss, adv_x)
        eta = grad / tf.norm(grad, ord=2) * self.eps
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(x + eta, self.clip_min, self.clip_max)
            eta = adv_x - x
        return eta

class MadryEtAl_L2(cleverhans.attacks.MadryEtAl):
    def __init__(self, model, transfer=None, back="tf", sess=None, dtypestr="float32"):
        super(MadryEtAl_L2, self).__init__(model, back=back, sess=sess, dtypestr=dtypestr)
        self.transfer_model = transfer

    # ord only decide use linf or l2 to clip
    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta
        from cleverhans.loss import attack_softmax_cross_entropy

        adv_x = x + eta
        model = self.model if self.transfer_model is not None else self.model
        logits = model.get_logits(adv_x)
        loss = attack_softmax_cross_entropy(y, logits)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        axis = list(range(1, len(grad.shape)))
        avoid_zero_div = 1e-12
        # scaled_signed_grad = self.eps_iter * grad / tf.sqrt(tf.maximum(avoid_zero_div,
        #                                                                tf.reduce_mean(tf.square(grad), axis, keep_dims=True)))
        scaled_signed_grad = self.eps_iter * grad / tf.sqrt(tf.maximum(avoid_zero_div,
                                                                       tf.reduce_sum(tf.square(grad), axis, keep_dims=True)))
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps) # by default: use linf to clip
        return eta

class MadryEtAl_transfer(cleverhans.attacks.MadryEtAl):
    def __init__(self, model, transfer, back="tf", sess=None, dtypestr="float32"):
        super(MadryEtAl_transfer, self).__init__(model, back=back, sess=sess, dtypestr=dtypestr)
        self.transfer_model = transfer

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta
        from cleverhans.loss import attack_softmax_cross_entropy

        adv_x = x + eta
        transfer_logits = self.transfer_model.get_logits(adv_x)
        transfer_loss = attack_softmax_cross_entropy(y, transfer_logits)
        if self.targeted:
            transfer_loss = -transfer_loss
        grad, = tf.gradients(transfer_loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        adv_x = tf.stop_gradient(adv_x)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

class MadryEtAl_transfer_re(MadryEtAl_transfer): # transfer and return early
    def __init__(self, model, transfer, back="tf", sess=None, dtypestr="float32"):
        super(MadryEtAl_transfer_re, self).__init__(model, transfer=transfer, back=back, sess=sess, dtypestr=dtypestr)
        self.structural_kwargs = ["ord", "nb_iter", "rand_init", "min_nb_iter"]

    def parse_params(self, *args, **kwargs):
        self.min_nb_iter = kwargs.get("min_nb_iter", 0)
        return super(MadryEtAl_transfer_re, self).parse_params(*args, **kwargs)

    def attack(self, x, y):
        import tensorflow as tf
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps,
                                    dtype=self.tf_dtype)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)


        y_label = tf.argmax(y, axis=-1)
        predict = tf.argmax(self.model.get_logits(x), axis=-1)

        # 1. Using tf.while_loop
        # while_condition = lambda eta_, predict_, iter_: tf.logical_and(iter_<self.nb_iter, tf.squeeze(tf.equal(predict_, y_label)))
        # def body(eta_, _, iter_):
        #     eta = self.attack_single_step(x, eta_, y)
        #     predict = tf.argmax(self.model.get_logits(x + eta), axis=-1)
        #     return [eta, predict, tf.add(iter_, 1)]
        # eta, _, _ = tf.while_loop(while_condition, body, [eta, predict, 0])

        # 2. Construct for nb_iter runs using tf.cond every iter; consistent with the style of other cleverhans attacks
        # def next_step_eta(eta_):
        #     def _func():
        #         eta = self.attack_single_step(x, eta_, y)
        #         predict = tf.argmax(self.model.get_logits(x + eta), axis=-1)
        #         return [eta, predict]
        #     return _func
        # def return_straight(eta, predict):
        #     def _func():
        #         return [eta, predict]
        #     return _func
        # for iter_ in range(self.nb_iter):
        #     still_correct = tf.squeeze(tf.equal(predict, y_label))
        #     eta, predict = tf.cond(still_correct, next_step_eta(eta), return_straight(eta, predict))

        # 3. Support batch_size > 1; it seems there are redundant calculation, but it will be actually more efficient as it can utilize the parallel computing power
        def next_step_eta(eta_):
            eta = self.attack_single_step(x, eta_, y)
            predict = tf.argmax(self.model.get_logits(x + eta), axis=-1)
            return [eta, predict]

        if self.min_nb_iter > 0:
            for iter_ in range(self.min_nb_iter):
                eta = self.attack_single_step(x, eta, y)

        for iter_ in range(self.min_nb_iter, self.nb_iter):
            still_correct = tf.equal(predict, y_label)
            new_eta, new_predict = next_step_eta(eta)
            eta = tf.where(still_correct, new_eta, eta)
            predict = tf.where(still_correct, new_predict, predict)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x

class MadryEtAl_L2_transfer_re(MadryEtAl_transfer_re, MadryEtAl_L2):
    pass
