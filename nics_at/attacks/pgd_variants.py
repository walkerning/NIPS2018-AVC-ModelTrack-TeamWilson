import cleverhans

class MadryEtAl_L2(cleverhans.attacks.MadryEtAl):
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
        logits = self.model.get_logits(adv_x)
        loss = attack_softmax_cross_entropy(y, logits)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, adv_x)
        axis = list(range(1, len(grad.shape)))
        scaled_signed_grad = self.eps_iter * grad / tf.sqrt(tf.reduce_mean(tf.square(grad), axis, keep_dims=True))
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
        logits = self.model.get_logits(adv_x)
        transfer_logits = self.transfer_model.get_logits(adv_x)
        transfer_loss = attack_softmax_cross_entropy(y, transfer_logits)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(transfer_loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta
