import cv2
import numpy as np
import functools

available_methods = {}

def add_to_registry(name):
    def _dec(func):
        @functools.wraps(func)
        def _func(model, image, label):
            return func(model, image, label)
        available_methods[name] = _func
        return _func
    return _dec
            
def visualize(type_, model, image, label):
    return available_methods[type_](model, image, label)

@add_to_registry("gradcam")
def visualize_gradcam(model, image, label):
    gradcam = model.models[0].model.sess.run(model.models[0].model.gradcam_pic,
                                             feed_dict={
                                                 model.models[0].model._images: np.expand_dims(image, 0)
                                             })
    gradcam = np.expand_dims(np.squeeze(gradcam), 2)
    gradcam = cv2.cvtColor(gradcam, cv2.COLOR_GRAY2RGB)
    return gradcam.astype(np.float32)

@add_to_registry("gradient")
def visualize_gradient(model, image, label):
    grad = model.predictions_and_gradient(image, label)[1]
    grad_im = (np.clip((grad - np.mean(grad, axis=-1, keepdims=True)) / (3 * np.std(grad, axis=-1, keepdims=True)), -1, 1) + 1) / 2
    return (grad_im * 255).astype(np.float32)
