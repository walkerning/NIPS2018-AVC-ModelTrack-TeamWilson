import numpy as np
import foolbox
from utils import substitute_argscope
from functools import wraps

__all__ = [
    "cw_l2_transfer_attack",
    "cw_l2_attack",
    "gaussian_attack", "saltnpepper_attack", "boundary_attack", "transfer_attack",
    "iterative_transfer_attack", "pgd_transfer_attack", "pgd_005_transfer_attack",
    "pgd_0063_00078_10_re_transfer_attack", # actually a bit stricter than fixed eps=16/step=2 and find the accuracy
    "pgd_0063_00078_10_last_transfer_attack",
    "pgd_03_001_40_re_transfer_attack", "pgd_03_001_40_bs_transfer_attack",
    "l2i_01_002_10_bs_transfer_attack", "l2i_01_002_10_nobs_transfer_attack",
    "l2i_03_005_10_nobs_transfer_attack", "l2i_05_01_10_nobs_transfer_attack",
    "l2i_05_02_5_nobs_transfer_attack",
    "l2i_05_005_10_nobs_transfer_attack",
    "aug_brightness_attack",
    "aug_huesat_attack",
    "aug_flip_attack",
    "aug_contrast_attack",
    "aug_gaussian_attack",
    "aug_nothing_attack"
]

def add_target_wrapper(fn):
    @wraps(fn)
    def _fn(*args, **kwargs):
        if kwargs.get("criterion", None) is None:
            target = kwargs.pop("target", None)
            if target:
                criterion = foolbox.criteria.TargetClass(target)
            else:
                criterion = foolbox.criteria.Misclassification()
            kwargs["criterion"] = criterion
        return fn(*args, **kwargs)
    return _fn

@add_target_wrapper
def cw_l2_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.CarliniWagnerL2Attack(model, criterion)
    return attack(image, label, max_iterations=500)

cw_l2_attack = cw_l2_transfer_attack

@add_target_wrapper
def pgd_0063_00078_10_last_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    with substitute_argscope(foolbox.Adversarial, {"return_last": True}): # we patch foolbox here... apply `other/d0d0df2919a8_patch` to foolbox:d0d0df2919a8
        return attack(image, label, binary_search=False, epsilon=0.063, stepsize=0.0078, iterations=10, return_early=False)

@add_target_wrapper
def pgd_0063_00078_10_re_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.063, stepsize=0.0078, iterations=10, return_early=True)

@add_target_wrapper
def pgd_03_001_40_re_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.3, stepsize=0.01, iterations=40, return_early=True)

@add_target_wrapper
def pgd_03_001_40_bs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.3, stepsize=0.01, iterations=40, return_early=True)

@add_target_wrapper
def pgd_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.2, stepsize=0.01, iterations=10, return_early=False)

@add_target_wrapper
def pgd_005_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.05, stepsize=0.01, iterations=10, return_early=False)

@add_target_wrapper
def iterative_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label)

@add_target_wrapper
def l2i_01_002_10_bs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.1, stepsize=0.02, iterations=10, binary_search=True)

@add_target_wrapper
def l2i_05_02_5_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.5, stepsize=0.2, iterations=5, binary_search=False)

@add_target_wrapper
def l2i_03_01_5_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.3, stepsize=0.1, iterations=5, binary_search=False)

@add_target_wrapper
def l2i_03_005_10_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.3, stepsize=0.05, iterations=10, binary_search=False)

@add_target_wrapper
def l2i_05_005_10_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.5, stepsize=0.05, iterations=10, binary_search=False)

@add_target_wrapper
def l2i_05_01_10_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.5, stepsize=0.1, iterations=10, binary_search=False)

@add_target_wrapper
def l2i_01_002_10_nobs_transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.1, stepsize=0.02, iterations=10, binary_search=False)

@add_target_wrapper
def transfer_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.GradientAttack(model, criterion)
    return attack(image, label, epsilons=100)

def gaussian_attack(model, image, label, verbose=False):
    if model(image) != label:
        return image
    epsilon = 1e-4

    perturbed_image = None
    for x in range(0, 1000):
        # draw noise pattern
        noise = np.random.uniform(-255, 255, size=image.shape)
        noise = noise.astype(image.dtype)

        # overlay noise pattern on image
        perturbed_image = image + epsilon * noise

        # clip pixel values to valid range [0, 1]
        perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)

        if model(perturbed_image) != label:
            break
        else:
            epsilon *= 2

    return perturbed_image

@add_target_wrapper
def saltnpepper_attack(model, image, label, criterion, verbose=False):
    attack = foolbox.attacks.SaltAndPepperNoiseAttack(model, criterion)
    return attack(image, label, epsilons=50, repetitions=10)

@add_target_wrapper
def boundary_attack(model, image, label, criterion, verbose=False):
    init_attack = foolbox.attacks.BlendedUniformNoiseAttack(model, criterion)
    init_adversarial = init_attack(
        image, label,
        epsilons=np.exp(np.linspace(np.log(0.01), np.log(2), num=90)))

    if init_adversarial is None:
        print('Init attack failed to produce an adversarial.')
        return None
    else:
        attack = foolbox.attacks.BoundaryAttack(model, criterion)
        return attack(image, label, iterations=45, max_directions=10,
                      tune_batch_size=False, starting_point=init_adversarial, verbose=verbose)


def aug_contrast_attack(model, image, label, verbose=False):
    from imgaug import augmenters as iaa
    im = np.clip(iaa.ContrastNormalization((0.8, 1.25), per_channel=True).augment_image(image), 0, 255)
    after_pred = int(np.argmax(model.predictions(im)))
    print("contrast: (0.8,1.25); True label; before aug label; after; :", label, np.argmax(model.predictions(image)), after_pred)
    return im, after_pred != label

def aug_brightness_attack(model, image, label, verbose=False):
    from imgaug import augmenters as iaa
    im = np.clip(iaa.Add((-30, 30), per_channel=True).augment_image(image), 0, 255)
    after_pred = int(np.argmax(model.predictions(im)))
    print("brightness: (-30, 30); True label; before aug label; after; :", label, np.argmax(model.predictions(image)), after_pred)
    return im, after_pred != label

def aug_huesat_attack(model, image, label, verbose=False):
    from imgaug import augmenters as iaa
    im = np.clip(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True).augment_image(image), 0,255)
    # im = iaa.AddToHueAndSaturation(value=(-30, 30), per_channel=True).augment_image(image)
    after_pred = int(np.argmax(model.predictions(im)))
    print("huesat: (-10,10): True label; before aug label; after; :", label, np.argmax(model.predictions(image)), after_pred)
    return im, after_pred != label

def aug_flip_attack(model, image, label, verbose=False):
    from imgaug import augmenters as iaa
    im = iaa.Fliplr(1).augment_image(image)
    after_pred = int(np.argmax(model.predictions(im)))
    print("flip; True label; before aug label; after; :", label, np.argmax(model.predictions(image)), after_pred)
    return im, after_pred != label

def aug_gaussian_attack(model, image, label, verbose=False):
    from imgaug import augmenters as iaa
    # im = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.08*255), per_channel=True).augment_image(image)
    im = np.clip(iaa.AdditiveGaussianNoise(loc=0, scale=0.08*255, per_channel=True).augment_image(image), 0, 255)
    after_pred = int(np.argmax(model.predictions(im)))
    print("gaussian noise: 0.08*255; True label; before aug label; after; :", label, np.argmax(model.predictions(image)), after_pred)
    return im, after_pred != label

def aug_nothing_attack(model, image, label, verbose=False): # only for adv_pttern test
    im = np.zeros(image.shape)
    return im, int(np.argmax(model.predictions(im))) != label
