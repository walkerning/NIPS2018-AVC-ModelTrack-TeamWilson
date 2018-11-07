import numpy as np
import foolbox

__all__ = [
    "cw_l2_transfer_attack",
    "gaussian_attack", "saltnpepper_attack", "boundary_attack", "transfer_attack",
    "iterative_transfer_attack", "pgd_transfer_attack", "pgd_005_transfer_attack",
    "pgd_03_001_40_re_transfer_attack", "pgd_03_001_40_bs_transfer_attack",
    "l2i_01_002_10_bs_transfer_attack", "l2i_01_002_10_nobs_transfer_attack",
    "l2i_03_005_10_nobs_transfer_attack", "l2i_05_01_10_nobs_transfer_attack",
    "l2i_05_02_5_nobs_transfer_attack"
]

def cw_l2_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.CarliniWagnerL2Attack(model, criterion)
    return attack(image, label)

def pgd_03_001_40_re_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.3, stepsize=0.01, iterations=40, return_early=True)

def pgd_03_001_40_bs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.3, stepsize=0.01, iterations=40, return_early=True)

def pgd_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.2, stepsize=0.01, iterations=10, return_early=False)

def pgd_005_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.PGD(model, criterion)
    return attack(image, label, binary_search=False, epsilon=0.05, stepsize=0.01, iterations=10, return_early=False)

def iterative_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label)

def l2i_01_002_10_bs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.1, stepsize=0.02, iterations=10, binary_search=True)

def l2i_05_02_5_nobs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.5, stepsize=0.2, iterations=5, binary_search=False)

def l2i_03_01_5_nobs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.3, stepsize=0.1, iterations=5, binary_search=False)

def l2i_03_005_10_nobs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.3, stepsize=0.05, iterations=10, binary_search=False)

def l2i_05_01_10_nobs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.5, stepsize=0.1, iterations=10, binary_search=False)

def l2i_01_002_10_nobs_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label, epsilon=0.1, stepsize=0.02, iterations=10, binary_search=False)

def transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
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

def saltnpepper_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.SaltAndPepperNoiseAttack(model, criterion)
    return attack(image, label, epsilons=50, repetitions=10)

def boundary_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
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
