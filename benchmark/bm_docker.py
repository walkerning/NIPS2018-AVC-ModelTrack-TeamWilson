#!/home/foxfi/anaconda3/bin/python3
import os
import sys
import pwd
import yaml
import logging
import argparse
import subprocess
import pickle
import contextlib

import numpy as np
from scipy.misc import imread

import foolbox
from fmodel import create_fmodel
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete

@contextlib.contextmanager
def substitute_argscope(_callable, dct):
    if isinstance(_callable, type): # class
        _callable.old_init = _callable.__init__
        def new_init(self, *args, **kwargs):
            kwargs.update(dct)
            return self.old_init(*args, **kwargs)
        _callable.__init__ = new_init
        yield
        _callable.__init__ = _callable.old_init
    else: # function/methods
        raise Exception("not implemented")
                
class ImageReader(object):
    available_methods = ["npy", "img"]
    def __init__(self, tp):
        assert tp in self.available_methods
        self.tp = tp

    def _read_image(self, key):
        input_folder = os.getenv('INPUT_IMG_PATH')
        img_path = os.path.join(input_folder, key)
        image = imread(img_path)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        return image

    def read_images(self):
        if self.tp == "npy":
            for key, im, label in read_images():
                yield (key, im, label)
        else: # img
            filepath = os.getenv('INPUT_YML_PATH')
            with open(filepath, 'r') as ymlfile:
                data = yaml.load(ymlfile)
            for key in data.keys():
                im = self._read_image(key)
                if im.shape != (64, 64, 3):
                    if im.shape != (64,64):
                        logging.warning("shape of image read from file {} is not (64, 64, 3) or (64, 64). ignore.".format(key))
                        continue
                    im = np.tile(np.expand_dims(im, -1), (1,1,3))
                    # continue
                yield (key, im, data[key])

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

avail_attacks = ["gaussian", "saltnpepper", "boundary", "transfer", "iterative_transfer", "pgd_transfer", "pgd_005_transfer", "pgd_03_001_40_re_transfer", "pgd_03_001_40_bs_transfer", "l2i_01_002_10_bs_transfer", "l2i_01_002_10_nobs_transfer", "l2i_03_005_10_nobs_transfer", "l2i_05_01_10_nobs_transfer", "l2i_05_02_5_nobs_transfer"]
bms = {n: globals()[n + "_attack"] for n in avail_attacks}

def main(reader, types, save, verbose=False, addi_name=None):
    # instantiate blackbox and substitute model
    test_user = pwd.getpwuid(os.getuid()).pw_name
    container_name = 'avc_test_model_submission_{}{}'.format(test_user, "_{}".format(addi_name) if addi_name else "")
    form = '{{ .NetworkSettings.IPAddress }}'
    cmd = "docker inspect --format='{}' {}".format(form, container_name)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    ip = p.stdout.read()[:-1].decode('UTF-8')
    
    p = subprocess.Popen("docker port " + container_name + " | cut -d/ -f1", shell=True, stdout=subprocess.PIPE)
    port = p.stdout.read()[:-1].decode('UTF-8')

    p = subprocess.Popen("docker exec " + container_name + " bash -c 'echo $EVALUATOR_SECRET'", shell=True, stdout=subprocess.PIPE)
    env_sec = p.stdout.read()[:-1].decode('UTF-8')
    print("Using container: {}. addr: {}:{}".format(container_name, ip, port))

    os.environ["MODEL_SERVER"] = ip
    os.environ["MODEL_PORT"] = port
    os.environ["EVALUATOR_SECRET"] = env_sec

    print('model url: ', "http://{ip}:{port}".format(ip=ip, port=port))

    forward_model = load_model()

    if any([n.endswith("transfer") for n in types]):
        backward_model = create_fmodel()

        # instantiate differntiable composite model
        # (predictions from blackbox, gradients from substitute)
        transfer_model = foolbox.models.CompositeModel(
            forward_model=forward_model,
            backward_model=backward_model)

    def distance(X, Y):
        X = X.astype(np.float64) / 255
        Y = Y.astype(np.float64) / 255
        return np.linalg.norm(X - Y)
    def worst_case_distance(X):
        worst_case = np.zeros_like(X)
        worst_case[X < 128] = 255
        return distance(X, worst_case)
    distances = [[] for _ in types]
    clean_predicts = []
    accuracy_counter = 0
    num_test = 0
    not_adv = {tp: 0 for tp in types}
    for ind, (file_name, image, label) in enumerate(reader.read_images()):
        num_test += 1
        predict_label = forward_model(image)
        clean_predicts.append((file_name, predict_label, label))
        print("image {}: {} {}".format(file_name, predict_label, label))
        if predict_label != label:
            accuracy_counter += 1
        for it, tp in enumerate(types):
            with substitute_argscope(foolbox.Adversarial, {"distance": foolbox.distances.Linf if "pgd" in tp else foolbox.distances.MSE}):
                if not tp.endswith("transfer"):
                    adversarial = bms[tp](forward_model, image, label, verbose=verbose)
                else:
                    adversarial = bms[tp](transfer_model, image, label, verbose=verbose)
            if adversarial is None:
                pixel_dis = float(worst_case_distance(image))
            else:
                adversarial = adversarial.astype(np.uint8)
                pixel_dis = float(distance(image, adversarial))
            # #pixel_dis = 100
            # if isinstance(adversarial, np.ndarray):
            #     pixel_dis = np.mean(np.abs(adversarial - image))
            distances[it].append(pixel_dis)
            print("image {}: {} attack / distance: {}".format(ind+1, tp, pixel_dis))
            sys.stdout.flush()
            # if args.save and adversarial is not None:
            #     # store_adversarial(os.path.join(tp, file_name + "_" + str(pixel_dis)), adversarial)
            #     store_adversarial(os.path.join(tp, os.path.basename(file_name)), adversarial)
        
            if args.save:
                # if adversarial is None or not_success:
                if adversarial is None:
                    # adversarial = adversarial or image
                    adversarial = image.astype(np.uint8)
                    not_adv[tp] += 1
                # store_adversarial(os.path.join(tp, file_name + "_" + str(pixel_dis)), adversarial)
                if args.use_tofile:
                    adversarial.tofile(os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], tp, os.path.basename(file_name).split(".")[0] + ".bin"))
                else:
                    store_adversarial(os.path.join(tp, os.path.basename(file_name)), adversarial)
                

    print("test accuracy: {:.2f}%".format(100. - accuracy_counter * 100. / num_test))
    print("not find adv samples: \n\t{}".format("\n\t".join(["{}: {}; {}%".format(tp, num, float(num)/num_test * 100) for tp, num in not_adv.items()])))
    open("file_predict_label.txt", "w").write("\n".join(["{} {} {}".format(*x) for x in clean_predicts]))
    for tp, dis in zip(types, distances):
        distance_array = np.array(dis)
        print("{}: median pixel distance: {}, mean pixel distance: {}".format(tp, np.median(distance_array), distance_array.mean()))

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    # attack_complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help="a yaml file contains image fpath and corresponding label.")
    parser.add_argument("--gpu", default="1", type=str, help="test using which gpu")
    parser.add_argument("--save", default=None, type=str, help="save adversarial to folder")
    parser.add_argument("--image-path", default=os.path.expanduser("~/test_images"), type=str, help="image base path")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="print verbose info")
    parser.add_argument("-t", "--type", action="append", choices=avail_attacks, default=[], help="what attack to be performed") # , required=True)
    parser.add_argument("--image-type", choices=ImageReader.available_methods, default="npy", help="image type")
    parser.add_argument("--use-tofile", action="store_true", default=False, help="use arr.tofile instead of np.save")
    parser.add_argument("--name", default=None, help="default to the current user name")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["INPUT_YML_PATH"] = args.test_file
    os.environ["INPUT_IMG_PATH"] = args.image_path
    if args.save is not None:
        os.environ["OUTPUT_ADVERSARIAL_PATH"] = args.save
        for tp in args.type:
            os.makedirs(os.path.join(args.save, tp), exist_ok=True)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    reader = ImageReader(args.image_type)
    main(reader, args.type, args.save is not None, verbose=args.verbose, addi_name=args.name)
