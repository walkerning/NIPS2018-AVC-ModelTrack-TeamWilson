#!/home/foxfi/anaconda3/bin/python3
import os
import pwd
import yaml
import logging
import argparse
import subprocess
import pickle

import numpy as np
from scipy.misc import imread

import foolbox
from fmodel import create_fmodel
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete

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
                    logging.warning("shape of image read from file {} is not (64, 64, 3). ignore.".format(key))
                    continue
                yield (key, im, data[key])
    
def iterative_transfer_attack(model, image, label, verbose=False):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label)

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

avail_attacks = ["gaussian", "saltnpepper", "boundary", "transfer", "iterative_transfer"]
bms = {n: globals()[n + "_attack"] for n in avail_attacks}

def main(reader, types, save, verbose=False):
    # instantiate blackbox and substitute model
    test_user = pwd.getpwuid(os.getuid()).pw_name
    container_name = 'avc_test_model_submission_{}'.format(test_user)
    form = '{{ .NetworkSettings.IPAddress }}'
    cmd = "docker inspect --format='{}' {}".format(form, container_name)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    ip = p.stdout.read()[:-1].decode('UTF-8')
    
    p = subprocess.Popen("docker port " + container_name + " | cut -d/ -f1", shell=True, stdout=subprocess.PIPE)
    port = p.stdout.read()[:-1].decode('UTF-8')

    p = subprocess.Popen("docker exec " + container_name + " bash -c 'echo $EVALUATOR_SECRET'", shell=True, stdout=subprocess.PIPE)
    env_sec = p.stdout.read()[:-1].decode('UTF-8')

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

    distance = []
    clean_predicts = []
    accuracy_counter = 0
    num_test = 0
    for ind, (file_name, image, label) in enumerate(reader.read_images()):
        num_test += 1
        predict_label = forward_model(image)
        clean_predicts.append((file_name, predict_label, label))
        print("image {}: {} {}".format(file_name, predict_label, label))
        if predict_label != label:
            accuracy_counter += 1
        for tp in types:
            if not tp.endswith("transfer"):
                adversarial = bms[tp](forward_model, image, label, verbose=verbose)
            else:
                adversarial = bms[tp](transfer_model, image, label, verbose=verbose)
            pixel_dis = 100
            if isinstance(adversarial, np.ndarray):
                pixel_dis = np.mean(np.abs(adversarial - image))
            distance.append(pixel_dis)
            print("image {}: {} attack / distance: {}".format(ind+1, tp, pixel_dis))
            if args.save:
                # store_adversarial(os.path.join(tp, file_name + "_" + str(pixel_dis)), adversarial)
                store_adversarial(os.path.join(tp, os.path.basename(file_name)), adversarial)

    print("test accuracy: {:.2f}%".format(100. - accuracy_counter * 100. / num_test))
    open("file_predict_label.txt", "w").write("\n".join(["{} {} {}".format(*x) for x in clean_predicts]))
    if len(types) == 1:
        distance_array = np.array(distance)
        print("median pixel distance: {}, mean pixel distance: {}".format(np.median(distance_array), distance_array.mean()))

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
    main(reader, args.type, args.save is not None, verbose=args.verbose)
