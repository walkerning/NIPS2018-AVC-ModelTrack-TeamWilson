#!/home/foxfi/anaconda3/bin/python3
import os
import pwd
import logging
import argparse
import subprocess

import numpy as np

import foolbox
from fmodel import create_fmodel
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete

def iterative_transfer_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.L2BasicIterativeAttack(model, criterion)
    return attack(image, label)

def transfer_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.GradientAttack(model, criterion)
    return attack(image, label, epsilons=100)

def gaussian_attack(model, image, label):
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

def saltnpepper_attack(model, image, label):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.SaltAndPepperNoiseAttack(model, criterion)
    return attack(image, label, epsilons=50, repetitions=10)

def boundary_attack(model, image, label):
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
                      tune_batch_size=False, starting_point=init_adversarial)

avail_attacks = ["gaussian", "saltnpepper", "boundary", "transfer", "iterative_transfer"]
bms = {n: globals()[n + "_attack"] for n in avail_attacks}

def main(types, save):
    # instantiate blackbox and substitute model
    test_user = pwd.getpwuid(os.getuid()).pw_name
    container_name = 'avc_test_model_submission_{}'.format(test_user)
    form = '{{ .NetworkSettings.IPAddress }}'
    cmd = "docker inspect --format='{}' {}".format(form, container_name)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.wait()
    ip = p.stdout.read()[:-1].decode('UTF-8')
    
    p = subprocess.Popen("docker port  avc_test_model_submission_foxfi | cut -d/ -f1", shell=True, stdout=subprocess.PIPE)
    port = p.stdout.read()[:-1].decode('UTF-8')
    os.environ["MODEL_SERVER"] = ip
    os.environ["MODEL_PORT"] = port
    print('model url: ', "http://{ip}:{port}".format(ip=ip, port=port))

    forward_model = load_model()

    if any([n.endswith("transfer") for n in types]):
        backward_model = create_fmodel()

        # instantiate differntiable composite model
        # (predictions from blackbox, gradients from substitute)
        transfer_model = foolbox.models.CompositeModel(
            forward_model=forward_model,
            backward_model=backward_model)

    for ind, (file_name, image, label) in enumerate(read_images()):
        for tp in types:
            print("image {}: {} attack".format(ind+1, tp))
            if not tp.endswith("transfer"):
                adversarial = bms[tp](forward_model, image, label)
            else:
                adversarial = bms[tp](transfer_model, image, label)                
            if args.save:
                store_adversarial(os.path.join(tp, file_name), adversarial)

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
    parser.add_argument("-t", "--type", action="append", choices=avail_attacks, help="what attack to be performed", required=True)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["INPUT_YML_PATH"] = args.test_file
    if args.save is not None:
        os.environ["OUTPUT_ADVERSARIAL_PATH"] = args.save
        for tp in args.type:
            os.makedirs(os.path.join(args.save, tp), exist_ok=True)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    os.environ["INPUT_IMG_PATH"] = args.image_path


    main(args.type, args.save is not None)
