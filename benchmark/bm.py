#!/home/foxfi/anaconda3/bin/python3
import os
import sys
import pwd
import logging
import datetime
import argparse
import subprocess

import numpy as np
import pandas as pd

import foolbox
from adversarial_vision_challenge import load_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import attacks
from attacks import *

avail_attacks = [n.rsplit("_", 1)[0] for n in attacks.__all__]
bms = {n: globals()[n + "_attack"] for n in avail_attacks}

def main(reader, types, save, backward_cfg=None, forward_cfg=None, verbose=False, addi_name=None, adv_pattern_dir=None):
    if forward_cfg is None:
        print("Using container, addi_name: {}".format(addi_name))
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
    else:
        forward_model = create_fmodel_cfg(forward_cfg)

    if any([n.endswith("transfer") for n in types]):
        if backward_cfg is None:
            backward_cfg = os.environ.get("FMODEL_MODEL_CFG", None)
            assert backward_cfg is not None, "Neither --backward-cfg nor FMODEL_MODEL_CFG env is provided or set"
            logging.warning("backward_cfg is not supplied on cmd line; use the environment variable value {} as backward config file".format(backward_cfg))
        backward_model = create_fmodel_cfg(backward_cfg)

        # instantiate differntiable composite model
        # (predictions from blackbox, gradients from substitute)
        transfer_model = foolbox.models.CompositeModel(
            forward_model=forward_model,
            backward_model=backward_model)


    distances = [[] for _ in types]
    clean_predicts = []
    adv_pattern_predicts = []
    accuracy_counter = 0
    num_test = 0
    not_adv = {tp: 0 for tp in types}
    for ind, (file_name, image, label) in enumerate(reader.read_images()):
        num_test += 1
        predict_label = np.argmax(forward_model.predictions(image))
        clean_predict = [file_name.split(".")[0], label, predict_label]
        print("image {}: {} {}".format(file_name, label, predict_label))
        if predict_label != label:
            accuracy_counter += 1
        if adv_pattern_dir:
            adv_im = np.load(os.path.join(adv_pattern_dir, os.path.basename(file_name).split(".")[0]) + ".npy")
            adv_pattern = adv_im - image
            before = np.argmax(forward_model.predictions(adv_im))
            adv_pattern_predict = [file_name.split(".")[0], label, before]
        for it, tp in enumerate(types):
            with substitute_argscope(foolbox.Adversarial, {"distance": foolbox.distances.Linf if "pgd" in tp else foolbox.distances.MSE}):
                if not tp.endswith("transfer"):
                    if tp.startswith("aug_"):
                        adversarial, suc = bms[tp](forward_model, image, label, verbose=verbose)
                    else:
                        adversarial = bms[tp](forward_model, image, label, verbose=verbose)
                        suc = adversarial is not None
                else:
                    adversarial = bms[tp](transfer_model, image, label, verbose=verbose)
                    suc = adversarial is not None
            if not suc:
                pixel_dis = float(worst_case_distance(image))
                not_adv[tp] += 1
                clean_predict.append(label) # predict to be the original albel
            else:
                adversarial = adversarial.astype(np.uint8)
                pixel_dis = float(distance(image, adversarial))
                clean_predict.append(np.argmax(forward_model.predictions(adversarial)))
            if adv_pattern_dir:
                assert adversarial is not None
                # if adversarial is not None: # only test augumentation for now so adversairal must not be None
                after = np.argmax(forward_model.predictions(adversarial + adv_pattern))
                adv_pattern_predict.append(after)
            distances[it].append(pixel_dis)
            print("image {}: {} attack / distance: {}".format(ind+1, tp, pixel_dis))
            sys.stdout.flush()
            if args.save:
                if adversarial is None:
                    adversarial = image
                adversarial = adversarial.astype(np.uint8)
                if args.use_tofile: # save bin file
                    adversarial.tofile(os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], tp, os.path.basename(file_name).split(".")[0] + ".bin"))
                else: # save npy
                    np.save(os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], tp, os.path.basename(file_name).split(".")[0]), adversarial)
        clean_predicts.append(clean_predict)
        if adv_pattern_dir:
            adv_pattern_predicts.append(adv_pattern_predict)
                

    print("test accuracy: {:.2f}%".format(100. - accuracy_counter * 100. / num_test))
    print("not find adv samples: \n\t{}".format("\n\t".join(["{}: {}; {:.3f}%".format(tp, num, float(num)/num_test * 100) for tp, num in not_adv.items()])))

    # open("file_predict_label.txt", "w").write("\n".join(["{} {} {}".format(*x) for x in clean_predicts]))
    predict_fname = os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], "predicts_{}.csv".format(datetime.datetime.now().strftime("%m-%d_%H-%M-%S")))
    print("Save predict to csv: ", predict_fname)
    pd.DataFrame(clean_predicts, columns=["filename", "label", "pred"] + list(types)).to_csv(predict_fname, index=False)
    input_link = os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], "pred")
    if os.path.islink(input_link):
        os.unlink(input_link) # remove already exists symbol link
    os.symlink(os.path.abspath(os.environ["INPUT_IMG_PATH"]), input_link)
    if adv_pattern_predicts:
        adv_pattern_predict_fname = os.path.join(os.environ["OUTPUT_ADVERSARIAL_PATH"], "adv_pattern_predicts_{}.csv".format(datetime.datetime.now().strftime("%m-%d_%H-%M-%S")))
        pd.DataFrame(adv_pattern_predicts, columns=["filename", "label", "pred"] + list(types)).to_csv(adv_pattern_predict_fname, index=False)
        adv_pattern_acc = np.sum(np.array([[p[1]] for p in adv_pattern_predicts]) == np.array([p[2:] for p in adv_pattern_predicts]), axis=0) / float(len(adv_pattern_predicts))
        print("adv pattern acc: \n\t{}".format("\n\t".join(["{}: {:.3f}%".format(tp, acc*100) for tp, acc in zip(["pred"]+list(types), adv_pattern_acc)])))
    for tp, dis in zip(types, distances):
        distance_array = np.array(dis)
        print("{}: median pixel distance: {}, mean pixel distance: {}".format(tp, np.median(distance_array), distance_array.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help="a yaml file contains image fpath and corresponding label.")
    parser.add_argument("--gpu", default="1", type=str, help="test using which gpu")
    parser.add_argument("--save", default=None, type=str, help="save adversarial to folder")
    parser.add_argument("--image-path", default=os.path.expanduser("~/test_images"), type=str, help="image base path")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="print verbose info")
    parser.add_argument("-t", "--type", action="append", choices=avail_attacks, default=[], help="what attack to be performed") # , required=True)
    parser.add_argument("--image-type", choices=ImageReader.available_methods, default="npy", help="image type")
    parser.add_argument("--dataset", choices={"tiny-imagenet", "cifar10"}, default="tiny-imagenet")
    # parser.add_argument("--shape", type=str, "An expression evaluated to a tuple of int", default=None)
    parser.add_argument("--use-tofile", action="store_true", default=False, help="use arr.tofile instead of np.save")
    parser.add_argument("--name", default=None, help="default to the current user name")
    parser.add_argument("--backward-cfg", default=None)
    parser.add_argument("--forward-cfg", metavar="FORWARD_CFG", default=None, help="if FORWARD_CFG is specified, will not use docker.")
    parser.add_argument("--adv-pattern-dir", default=None, help="If specified, will evaluate the transferability adv directions to the all the transfer types")

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
        logging.getLogger().setLevel(logging.INFO) # set root logger level
    else:
        logging.basicConfig(level=logging.WARNING) # this does not work if some other modules configued the logging module already
        logging.getLogger().setLevel(logging.WARNING) # set root logger level

    reader = ImageReader(args.image_type, dataset=args.dataset)

    print("CMD: ", " ".join(sys.argv))

    main(reader, args.type, args.save is not None, args.backward_cfg, args.forward_cfg, verbose=args.verbose, addi_name=args.name, adv_pattern_dir=args.adv_pattern_dir)
