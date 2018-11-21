#!/bin/bash
set -e
function assert_ne() {
    if [[ -z "$1" ]];  then
	echo "$2 must not be empty"
	exit 1
    fi
}
save_name=${1}
FORWARD_CFG=${FORWARD_CFG}
BACKWARD_CFG=${BACKWARD_CFG}
TESTTYPE=${TESTTYPE:-"l2i_05_005_10_nobs_transfer"}
assert_ne ${save_name} "BACKWARD_CFG"
assert_ne ${FORWARD_CFG} "FORWARD_CFG"
assert_ne ${BACK_CFG} "BACKWARD_CFG"
save_dir=${SAVE_DIR:-/home/eva_share/foxfi/nips2018/data/raw_gen}
GPU=${GPU:-0}

result_dir=${save_dir}/${save_name}
mkdir -p ${result_dir}
# train
python benchmark/bm.py ~/tiny-imagenet-200/train_labels.yml --use-tofile --image-type img --image-path ~/tiny-imagenet-200/ --forward-cfg ${FORWARD_CFG} --backward-cfg ${BACKWARD_CFG} --gpu ${GPU} --save ${result_dir}/train/ -t ${TESTTYPE} 2>&1 | tee ${result_dir}/train.log
# val
python benchmark/bm.py ~/tiny-imagenet-200/val/val_labels.yml --use-tofile --image-type img --image-path ~/tiny-imagenet-200/val/images --forward-cfg ${FORWARD_CFG} --backward-cfg ${BACKWARD_CFG} --gpu ${GPU} --save ${result_dir}/val/ -t ${TESTTYPE} 2>&1 | tee ${result_dir}/val.log
