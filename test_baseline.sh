#!/bin/bash
set -e
# NOTE: change these to the paths on your machine
VAL2000_LABEL=/home/foxfi/yml_files/val_labels_2000sample.yaml
TINY_IMGNET_VALDIR=/home/foxfi/tiny-imagenet-200/val/images/

DATASET=${DATASET:-"tiny-imagenet"}
IMAGE_TYPE=${IMAGE_TYPE:-"npy"}

# Handle configurations
VERBOSE=${VERBOSE:-0}
gpu=${GPU:-0} # use which gpu
LABEL_FILE=${LABEL_FILE:-"/home/foxfi/yml_files/labels.yml"}
TEST_FIRST=${TEST_FIRST:-""} # only test first n pics (require the content of yaml file to be orgnized as "one example per line" format)
TEST_LOCAL=${TEST_LOCAL:-""}
ADDI_NAME=${ADDI_NAME:-none} # docker addi name
TEST_2000=${TEST_2000:-0} # using val 2000 to test
TESTTYPE=${TESTTYPE:-transfer} # transfer/iterative transfer

TEST_USING=${TEST_USING:-""} # transfer test using the model specfiied by this cfg
TEST_RESNET=${TEST_RESNET:-1}
TEST_INCEPTION=${TEST_INCEPTION:-0}
TEST_INCEPTION_RES=${TEST_INCEPTION_RES:-0}
TEST_VGG=${TEST_VGG:-0}
TEST_BOUNDARY=${TEST_BOUNDARY:-0}
TEST_CW=${TEST_CW:-0}

# log and attacked pics will be saved to ${result_dir}/${test_name} or ${result_dir}/${test_name}-val2000
test_name=${1}
result_dir=${2:-results_new}

forward_arg=""
if [[ ! -z "${TEST_LOCAL}" ]]; then
    forward_arg="--forward-cfg ${TEST_LOCAL}"
fi
if [[ -z "${test_name}" ]]; then
   echo "must supply test name, results will be saved to ${result_dir}/<test_name>"
   exit 1;
fi
if [[ "${ADDI_NAME}" == "none" ]]; then
    addi_arg=""
else
    addi_arg="--name ${ADDI_NAME}"
fi
if [[ "${TEST_2000}" -gt 0 ]]; then
    label_f=${VAL2000_LABEL}
    impath_arg="--image-path ${TINY_IMGNET_VALDIR} --image-type img"
    test_name=${test_name}-val2000
else
    label_f=${LABEL_FILE}
    impath_arg=""
fi

dataset_arg="--dataset ${DATASET} --image-type ${IMAGE_TYPE}"
if [[ "${VERBOSE}" -gt 0 ]]; then
    dataset_arg="${dataset_arg} --verbose"
fi

if [[ ! -z "${TEST_FIRST}" ]]; then
    temp_f=$(tempfile)
    head -n ${TEST_FIRST} ${label_f} >${temp_f}
    label_f=${temp_f}
fi

mkdir -p ${result_dir}/${test_name}

# transfer test using another model
if [[ ! -z ${TEST_USING} ]]; then
    using_basename=`basename ${TEST_USING}`
    FMODEL_MODEL_CFG=${TEST_USING} python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t ${TESTTYPE}  ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/${using_basename}/ 2>&1 | tee ${result_dir}/${test_name}/${TESTTYPE}_${using_basename}.log
fi

# resnet transfer  attack
if [[ ${TEST_RESNET} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/resnet18.yaml python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t ${TESTTYPE}  ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/ 2>&1 | tee ${result_dir}/${test_name}/${TESTTYPE}.log
fi
# inception transfer  attack
if [[ ${TEST_INCEPTION} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/inception.yaml python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t ${TESTTYPE} ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/genbyinception 2>&1 | tee ${result_dir}/${test_name}/${TESTTYPE}_inception.log
fi

# vgg transfer attack
if [[ ${TEST_VGG} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/vgg11.yaml python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t ${TESTTYPE} ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/genbyvgg 2>&1 | tee ${result_dir}/${test_name}/${TESTTYPE}_vgg.log
fi

# inception-res-v2 transfer attack
if [[ ${TEST_INCEPTION_RES} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/inception_res_v2.yaml python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t ${TESTTYPE} ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/genbyinceptionresv2 2>&1 | tee ${result_dir}/${test_name}/${TESTTYPE}_inceptionresv2.log
fi

# boundary attack
if [[ ${TEST_BOUNDARY} -gt 0 ]]; then
    python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t boundary ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/ 2>&1 | tee ${result_dir}/${test_name}/boundary.log
fi

# boundary attack
if [[ ${TEST_BOUNDARY} -gt 0 ]]; then
    python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t boundary ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/ 2>&1 | tee ${result_dir}/${test_name}/boundary.log
fi

# white-box C&W attack, a strong attack
if [[ ${TEST_CW} -gt 0 ]]; then
    python benchmark/bm.py ${label_f} ${impath_arg} ${addi_arg} --gpu ${gpu} -t cw_l2 ${forward_arg} ${dataset_arg} --save  ${result_dir}/${test_name}/ 2>&1 | tee ${result_dir}/${test_name}/cw_l2.log
fi

if [[ ! -z "${TEST_FIRST}" ]]; then
    rm ${temp_f}
fi
