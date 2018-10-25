#!/bin/bash
set -e
ADDI_NAME=${ADDI_NAME:-none}
TESTTYPE=${TESTTYPE:-transfer}
TEST_RESNET=${TEST_RESNET:-1}
TEST_INCEPTION=${TEST_INCEPTION:-1}
TEST_VGG=${TEST_VGG:-0}
gpu=${GPU:-0}
#tmp_model_cfg=${TMP_MODEL_CFG:-$(tempfile)}
test_name=${1}
if [[ -z "${test_name}" ]]; then
   echo "must supply test name, results will be saved to results/<test_name>"
   exit 1;
fi
mkdir -p results/${test_name}
if [[ "${ADDI_NAME}" == "none" ]]; then
    addi_arg=""
else
    addi_arg="--name ${ADDI_NAME}"
fi
#cp cfgs/resnet18.yaml ${tmp_model_cfg}   
# 将当前模型使用avc-test-model-nics跑起来, 测试当前模型的saltnpepper, gaussian攻击距离
# FMODEL_MODEL_CFG=${tmp_model_cfg} python main.py ~/yml_files/labels.yml --gpu ${gpu} -t gaussian -t saltnpepper 2>&1 | tee results/${test_name}/gaussian_saltnpepper.log
# 测试当前模型用resnet18攻击的距离
# FMODEL_MODEL_CFG=${tmp_model_cfg} python main.py ~/yml_files/labels.yml --gpu ${gpu} -t transfer -t iterative_transfer --save  results/${test_name}/ 2>&1 | tee results/${test_name}/transfer.log
if [[ ${TEST_RESNET} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/resnet18.yaml python main.py ~/yml_files/labels.yml ${addi_arg} --gpu ${gpu} -t ${TESTTYPE}  --save  results/${test_name}/ 2>&1 | tee results/${test_name}/${TESTTYPE}.log
fi
# 测试当前模型用resnet18 adv trained (submit3)攻击的距离
# cp cfgs/resnet_advtrained.yaml ${tmp_model_cfg}   
# FMODEL_MODEL_CFG=${tmp_model_cfg} python main.py ~/yml_files/labels.yml --gpu ${gpu} -t transfer -t iterative_transfer --save  results/${test_name}/genbysubmit3 2>&1 | tee results/${test_name}/transfer_submit3.log
# 测试当前模型用inception baseline攻击的距离
# FMODEL_MODEL_CFG=cfgs/inception.yaml python main.py ~/yml_files/labels.yml --gpu ${gpu} -t transfer -t iterative_transfer --save  results/${test_name}/genbyinception 2>&1 | tee results/${test_name}/transfer_submit3.log
if [[ ${TEST_INCEPTION} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/inception.yaml python main.py ~/yml_files/labels.yml ${addi_arg} --gpu ${gpu} -t ${TESTTYPE} --save  results/${test_name}/genbyinception 2>&1 | tee results/${test_name}/${TESTTYPE}_inception.log
fi

if [[ ${TEST_VGG} -gt 0 ]]; then
    FMODEL_MODEL_CFG=cfgs/vgg11.yaml python main.py ~/yml_files/labels.yml ${addi_arg} --gpu ${gpu} -t ${TESTTYPE} --save  results/${test_name}/genbyvgg 2>&1 | tee results/${test_name}/${TESTTYPE}_vgg.log
fi


# 单独出去: 当前模型生成的白盒样本存下来, 攻击resnet18的transfer情况
# 单独出去: 测试当前模型攻击resnet18的情况


