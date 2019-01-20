#!/bin/bash
name=${1}
gpu=${GPU:-2}
TEST_FIRST=${TEST_FIRST:-1000}
testtype=${TESTTYPE:-pgd_0031_00078_7_last_transfer}
phase=${PHASE:-test}
label_file=cifar10_${phase}.yaml
AGAIN_ONLY=${AGAIN_ONLY:-0}
again=${AGAIN:-0}
test_cw=${TEST_CW:-0}
nat=${NAT:-./cfgs/cifar10/resnet_baseline.yaml}
WHITE_ONLY=${WHITE_ONLY:-0}
cfg=cfgs/cifar10/${name}.yaml

if [[ ${AGAIN_ONLY} -eq 0 ]]; then
  if [[ ${nat} != "0" && ${WHITE_ONLY} -eq 0 ]]; then
      # resnet baseline/A_nat for now models.
      GPU=${gpu} TEST_FIRST=${TEST_FIRST} TESTTYPE=${testtype} IMAGE_TYPE=bin DATASET=cifar10 TEST_USING=${nat} TEST_RESNET=0 TEST_LOCAL=${cfg} LABEL_FILE=${label_file} bash test_baseline.sh ${name} results_madrytest/${phase}
  fi
  
  # whitebox
  GPU=${gpu} TEST_FIRST=${TEST_FIRST} TESTTYPE=${testtype} IMAGE_TYPE=bin DATASET=cifar10 TEST_USING=${cfg} TEST_RESNET=0 TEST_LOCAL=${cfg} LABEL_FILE=${label_file} bash test_baseline.sh ${name} results_madrytest/${phase}
fi  
# A'; same arch(the hardest transfer scneario)
if [[ ${AGAIN} -gt 0 ]]; then
    cfg_again=cfgs/cifar10/${name}_again.yaml
    GPU=${gpu} TEST_FIRST=${TEST_FIRST} TESTTYPE=${testtype} IMAGE_TYPE=bin DATASET=cifar10 TEST_USING=${cfg_again} TEST_RESNET=0 TEST_LOCAL=${cfg} LABEL_FILE=${label_file} bash test_baseline.sh ${name} results_madrytest/${phase}
fi

if [[ ${TEST_CW} -gt  0 ]]; then
    GPU=${GPU} TEST_FIRST=200 TESTTYPE=gaussian IMAGE_TYPE=bin DATASET=cifar10 TEST_CW=1 TEST_RESNET=0 TEST_LOCAL=${cfg} LABEL_FILE=cifar10_${phase}.yaml bash test_baseline.sh ${name} results_madrytest/${phase}
fi
