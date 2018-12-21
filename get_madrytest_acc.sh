#!/bin/bash
name=${1}
TESTTYPE=pgd_0031_00078_7_last_transfer
GPU=${GPU:-0}

# self
echo -n "self(white): " && python benchmark/bm.py cifar10_yaml/cifar10_test_1000.yaml --forward-cfg cfgs/cifar10/${name}.yaml --image-path results_madrytest/test/${name}/${name}.yaml/${TESTTYPE} --dataset cifar10 --image-type npy --gpu ${GPU} 2>&1 | grep "test accuracy" | awk '{print $NF}'

# again
echo -n "A': " && python benchmark/bm.py cifar10_yaml/cifar10_test_1000.yaml --forward-cfg cfgs/cifar10/${name}.yaml --image-path results_madrytest/test/${name}/${name}_again.yaml/${TESTTYPE} --dataset cifar10 --image-type npy --gpu ${GPU} 2>&1  | grep "test accuracy" | awk '{print $NF}'

# base
echo -n "baseline: " && python benchmark/bm.py cifar10_yaml/cifar10_test_1000.yaml --forward-cfg cfgs/cifar10/${name}.yaml --image-path results_madrytest/test/${name}/resnet_baseline.yaml/${TESTTYPE} --dataset cifar10 --image-type npy --gpu ${GPU} 2>&1 | grep "test accuracy" | awk '{print $NF}'
