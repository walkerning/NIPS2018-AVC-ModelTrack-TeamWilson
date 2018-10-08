#!/bin/bash
set -e

gpu=${GPU:-0}
test_name=${TEST:-"pgd_bigiter"}
tmp_model_cfg=${TMP_MODEL_CFG:-$(tempfile)}
tmp_test_log=${TMP_TEST_LOG:-$(tempfile)}
echo "use temp model cfg file: ${tmp_model_cfg}; temp test log file: ${tmp_test_log}"
for name in $@; do
    echo "gen adv on model $name"
    cp cfgs/${name}.yaml ${tmp_model_cfg}
    FMODEL_MODEL_CFG=${tmp_model_cfg} python main_gen_ch.py ~/yml_files/labels.yml --gpu ${gpu} --save transfer/test500_${name} -t ${test_name} >transfer/logs/${name}_{test_name}.log 2>&1  
    echo "test the advs gened by model $name on resnet18"
    cp cfgs/resnet18.yaml ${tmp_model_cfg}
    FMODEL_MODEL_CFG=${tmp_model_cfg} python main_gen_ch.py ~/yml_files/labels.yml --image-path /home/foxfi/bm/iterative_transfer_untargeted_attack_baseline/transfer/test500_${name}/${test_name}/ --gpu ${gpu} > tmp_test_log
    acc=$(grep "test accuracy" tmp_test_log | awk '{print $NF}')
    echo "Test on resnet18 acc: ${acc}" >> transfer/logs/${name}_${test_name}.log
done
