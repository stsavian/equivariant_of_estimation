#!/bin/bash

stage=chairs
validation=chairs
imb_train_norm=L1
gpus=0
num_steps=120000
num_steps=5
#debug

batch_size=8
lr=0.00025

wdecay=0.0001
clip=1.0
val_freq=5000

########## PATHS
absolute_path=/media/ssavian/Data/EVALUATION/wacv_repo_training
BASE_DATA_PTH=/media/ssavian/Data/DATASETS/
PYTHON_PATH=/home/ssavian/anaconda3/envs/raft_v2/bin/python

results_pth=/media/ssavian/Data/EVALUATION/wacv_repo_training
plots_pth="$(dirname ${results_pth})/$(basename ${results_pth})_plots"

######## DATA PATHS
things_path=${BASE_DATA_PTH}/FlyingThings3D
chairs_path=${BASE_DATA_PTH}/FlyingChairs_release/data
chairs2_path=${BASE_DATA_PTH}/FlyingChairs2
sintel_path=${BASE_DATA_PTH}/sintel_new


##########testing path
stage=chairs
validation=chairs
options="--mixed_precision --add_mirroring --average_loss --double_fwd --train_on_average_epe --no_grad_on_rot_input"
chairs2_direction=fwd
beta=0.6
name_training="raft-chairs-mir_b06_FWDG_o-mean"
echo $name_training
${PYTHON_PATH} -u ../original_repo/train.py \
--things_path $things_path --chairs2_path $chairs2_path --chairs2_flow_direction $chairs2_direction --name  $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 496 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth



model_pth="${absolute_path}/${name_training}/${name_training}.pth"
#ADD TEST NORMAL
cd ../../benchmark_networks/bash
bash evaluate.sh -a "$name_training" -b "$model_pth" -c "$results_pth" -d "sintel" -e "true" -f "${BASE_DATA_PTH}" -g "${PYTHON_PATH}"
#TEST BY AVERAGING O
bash evaluate.sh -a "${name_training}_averaged" -b "$model_pth" -c "$results_pth" -d "sintel" -e "false" -f "${BASE_DATA_PTH}" -g "${PYTHON_PATH}"
cd ../../raft_training/bash
#######################
#THINGS
#######################

stage=things
validation=sintel

imb_train_norm=L1
gpus=0
num_steps=120000
num_steps=8
batch_size=5
lr=0.0001
#image_size=400 720
wdecay=0.0001
clip=1.0
val_freq=5000

results_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O
plots_pth=/home/clusterusers/ssavian/plots_slurm


restore_ckpt=$model_pth
name_training="${name_training}_fine_tuned_on_things"

echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 400 720 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth


model_pth="${absolute_path}/${name_training}/${name_training}.pth"
#ADD TEST NORMAL
cd ../../benchmark_networks/bash
bash evaluate.sh -a "$name_training" -b "$model_pth" -c "$results_pth" -d "sintel" -e "true" -f "${BASE_DATA_PTH}" -g "${PYTHON_PATH}"
#TEST BY AVERAGING O
cd ../../benchmark_networks/bash
bash evaluate.sh -a "${name_training}_averaged" -b "$model_pth" -c "$results_pth" -d "sintel" -e "false" -f "${BASE_DATA_PTH}" -g "${PYTHON_PATH}"
