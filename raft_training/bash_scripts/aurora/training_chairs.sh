#!/bin/bash
#cd /home/clusterusers/ssavian/raft/project
#export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
#export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
#export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
###TESTING
#export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts
echo $PYTHONPATH

#name_training=raft-chairs_mir_b12_FWDS
stage=chairs
validation=chairs
#beta=1.2
imb_train_norm=L1
#options="--mixed_precision --add_mirroring --no_grad_on_rot_input"

gpus=0
num_steps=120000
#num_steps=10
batch_size=8
lr=0.00025

wdecay=0.0001
clip=1.0
val_freq=5000

########## PATHS
absolute_path=/media/ssavian/Data/TEST/RAFT_test_training
######## DATA PATHS
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/media/ssavian/Data/DATASETS/FlyingChairs_release/data
chairs2_path=/scratch/ssavian/FlyingChairs2
plots_pth=/media/ssavian/Data/TEST/test_training
sintel_path=/media/ssavian/Data/DATASETS/sintel


##########testing path
stage=chairs
validation=chairs
results_pth=/media/ssavian/Data/TEST/test_training
options="--mixed_precision --add_mirroring --no_grad_on_rot_input"
chairs2_direction=fwd
beta=0
name_training="raft-chairs-mir-TEST"
echo $name_training
/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/CODE/raft_training/original_repo/train.py \
--things_path $things_path --chairs2_path $chairs2_path --chairs2_flow_direction $chairs2_direction --name  $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 496 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth



# model_pth="${absolute_path}/${name_training}/${name_training}.pth"
# #ADD TEST NORMAL
# bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "$name_training" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "true"
# #TEST BY AVERAGING O
# bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "${name_training}" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "false"
