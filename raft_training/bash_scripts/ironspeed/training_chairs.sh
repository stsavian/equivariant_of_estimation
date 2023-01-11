#!/bin/bash
cd /home/ssavian/pycharm-projects
export PYTHONPATH=$PYTHONPATH:$/home/ssavian/pycharm-projects
export PYTHONPATH=$PYTHONPATH:$/home/ssavian/pycharm-projects/original_repo
export PYTHONPATH=$PYTHONPATH:$/home/ssavian/pycharm-projects//original_repo/utils
echo $PYTHONPATH

name_training=raft-chairs_mir_b12_FWDS
stage=chairs
validation=chairs
beta=0.6
imb_train_norm=L1
#options="--mixed_precision --add_mirroring --no_grad_on_rot_input"
options="--mixed_precision --add_mirroring --no_grad_on_rot_input --double_fwd"
gpus="0 1"
num_steps=120000
batch_size=8
lr=0.00025

wdecay=0.0001
clip=1.0
val_freq=5000

absolute_path=/home/ssavian/training/RAFT_trained_models
things_path=/home/ssavian/training/FlyingThings3D
chairs_path=/home/ssavian/training/FlyingChairs_release/data
plots_pth=/home/ssavian/training/plots_ironspeed
sintel_path=/home/ssavian/training/sintel

echo $name_training
/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 384 512 --wdecay $wdecay --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth





