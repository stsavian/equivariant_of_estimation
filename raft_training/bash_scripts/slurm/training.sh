#!/bin/bash
cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$(pwd)
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo/utils
echo $PYTHONPATH

name_training=$1
stage=things
validation=sintel
gpus=0
num_steps=120000
batch_size=5
lr=0.0001
image_size=400 720
wdecay=0.0001


beta=0.0

clip=1.0
imb_train_norm=L2
double_fwd=True
val_freq=5000

absolute_path=/scratch/ssavian/RAFT_trained_models
restore_ckpt=/scratch/ssavian/raft-chairs_mir/raft-chairs_mir.pth

things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
plots_pth=/home/clusterusers/ssavian/plots_slurm
sintel_path=/scratch/ssavian/sintel


/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 400 720 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm L1 \ --chairs_path $chairs_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq --double_fwd $double_fwd \
 --mixed_precision --add_mirroring --no_grad_on_rot_input