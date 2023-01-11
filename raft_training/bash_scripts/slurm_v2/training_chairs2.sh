#!/bin/bash
cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
###TESTING
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts
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
absolute_path=/scratch/ssavian/RAFT_trained_models_v2
######## DATA PATHS
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
chairs2_path=/scratch/ssavian/FlyingChairs2
plots_pth=/home/clusterusers/ssavian/plots_slurm/chairs2_trainings
sintel_path=/scratch/ssavian/sintel


##########testing path
stage=chairs
validation=chairs
results_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O
options="--mixed_precision --add_mirroring --no_grad_on_rot_input --double_fwd --train_on_average_epe"
chairs2_direction=fwd
beta=0.6
name_training="raft-chairs-mir_b06_FWDS_o-mean"
echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --chairs2_path $chairs2_path --chairs2_flow_direction $chairs2_direction --name  $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 496 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth



model_pth="${absolute_path}/${name_training}/${name_training}.pth"
#ADD TEST NORMAL
bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "$name_training" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "true"
#TEST BY AVERAGING O
bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "${name_training}_ev" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "false"
