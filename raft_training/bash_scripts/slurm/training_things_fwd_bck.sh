#!/bin/bash
cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
echo $PYTHONPATH
###TESTING
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts

stage=things
validation=sintel
beta=0.0
imb_train_norm=L1


gpus=0
num_steps=120000
batch_size=5
lr=0.0001
#image_size=400 720
wdecay=0.0001
clip=1.0
val_freq=5000

absolute_path=/scratch/ssavian/RAFT_trained_models
results_pth=/home/clusterusers/ssavian/plots_slurm/september_trainings/
plots_pth=/home/clusterusers/ssavian/plots_slurm


##########testing path
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
sintel_path=/scratch/ssavian/sintel

#################
# ONLY FOWRARD FLOW
################
options="--mixed_precision --add_mirroring"
things_flow_direction=into_future
restore_ckpt=/scratch/ssavian/RAFT_trained_models/raft_mir_batch_8/raft_mir_batch_8.pth
name_training=things_fine_tune_raft-chairs_mir_only_FWD_FLOW
beta=0.6

echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 400 720 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth \
 --things_flow_direction $things_flow_direction


/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth $absolute_path/$name_training/$name_training.pth  \
--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
--results_file_pth $results_pth \
--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth
#################
# ONLY PAST FLOW
################
options="--mixed_precision --add_mirroring"
things_flow_direction=into_past
restore_ckpt=/scratch/ssavian/RAFT_trained_models/raft_mir_batch_8/raft_mir_batch_8.pth
name_training=things_fine_tune_raft-chairs_mir_only_PAST_FLOW
beta=0.6

echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 400 720 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth \
 --things_flow_direction $things_flow_direction


/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth $absolute_path/$name_training/$name_training.pth  \
--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
--results_file_pth $results_pth \
--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth
