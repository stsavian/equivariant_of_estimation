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

#options="--mixed_precision --add_mirroring --no_grad_on_rot_input"

gpus=0
num_steps=120000
batch_size=8
lr=0.00025

wdecay=0.0001
clip=1.0
val_freq=5000

##########
absolute_path=/scratch/ssavian/RAFT_trained_models
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
plots_pth=/home/clusterusers/ssavian/plots_slurm
sintel_path=/scratch/ssavian/sintel

##########testing path
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
results_pth=/home/clusterusers/ssavian/plots_slurm/statistical_imbalance_m2/
options="--mixed_precision --add_mirroring --double_fwd  --no_grad_on_rot_input"
imb_train_norm=stat_m2

for beta in 0.4 0.6
do
  name_training="raft-chairs_mir_${beta//./}_FWDS_stat_imb_m2"
  echo $name_training
  /home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 496 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth


/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth $absolute_path/$name_training/$name_training.pth  \
--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
--results_file_pth $results_pth \
--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth

done