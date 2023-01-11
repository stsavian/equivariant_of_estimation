#!/bin/bash
cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
echo $PYTHONPATH

#name_training=raft-chairs_mir_b12_FWDS
stage=chairs
validation=chairs
imb_train_norm=L1
#options="--mixed_precision --add_mirroring --no_grad_on_rot_input"
options="--mixed_precision --add_mirroring --no_grad_on_rot_input --spatium_error"
spatium_norm=2
psi_spatium=0.1

gpus=0
num_steps=120000
batch_size=8
lr=0.00025

wdecay=0.0001
clip=1.0
val_freq=5000

absolute_path=/scratch/ssavian/RAFT_trained_models
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
plots_pth=/home/clusterusers/ssavian/plots_slurm
sintel_path=/scratch/ssavian/sintel

start=$SECONDS
name_training=raft-chairs_mir_spatium_psi_01_L2
beta=0
echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 496 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --val_freq $val_freq  $options --plots_pth $plots_pth --psi_spatium $psi_spatium \
 --spatium_norm $spatium_norm


end=$SECONDS
duration=$(( end - start ))
duration_hours=$((duration/3600))
echo "${name_training}  job id: ${SLURM_JOBID}  ${duration} seconds," " hours ${duration_hours}" >> '/home/clusterusers/ssavian/plots_slurm/logs/jobs_time.txt'

###TESTING
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth "$absolute_path/$name_training/$name_training.pth"  \
--sintel_pth /scratch/ssavian/sintel --results_pth /home/clusterusers/ssavian/plots_slurm/TEST_report_script \
--results_file_pth /home/clusterusers/ssavian/plots_slurm/TEST_report_script \
--model_name $name_training \
--matlab_pth /scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
