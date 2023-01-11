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


results_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O
plots_pth=/home/clusterusers/ssavian/plots_slurm

########## PATHS
absolute_path=/scratch/ssavian/RAFT_trained_models_v2
######## DATA PATHS
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
sintel_path=/scratch/ssavian/sintel
kitti_path=/scratch/ssavian/KITTI2015
hd1k_path=/scratch/ssavian/HD1K
monkaa_path=/scratch/ssavian/monkaa


######
# mir beta 0.6
####
#options="--mixed_precision --add_mirroring --double_fwd "
options="--mixed_precision  --double_fwd --no_grad_on_rot_input --train_on_average_epe"
restore_ckpt=/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-no-mir-o-mean/raft-chairs-no-mir-o-mean.pth
name_training=RAFT_things-no-mir-average-o_fine_tune_raft-chairs-no-mir-o-mean
beta=0.0
echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 400 720 --wdecay $wdecay --mixed_precision --beta $beta \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth


model_pth="${absolute_path}/${name_training}/${name_training}.pth"
#ADD TEST NORMAL
bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "$name_training" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "false"
#TEST BY AVERAGING O
bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "${name_training}_ev" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "true"
