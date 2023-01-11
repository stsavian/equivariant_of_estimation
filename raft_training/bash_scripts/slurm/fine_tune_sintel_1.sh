
#!/bin/bash
cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
echo $PYTHONPATH
###TESTING
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts


stage=sintel
validation=sintel
beta=0.0
imb_train_norm=L1


gpus=0
num_steps=120000
batch_size=5
lr=0.0001
#image_size=400 720
wdecay=0.00001
clip=1.0
val_freq=5000
gamma=0.85

absolute_path=/scratch/ssavian/RAFT_trained_models
results_pth=/home/clusterusers/ssavian/plots_slurm/SINTEL_fine_tune/
plots_pth=$results_pth


##########testing path
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
sintel_path=/scratch/ssavian/sintel
kitti_path=/scratch/ssavian/KITTI2015
hd1k_path=/scratch/ssavian/HD1K

#######################
#### TESTING FWDS 1;1
#######################
#options="--mixed_precision --add_mirroring --no_grad_on_rot_input --double_fwd"
#restore_ckpt=/scratch/ssavian/RAFT_trained_models/things_beta_10_fine_tune_raft-chairs_mir_b10_FWDS/things_beta_10_fine_tune_raft-chairs_mir_b10_FWDS.pth
#name_training=sintel_b10_fine_tune_chairs_b10_things_b10
#beta=1.0
#
#echo $name_training
#/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
#--things_path $things_path --name $name_training \
#--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
#--batch_size $batch_size --lr $lr --image_size 368 768 --wdecay $wdecay --mixed_precision --beta $beta --gamma $gamma \
#--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path --kitti_path $kitti_path --hd1k_path $hd1k_path \
# --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth
#

#######################
#### TESTING FWDS 1;1
#######################
options="--mixed_precision --add_mirroring --no_grad_on_rot_input --double_fwd"
restore_ckpt=/scratch/ssavian/RAFT_trained_models/things_beta_10_fine_tune_raft-chairs_mir_b10_FWDS/things_beta_10_fine_tune_raft-chairs_mir_b10_FWDS.pth
name_training=sintel_b05_fine_tune_chairs_b10_things_b10
beta=0.5

echo $name_training
/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
--things_path $things_path --name $name_training \
--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
--batch_size $batch_size --lr $lr --image_size 368 768 --wdecay $wdecay --mixed_precision --beta $beta --gamma $gamma \
--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path --kitti_path $kitti_path --hd1k_path $hd1k_path \
 --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth

 /home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth $absolute_path/$name_training/$name_training.pth  \
--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
--results_file_pth $results_pth \
--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth


#######################
#### TESTING FWDS 1.2;1.2
#######################
#options="--mixed_precision --add_mirroring --no_grad_on_rot_input --double_fwd"
#restore_ckpt=/scratch/ssavian/RAFT_trained_models/things_beta_12_fine_tune_raft-chairs_mir_b12_FWDS/things_beta_12_fine_tune_raft-chairs_mir_b12_FWDS.pth
#name_training=sintel_b12_fine_tune_chairs_b12_things_b12
#beta=1.2
#
#echo $name_training
#/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
#--things_path $things_path --name $name_training \
#--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
#--batch_size $batch_size --lr $lr --image_size 368 768 --wdecay $wdecay --mixed_precision --beta $beta --gamma $gamma \
#--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path --kitti_path $kitti_path --hd1k_path $hd1k_path \
# --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth
#
# /home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
#--train_pth $absolute_path/$name_training/$name_training.pth  \
#--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
#--results_file_pth $results_pth \
#--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
#$matlab_equivariance_pth

#######################
#### TESTING FWDG 0.4;0.4
#######################
#options="--mixed_precision --add_mirroring --double_fwd"
#restore_ckpt=/scratch/ssavian/RAFT_trained_models/things_beta_04_fine_tune_raft_chairs_mir_b04_FWDG/things_beta_04_fine_tune_raft_chairs_mir_b04_FWDG.pth
#name_training=sintel_b04_fine_tune_chairs_b04_things_b04_FWDG
#beta=0.4
#
#echo $name_training
#/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/project/original_repo/train.py \
#--things_path $things_path --name $name_training \
#--absolute_path $absolute_path --stage $stage --validation $validation --gpus $gpus --num_steps $num_steps \
#--batch_size $batch_size --lr $lr --image_size 368 768 --wdecay $wdecay --mixed_precision --beta $beta --gamma $gamma \
#--clip $clip --imb_train_norm $imb_train_norm --chairs_path $chairs_path --kitti_path $kitti_path --hd1k_path $hd1k_path \
# --sintel_path $sintel_path --restore_ckpt $restore_ckpt --val_freq $val_freq  $options --plots_pth $plots_pth
#
#
#/home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
#--train_pth $absolute_path/$name_training/$name_training.pth  \
#--sintel_pth /scratch/ssavian/sintel --results_pth $results_pth \
#--results_file_pth $results_pth \
#--model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
#$matlab_equivariance_pth

