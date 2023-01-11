










##########testing path
stage=chairs
validation=chairs
results_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O
options="--mixed_precision --add_mirroring --no_grad_on_rot_input"
chairs2_direction=fwd
beta=0
name_training="raft-chairs-mir"
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
bash /home/clusterusers/ssavian/raft/project/bash_scripts/slurm_v2/evaluate.sh -a "${name_training}" -b "$model_pth" -c "$results_pth" -d "sintel kaleidoscope" -e "false"
