#!/bin/bash
#cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts/compare_networks
echo $PYTHONPATH

DIR=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)
parentdir="$(dirname "$DIR")"
#echo $parentdir

#set PYTHONPATH ='parentdir'#"$("$parentdir")"
export PYTHONPATH=$PYTHONPATH:$parentdir
echo $PYTHONPATH
echo path
echo $BASH_SOURCE[0]
#general paths
sintel_pth=/scratch/ssavian/sintel
results_pth=/home/clusterusers/ssavian/plots_slurm/TEST_spatium
results_file_pth=/home/clusterusers/ssavian/plots_slurm/TEST_spatium
matlab_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples


#trained models
baseline_pth=/scratch/ssavian/RAFT_trained_models/things_fine_tune_raft-chairs_mir/things_fine_tune_raft-chairs_mir.pth
train_pth=$baseline_pth
model_name=things_mir
/home/clusterusers/ssavian/.conda/envs/raft/bin/python test_raft.py --matlab_pth $matlab_pth --results_file_pth $results_file_pth \
--results_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name


psi0path=/scratch/ssavian/RAFT_trained_spatium/things_spatium_psi_00/things_spatium_psi_00.pth
psi1path=/scratch/ssavian/RAFT_trained_spatium/things_spatium_psi_01/things_spatium_psi_01.pth
psi2path=/scratch/ssavian/RAFT_trained_spatium/things_spatium_psi_10/things_spatium_psi_10.pth


train_pth=$psi0path
model_name=things_spatium_psi_00
/home/clusterusers/ssavian/.conda/envs/raft/bin/python test_raft.py --matlab_pth $matlab_pth --results_file_pth $results_file_pth \
--results_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name

train_pth=$psi01path
model_name=things_spatium_psi_01
/home/clusterusers/ssavian/.conda/envs/raft/bin/python test_raft.py --matlab_pth $matlab_pth --results_file_pth $results_file_pth \
--results_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name

train_pth=$psi2path
model_name=things_spatium_psi_10
/home/clusterusers/ssavian/.conda/envs/raft/bin/python test_raft.py --matlab_pth $matlab_pth --results_file_pth $results_file_pth \
--results_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name


###merge the stuff
/home/clusterusers/ssavian/.conda/envs/raft/bin/python compare_networks/generate_summary_csv_from_results --results_pth $results_pth \
--results_file_pth $results_file_pth

/home/clusterusers/ssavian/.conda/envs/raft/bin/python compare_networks/merge_dataset_csv  --summary_pth \
$results_pth --merged_summary_pth "$results_file_pth/summary"