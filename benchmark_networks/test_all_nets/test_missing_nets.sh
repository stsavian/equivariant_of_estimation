#!/bin/bash

DIR=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)
parentdir="$(dirname "$DIR")"
#echo $parentdir

#set PYTHONPATH ='parentdir'#"$("$parentdir")"
export PYTHONPATH=$PYTHONPATH:$parentdir
export PYTHONPATH=$PYTHONPATH:$parentdir/compare_networks

echo $PYTHONPATH

#general paths
sintel_pth=/home/ssavian/training/sintel
results_pth=/home/ssavian/training/plots_ironspeed/TEST_script_new_datasets
result_summary_pth=/home/ssavian/training/plots_ironspeed/TEST_monkaa_script_summary
matlab_pth=/home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
monkaa_pth=/home/ssavian/training/monkaa/
hd1k_pth=/home/ssavian/training/HD1K
kitti_pth=/home/ssavian/training/KITTI2015



#train_things_no_mir_pth=/home/ssavian/training/transfer/things_no_mir_fine_tune_RAFT_chairs_no_mirror.pth
#train_pth=$train_things_no_mir_pth
#model_name=RAFT_things_no_mir_pth
train_pth=/home/ssavian/training/trained_models/main_mirror_baseline/model_best.pth.tar
model_name=FlowNetC-M
/home/ssavian/anaconda3/envs/FNC_env_p35/bin/python test_FNC.py --matlab_pth $matlab_pth --results_pth $results_pth \
--results_file_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name \
--matlab_equivariance_pth $matlab_equivariance_pth --monkaa_pth $monkaa_pth --hd1k_pth $hd1k_pth --kitti_pth $kitti_pth

#/home/ssavian/anaconda3/envs/raft/bin/python test_raft.py --matlab_pth $matlab_pth --results_pth $results_pth \
#--results_file_pth $results_pth --sintel_pth $sintel_pth --train_pth $train_pth --model_name $model_name

/home/ssavian/anaconda3/envs/raft/bin/python ../compare_networks/generate_summary_csv_from_results.py --result_summary_pth \
$result_summary_pth --folder_pth $results_pth

/home/ssavian/anaconda3/envs/raft/bin/python ../compare_networks/merge_dataset_csv.py --summary_pth \
$result_summary_pth --merged_summary_pth ${result_summary_pth}/merged