#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts
echo $PYTHONPATH


results_pth=/home/clusterusers/ssavian/plots_slurm/test_mean_pred
results_summary_pth="$results_pth"_summary
results_plots_pth="$results_pth"_plots
plots_pth=$results_pth

##########testing path
things_path=/scratch/ssavian/FlyingThings3D
chairs_path=/scratch/ssavian/FlyingChairs_release/data
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
sintel_path=/scratch/ssavian/sintel
kitti_path=/scratch/ssavian/KITTI2015
hd1k_path=/scratch/ssavian/HD1K
monkaa_path=/scratch/ssavian/monkaa

#testing_datasets='sintel kaleidoscope repeated_frames'
testing_datasets='sintel'
#root_folders="/scratch/ssavian/transfer/MODELS_chairs/FWDS_mir /scratch/ssavian/transfer/MODELS_chairs/FWDS_no_mir /scratch/ssavian/transfer/MODELS_things/ /scratch/ssavian/transfer/MODELS_chairs/FWDG_mir"
#root_folders="/scratch/ssavian/transfer/MODELS_baseline"
folder_pth='/scratch/ssavian/RAFT_trained_models/'
#for folder_pth in $root_folders; do
#  echo $folder_pth

baseline_pth='/scratch/ssavian/transfer/MODELS_baseline/raft_things_mir_batch5/raft_things_mir_batch5.pth'
FWDG_b10_things_pth='/scratch/ssavian/RAFT_trained_models/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss.pth'

myArray=($baseline_pth $FWDG_b10_things_pth)
for str in ${myArray[@]}; do
  #echo  $filename
  path=$str
  name_training="$(basename $path)"
  train_pth=$path
  echo $train_pth
   /home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
  --train_pth $train_pth  \
  --sintel_pth $sintel_path --results_pth $results_pth \
  --results_file_pth $results_pth \
  --model_name $name_training --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
  $matlab_equivariance_pth  --monkaa_pth $monkaa_path --hd1k_pth $hd1k_path  --kitti_pth $kitti_path --testing_datasets $testing_datasets
done
#done

echo $results_summary_pth
#echo $(pwd/../../compare_networks/)
(cd ../../compare_networks; echo $(pwd))
/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/generate_summary_csv_from_results.py \
 --folder_pth $results_pth --result_summary_pth $results_summary_pth

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/merge_dataset_csv.py --summary_pth \
$results_summary_pth --merged_summary_pth ${results_summary_pth}/merged

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/local_scripts/full_frame_stats.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/local_scripts/masked_stats_for_hist.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth