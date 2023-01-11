
#!/bin/bash
#cd /home/clusterusers/ssavian/raft/project
export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts

echo $PYTHONPATH

#general paths
sintel_pth=/scratch/ssavian/sintel
matlab_sign_imbalance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/scratch/ssavian/img_formation_datasets/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples

#results_pth =folder with
results_pth=/home/clusterusers/ssavian/plots_slurm/FWDS_no_mir_beta_maj_10
results_summary_pth=/home/clusterusers/ssavian/plots_slurm/FWDS_no_mir_beta_maj_10_summary
results_plots_pth=/home/clusterusers/ssavian/plots_slurm/FWDS_no_mir_beta_maj_10_plots

FWDS_mir_root=/scratch/ssavian/RAFT_trained_models/*__no_mir_*
for filename in $FWDS_mir_root; do
  model_name=${filename##*/}
  echo $model_name
  path="$filename/$model_name.pth"
  echo $path
  /home/clusterusers/ssavian/.conda/envs/raft/bin/python -u /home/clusterusers/ssavian/raft/TEST_scripts/test_raft.py \
--train_pth $path  --sintel_pth $sintel_pth  --results_pth $results_pth  \
--results_file_pth $results_pth  --model_name $model_name --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth
done

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/generate_summary_csv_from_results.py \
 --folder_pth $results_pth --result_summary_pth $results_summary_pth

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/merge_dataset_csv.py --summary_pth \
$results_summary_pth --merged_summary_pth ${results_summary_pth}/merged

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/local_scripts/full_frame_stats.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth

/home/clusterusers/ssavian/.conda/envs/raft/bin/python ../../compare_networks/local_scripts/masked_stats_for_hist.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth