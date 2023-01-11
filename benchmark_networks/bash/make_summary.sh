PYTHON_PATH=$1
results_pth=$2

# PYTHON_PATH=/home/ssavian/anaconda3/envs/raft_v2/bin/python
# results_pth=/media/ssavian/Data/TEST/wacv_repo_v3/

results_summary_pth="$(dirname ${results_pth})/$(basename ${results_pth})_summary"
results_plots_pth="$(dirname ${results_pth})/$(basename ${results_pth})_plots"

$PYTHON_PATH ../compare_networks/generate_summary_csv_from_results.py \
 --folder_pth $results_pth --result_summary_pth $results_summary_pth
echo 1
$PYTHON_PATH ../compare_networks/local_scripts_v2/full_frame_stats.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth
echo 2
$PYTHON_PATH ../compare_networks/local_scripts/masked_stats_for_hist.py --summary_pth \
$results_summary_pth --plots_pth $results_plots_pth
echo 3

# merge_labels="EPE EPE_180 I_L2_m1 cos_sim"
# full_frame_merged_pth="${results_plots_pth}_merged"
# $PYTHON_PATH "../compare_networks/local_scripts_v2/merge_full_frame.py"\
#   --full_frame_merged_pth "$full_frame_merged_pth" --full_frame_pth "$results_plots_pth" --columns $merge_labels

