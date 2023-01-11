
# /home/ssavian/anaconda3/envs/raft_v2/bin/python /home/ssavian/CODE/test_raft_aurora/visualization/dataset_visualization/visualize_sintel.py \
# --results_pth "/media/ssavian/Data1/EVALUATION/WACV_visualization" --sintel_pth /media/ssavian/Data/DATASETS/sintel \
# --train_pth /media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/things_fine_tune_raft-chairs_mir/things_fine_tune_raft-chairs_mir.pth \
# --mode "final"

# /home/ssavian/anaconda3/envs/raft_v2/bin/python /home/ssavian/CODE/test_raft_aurora/visualization/dataset_visualization/visualize_sintel.py \
# --results_pth "/media/ssavian/Data1/EVALUATION/WACV_visualization" --sintel_pth /media/ssavian/Data/DATASETS/sintel \
# --train_pth '/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/raft_things_original_param/120000_raft_things_original_param.pth' \
# --mode "final"


/home/ssavian/anaconda3/envs/raft_v2/bin/python /home/ssavian/CODE/test_raft_aurora/visualization/dataset_visualization/visualize_sintel.py \
--results_pth "/media/ssavian/Data1/EVALUATION/WACV_visualization_fixed" --sintel_pth /media/ssavian/Data/DATASETS/sintel \
--train_pth '/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/raft_things_original_param/120000_raft_things_original_param.pth' \
--mode "final" --normalization "fixed" --norm_values 0 13