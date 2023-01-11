
# /home/ssavian/anaconda3/envs/raft_v2/bin/python /home/ssavian/CODE/test_raft_aurora/visualization/dataset_visualization/visualize_sintel.py \
# --results_pth "/media/ssavian/Data1/EVALUATION/WACV_visualization" --sintel_pth /media/ssavian/Data/DATASETS/sintel \
# --train_pth '/media/ssavian/Data/BACKUP_v2/BACKUP_models/RAFT_trained_models_v2/RAFT-things-mir_b10_FWDS_o-mean/RAFT-things-mir_b10_FWDS_o-mean.pth' \
# --mode "final"


/home/ssavian/anaconda3/envs/raft_v2/bin/python /home/ssavian/CODE/test_raft_aurora/visualization/dataset_visualization/visualize_sintel.py \
--results_pth "/media/ssavian/Data1/EVALUATION/WACV_visualization_fixed" --sintel_pth /media/ssavian/Data/DATASETS/sintel \
--train_pth '/media/ssavian/Data/BACKUP_v2/BACKUP_models/RAFT_trained_models_v2/RAFT-things-mir_b10_FWDS_o-mean/RAFT-things-mir_b10_FWDS_o-mean.pth' \
--mode "final" --normalization "fixed" --norm_values 0 13