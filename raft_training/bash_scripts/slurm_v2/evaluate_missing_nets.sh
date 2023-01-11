#bash evaluate.sh -a "raft-chairs-mir" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir/raft-chairs-mir.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "false"
#
#bash evaluate.sh -a "raft-chairs-mir-o-mean" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir-o-mean/raft-chairs-mir-o-mean.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "false"
#
#bash evaluate.sh -a "raft-chairs-mir_b10_FWDS_o-mean" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir_b10_FWDS_o-mean/raft-chairs-mir_b10_FWDS_o-mean.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "false"

#bash evaluate.sh -a "RAFT_things-mir_ev_m" -b "/scratch/ssavian/RAFT_trained_models_v2/RAFT_things-mir/RAFT_things-mir.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "true"
#
#bash evaluate.sh -a "RAFT_things-O-mean_ev_m" -b "/scratch/ssavian/RAFT_trained_models_v2/RAFT_things-O-mean/RAFT_things-O-mean.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "true"
#
#bash evaluate.sh -a "RAFT-things-mir_b10_FWDS_o-mean_ev_m" -b "/scratch/ssavian/RAFT_trained_models_v2/RAFT-things-mir_b10_FWDS_o-mean/RAFT-things-mir_b10_FWDS_o-mean.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel kaleidoscope" -e "true"

#bash evaluate.sh -a "raft-chairs-mir_ev_mTEST5" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir/raft-chairs-mir.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "kaleidoscope" -e "false"
#bash evaluate.sh -a "raft-chairs-mirTEST5" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir/raft-chairs-mir.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "kaleidoscope" -e "true"
#bash evaluate.sh -a "raft-chairs-mir" -b "/scratch/ssavian/RAFT_trained_models_v2/raft-chairs-mir/raft-chairs-mir.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "kaleidoscope" -e "true"

#bash evaluate.sh -a "things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss_ev" -b "/scratch/ssavian/RAFT_trained_models/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel" -e "false"
bash evaluate.sh -a "things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss" -b "/scratch/ssavian/RAFT_trained_models/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss/things_mir_b10_fine_tune_raft-chairs_mir_10_FWDG_averaged_loss.pth" -c "/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O" -d "sintel" -e "true"