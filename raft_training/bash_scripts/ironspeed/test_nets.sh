#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo/utils
echo $PYTHONPATH


tmux pipe-pane 'cat >/raft/RAFT_trained_models/logs/raft-things.txt'
start=$SECONDS
echo 'things mirroring p 0.5'
/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/test_epe_and_imbalance/test_raft.py \
--train_pth /home/ssavian/training/RAFT_trained_models/first_run/raft-chairs.pth \
--sintel_pth /home/ssavian/training/sintel --plots_pth /home/ssavian/training/plots_ironspeed \
 --note raft-chairs --mode clean

end=$SECONDS
duration=$(( end - start ))
duration_hours=$((num_tasks/3600))
echo " raft-things $duration seconds," " hours $duration_hours" >> '/raft/RAFT_trained_models/logs/training_time_logs.txt'