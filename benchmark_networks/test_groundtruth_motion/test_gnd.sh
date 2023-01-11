#!/bin/bash

DIR=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)
parentdir="$(dirname "$DIR")"
#echo $parentdir

#set PYTHONPATH ='parentdir'#"$("$parentdir")"
export PYTHONPATH=$PYTHONPATH:$parentdir
echo $PYTHONPATH

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/PYCHARM_PROJECTS/test_raft/test_groundtruth_motion/dataset_iterators.py \
 --chairs_pth /home/ssavian/training/FlyingChairs_release/data --things_pth /home/ssavian/training/FlyingThings3D  \
 --sintel_pth /home/ssavian/training/sintel --matlab_pth /home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples \
 --matlab_equiv_pth /home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples \
 --results_pth /home/ssavian/training/plots_ironspeed/GND_stats_v1 --monkaa_pth /home/ssavian/training/monkaa/ \
--hd1k_pth /home/ssavian/training/HD1K --kitti_pth /home/ssavian/training/KITTI2015 --chairs2_pth /home/ssavian/training/FlyingChairs2 \
 --chairsOcc_pth /home/ssavian/training/FlyingChairsOcc