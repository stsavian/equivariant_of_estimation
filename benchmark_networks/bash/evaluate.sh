#!/bin/bash

#https://unix.stackexchange.com/questions/31414/how-can-i-pass-a-command-line-argument-into-a-shell-script
helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB -c parameterC"
   echo -e "\t-a model_name"
   echo -e "\t-b model_path"
   echo -e "\t-c destination_folder"
   echo -e "\t-d testing_datasets"
   echo -e "\t-e test mean output?"
   echo -e "\t-f datasets base path"
   echo -e "\t-g PYTHONPATH"
   exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:e:f:g:" opt
do
   case "$opt" in
      a ) model_name="$OPTARG" ;;
      b ) model_path="$OPTARG" ;;
      c ) destination_folder="$OPTARG" ;;
      d ) testing_datasets="$OPTARG" ;;
      e ) test_mean_O_2="$OPTARG" ;;
      f ) BASE_PTH="$OPTARG" ;;
      g ) PYTHON_PATH="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$model_name" ] || [ -z "$model_path" ] || [ -z "$destination_folder" ] || [ -z "$testing_datasets" ] || [ -z "$test_mean_O_2" ] || [ -z "$BASE_PTH" ] || [ -z "$PYTHON_PATH" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "INPUT PARAMETERS"
echo "$model_name"
echo "$model_path"
echo "$destination_folder"
echo "$testing_datasets"
echo "$test_mean_O_2"


if [ "$test_mean_O_2" = "true" ]
then
  test_mean="--test_mean"
else
  test_mean=""
fi
### PUTTING THE PARAMETERS INTO EXISTING VARIABLES

# BASE_PTH=/media/ssavian/Data/DATASETS/
# PYTHON_PATH=/home/ssavian/anaconda3/envs/raft_v2/bin/python
##########testing path
things_path=${BASE_PTH}FlyingThings3D
chairs_path=${BASE_PTH}FlyingChairs_release/data
sintel_path=${BASE_PTH}sintel_new
kitti_path=${BASE_PTH}KITTI2015
hd1k_path=${BASE_PTH}HD1K
monkaa_path=${BASE_PTH}monkaa



$PYTHON_PATH -u ../test_all_nets/test_raft.py \
--train_pth $model_path  --sintel_pth $sintel_path --results_pth $destination_folder  --results_file_pth $destination_folder \
--model_name $model_name  --monkaa_pth $monkaa_path --hd1k_pth $hd1k_path  --kitti_pth $kitti_path --testing_datasets $testing_datasets $test_mean \


