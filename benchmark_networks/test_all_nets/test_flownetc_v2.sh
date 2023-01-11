#!/bin/bash
DIR=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd)
parentdir="$(dirname "$DIR")"


export PYTHONPATH=$PYTHONPATH:$parentdir
export PYTHONPATH=$PYTHONPATH:$parentdir/compare_networks
echo $PYTHONPATH
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
   exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:e:" opt
do
   case "$opt" in
      a ) model_name="$OPTARG" ;;
      b ) model_path="$OPTARG" ;;
      c ) destination_folder="$OPTARG" ;;
      d ) testing_datasets="$OPTARG" ;;
      e ) test_mean_O_2="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$model_name" ] || [ -z "$model_path" ] || [ -z "$destination_folder" ] || [ -z "$testing_datasets" ]|| [ -z "$test_mean_O_2" ]
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



##########testing path
# things_path=/scratch/ssavian/FlyingThings3D
# chairs_path=/scratch/ssavian/FlyingChairs_release/data
matlab_sign_imbalance_pth=/home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis1000_samples
matlab_equivariance_pth=/home/ssavian/training/img_formation_mat/TCSVT_repository/balanced_dataset_textured_std_60_radius_50_limit_dis_for_equivariance1000_samples
sintel_path=/home/ssavian/training/sintel
kitti_path=/home/ssavian/training/KITTI2015
hd1k_path=/home/ssavian/HD1K
monkaa_path=/home/ssavian/training/monkaa/

##FIXING THE PYTHONPATH
#cd /home/clusterusers/ssavian/raft/project
# export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project
# export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo
# export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/project/original_repo/utils
# echo $PYTHONPATH
# ###TESTING
# export PYTHONPATH=$PYTHONPATH:$/home/clusterusers/ssavian/raft/TEST_scripts

/home/ssavian/anaconda3/envs/FNC_env_p35/bin/python -u test_FNC.py  \
--train_pth $model_path  --sintel_pth $sintel_path --results_pth $destination_folder  --results_file_pth $destination_folder \
--model_name $model_name --matlab_pth $matlab_sign_imbalance_pth --matlab_equivariance_pth \
$matlab_equivariance_pth  --monkaa_pth $monkaa_path --hd1k_pth $hd1k_path  --kitti_pth $kitti_path --testing_datasets $testing_datasets $test_mean \


