#!/bin/bash
#SBATCH --job-name=things_FWDG
#SBATCH --output=/home/clusterusers/ssavian/plots_slurm/logs/job-%j.out
#SBATCH --error=/home/clusterusers/ssavian/plots_slurm/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ssavian@unibz.it
#SBATCH --partition gpu
#SBATCH --gres=gpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=16

module load cuda-10.2
module load anaconda3
source /opt/packages/anaconda3/etc/profile.d/conda.sh
hostname
nvidia-smi

name=""
#start=$SECONDS
#bash ./slurm/training_things.sh $name
#bash ./slurm/training_chairs_spatium.sh $name
#bash ./slurm/training_chairs.sh $name
#bash ./slurm/training_things_spatium.sh $name
#bash ./slurm/fine_tune_sintel.sh
#bash ./slurm/fine_tune_sintel_1.sh
bash ./slurm/test_network.sh
#end=$SECONDS
#duration=$(( end - start ))
#duration_hours=$((duration/3600))
#echo "${name}  job id: ${SLURM_JOBID}  $duration seconds," " hours $duration_hours" >> '/home/clusterusers/ssavian/plots_slurm/logs/jobs_time.txt'