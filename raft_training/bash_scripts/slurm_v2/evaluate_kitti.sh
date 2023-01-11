folder_path=/scratch/ssavian/transfer/wacv_supp/baseline/*
#folder_path=/scratch/ssavian/transfer/wacv_supp/FWDG_chairs/*
#folder_path=/scratch/ssavian/transfer/wacv_supp/FWDS_chairs_mir/*
#folder_path=/scratch/ssavian/transfer/wacv_supp/FWDS_chairs_no_mir/*
#folder_path=/scratch/ssavian/transfer/wacv_supp/things/*
#folder_path=/scratch/ssavian/transfer/wacv_supp/missin_nets/*
plots_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_kitti


for filename in $folder_path; do
  model_name=${filename##*/}
  echo $model_name
  path="$filename/$model_name.pth"
  echo $path
  #bash evaluate.sh -a "${model_name}_ev" -b "$path" -c "${plots_pth}" -d "kitti" -e "false"
  bash evaluate.sh -a "${model_name}" -b "$path" -c "${plots_pth}" -d "kitti" -e "true"

done