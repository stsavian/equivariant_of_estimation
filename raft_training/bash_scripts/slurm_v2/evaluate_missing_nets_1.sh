


#folder_path=/scratch/ssavian/transfer/benchmarking/*
#plots_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O_baseline
#
#for filename in $folder_path; do
#  model_name=${filename##*/}
#  echo $model_name
#  path="$filename/$model_name.pth"
#  echo $path
#  bash evaluate.sh -a "${model_name}_ev" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "false"
#  bash evaluate.sh -a "${model_name}" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "true"
#
#done


folder_path=/home/clusterusers/ssavian/raft/project/original_repo/models/*chairs*
plots_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O_baseline

for filename in $folder_path; do
  model_name=${filename##*/}
  echo $model_name
  path="$filename/$model_name.pth"
  echo $path
  bash evaluate.sh -a "${model_name}_paper_ev" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "false"
  bash evaluate.sh -a "${model_name}_paper" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "true"

done

folder_path=/home/clusterusers/ssavian/raft/project/original_repo/models/*things*
plots_pth=/home/clusterusers/ssavian/plots_slurm/RAFT_averaged_O_baseline

for filename in $folder_path; do
  model_name=${filename##*/}
  echo $model_name
  path="$filename/$model_name.pth"
  echo $path
  bash evaluate.sh -a "${model_name}_paper_ev" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "false"
  bash evaluate.sh -a "${model_name}_paper" -b "$filename" -c "${plots_pth}" -d "sintel kaleidoscope" -e "true"

done