export PATH=$PATH:${pwd}
# echo $PATH

BASE_DATA_PTH=/media/ssavian/Data/DATASETS/
PYTHON_PATH=/home/ssavian/anaconda3/envs/raft_v2/bin/python
RESULTS_PTH=/media/ssavian/Data/TEST/wacv_repo_v4/

model_path=/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/things_fine_tune_raft-chairs_mir/things_fine_tune_raft-chairs_mir.pth
/bin/bash evaluate.sh -a "raft_test_bash3" -b "${model_path}" -c "${RESULTS_PTH}" -d "sintel" -e "false" -f "${BASE_DATA_PTH}" -g "${PYTHON_PATH}"

# /bin/bash evaluate.sh -a "raft_test_bash1" -b "/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/things_fine_tune_raft-chairs_mir/things_fine_tune_raft-chairs_mir.pth" -c "/media/ssavian/Data/TEST/wacv_repo_v3/" -d "sintel" -e "false"

# /bin/bash evaluate.sh -a "raft_test_bash2" -b "/media/ssavian/Data/BACKUP_v2/BACKUP_models/transfer/MODELS_things/things_fine_tune_raft-chairs_mir/things_fine_tune_raft-chairs_mir.pth" -c "/media/ssavian/Data/TEST/wacv_repo_v3/" -d "sintel" -e "false"

echo "evaluation completed"
/bin/bash make_summary.sh ${PYTHON_PATH} ${RESULTS_PTH}
echo "summary and plots generated"
