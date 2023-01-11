#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo
export export PYTHONPATH=$PYTHONPATH:$(pwd)/original_repo/utils
echo $PYTHONPATH

#echo ' TEST chairs FWDS mirror beta 0.3'

#/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b03_FWDS_TEST --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.3 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs mirror'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring

echo 'chairs no mirror'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision


echo 'chairs FWDS mirror beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b0_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.0 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS mirror beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b02_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.2 --no_grad_on_rot_input --double_fwd --clip 1


echo 'chairs FWDS mirror beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b04_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.4 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS mirror beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b06_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.6 --no_grad_on_rot_input --double_fwd --clip 1


echo 'chairs FWDS mirror beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b08_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.8 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS mirror beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b10_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 1.0 --no_grad_on_rot_input --double_fwd --clip 1



echo '***********'
echo 'no mirroring'
echo 'chairs FWDS  beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b00_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.0 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS  beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b02_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.2 --no_grad_on_rot_input --double_fwd --clip 1



echo 'chairs FWDS  beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b04_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --beta 0.4 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS  beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b06_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.6 --no_grad_on_rot_input --double_fwd --clip 1


echo 'chairs FWDS  beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b08_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.8 --no_grad_on_rot_input --double_fwd --clip 1

echo 'chairs FWDS  beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b10_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 1.0 --no_grad_on_rot_input --double_fwd --clip 1



echo 'WITH GRADIENT'


echo 'chairs FWDG mirror beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b0_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.0 --double_fwd --clip 1

echo 'chairs FWDG mirror beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b02_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.2 --double_fwd --clip 1



echo 'chairs FWDG mirror beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b04_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.4 --double_fwd --clip 1

echo 'chairs FWDG mirror beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b06_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.6 --double_fwd --clip 1


echo 'chairs FWDG mirror beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b08_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.8 --double_fwd --clip 1

echo 'chairs FWDG mirror beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b10_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 1.0 --double_fwd --clip 1



echo '***********'
echo 'no mirroring'
echo 'chairs FWDG  beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b00_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.0 --double_fwd --clip 1

echo 'chairs FWDG  beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b02_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.2 --double_fwd --clip 1



echo 'chairs FWDG  beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b04_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --beta 0.4 --double_fwd --clip 1

echo 'chairs FWDG  beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b06_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.6 --double_fwd --clip 1


echo 'chairs FWDG  beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b08_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.8 --double_fwd --clip 1

echo 'chairs FWDG  beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b10_FWDG --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 1.0 --double_fwd --clip 1


echo '*****'
echo 'WITH L2'


echo 'chairs FWDS mirror beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b0_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.0 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS mirror beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b02_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.2 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2



echo 'chairs FWDS mirror beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b04_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.4 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS mirror beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b06_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.6 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2


echo 'chairs FWDS mirror beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b08_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.8 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS mirror beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b10_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 1.0 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2



echo '***********'
echo 'no mirroring'
echo 'chairs FWDS  beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b00_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.0 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS  beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b02_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.2 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2



echo 'chairs FWDS  beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b04_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --beta 0.4 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS  beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b06_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.6 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2


echo 'chairs FWDS  beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b08_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.8 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDS  beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b10_FWDS --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 1.0 --no_grad_on_rot_input --double_fwd --clip 1 --imb_train_norm L2



echo 'WITH GRADIENT'


echo 'chairs FWDG_L2 mirror beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b0_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.0 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2 mirror beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b02_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.2 --double_fwd --clip 1 --imb_train_norm L2



echo 'chairs FWDG_L2 mirror beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b04_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.4 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2 mirror beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b06_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.6 --double_fwd --clip 1 --imb_train_norm L2


echo 'chairs FWDG_L2 mirror beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b08_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 0.8 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2 mirror beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_mir_b10_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --add_mirroring --beta 1.0 --double_fwd --clip 1 --imb_train_norm L2



echo '***********'
echo 'no mirroring'
echo 'chairs FWDG_L2  beta 0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b00_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.0 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2  beta 0.2'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b02_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.2 --double_fwd --clip 1 --imb_train_norm L2



echo 'chairs FWDG_L2  beta 0.4'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b04_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision --beta 0.4 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2  beta 0.6'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b06_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.6 --double_fwd --clip 1 --imb_train_norm L2


echo 'chairs FWDG_L2  beta 0.8'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b08_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 0.8 --double_fwd --clip 1 --imb_train_norm L2

echo 'chairs FWDG_L2  beta 1.0'

/home/ssavian/anaconda3/envs/raft/bin/python -u /home/ssavian/pycharm-projects/original_repo/train.py --name raft-chairs_b10_FWDG_L2 --absolute_path /home/ssavian/training/RAFT_trained_models/july_training --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 8 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --mixed_precision  --beta 1.0 --double_fwd --clip 1 --imb_train_norm L2


