# Towards Equivariant Optical Flow Estimation with Deep Learning
## _official repository of the paper_
https://openaccess.thecvf.com/content/WACV2023/papers/Savian_Towards_Equivariant_Optical_Flow_Estimation_With_Deep_Learning_WACV_2023_paper.pdf

## Features

- benchmark different networks for optical flow estimation
- analyze the groundtruth data
- re-train RAFT with our strategies, and,
- compare in detail the results
- visualize the estimations and compare them against the groundtruth data
- quickly implement novel metrics and evaluate the results

This repository can be quickly extended to new metrics, data, and networks.

To start with, you can try the bash scripts for evaluation. 
To do that you need to download the testing data, e.g. Sintel; and to install the networks of interest.

You can use the bash scripts under test_networks/bash for automatically testing the networks. The bash script works out of the box for raft, with minimal changes it can be used to evaluate other networks.

## HOW TO START
- install raft anaconda enviroment according to the official repository
- move to benchmark_networks and run pip install -e . (this for a correct python importing)
- download the training and/or testing data (Sintel, KITTI, Monkaa)
- (optional) test your raft installation by using the script under inference_scripts on two images
- move to the bash folder
- edit the PYTHONPATH and DATAPATH in bash/launch_test.sh
- edit the launch test to point to your model path, desired output folder, and testing data
-  run "bash launch_test.sh"
- This will generate some csv files and histograms based on the metrics.

## Re-training RAFT
- move to raft_training
- run pip install -e . (this for a correct python importing)
-  move to raft_training/bash
-  edit the data path, pythonpath, your desidered hyperparameters and so on.
-  run the bash script
-  the script will train RAFT on FlyingChairs and on FlyingThings3D and evaluate the results
-  the results are presented with .csv files and histograms

##DEVELOPMENT
This is the first version of this benchmark, please feel free to open issues and to contact me for any problem or desired improvement. The repository is in develoment and it aims for an easy benchmarking of different optical flow estimators.


## credits
This repository includes the code of the tested networks, and additionaly it uses some (adapted) scrips from https://github.com/philferriere/tfoptflow

