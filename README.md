This is the official repository of "Towards Equivariant"

With this repository you can:
-benchmark different networks for optical flow estimation
-analyze the groundtruth data
-re-train RAFT with our strategies, and,
-compare in detail the results
-visualize the estimations and compare them against the groundtruth data
-quickly implement novel metrics and evaluate the results

This repository can be quickly extended to new metrics, data, and networks.


To start with, you can try the bash scripts for raft:
To do that you need to:

-install raft conda enviroment according to the official repository
-move to benchmark_networks and run pip install -e . (this for a correct importing)
-download the training and/or testing data
-test your raft installation by using the script under inference_scripts
-move to the bash folder
-edit the PYTHONPATH and DATA PATH in bash/evaluate.sh
-edit the launch test to point to your model path, desired output folder, testing data
- move to the bash folder and run "bash launch_test.sh"
-once you run different models you can compare them by running "bash make_summary.sh". this will generate some csv files and histograms based on the metrics.

-mention the output file structure

IMPLEMENTED:
-RAFT evaluation

TODO
-RAFT training
-groundtruth data
-simple example
-jupyter notebooks with example/ google colab


please feel free to open issues and to extend this repository!

