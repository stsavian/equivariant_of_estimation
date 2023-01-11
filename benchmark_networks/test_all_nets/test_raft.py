print(__file__)
import inference_scripts.inference_RAFT as inference_RAFT
import argparse
import dataset_iterators.sintel_iterator as sintel_iterator
import dataset_iterators.matlab_dataset_iterator as matlab_iterator
import dataframe_operations.save_dataframe as save_dataframes
import os
import pandas as pd
#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
import test_all_nets.test_core as test_core
def main(args):
    args_raft = argparse.Namespace(alternate_corr=False, corr_levels=4, corr_radius=4, dropout=0, mixed_precision=False, \
                               model=args.train_pth[0], path=None, small=False)
    model_inference = inference_RAFT.inference(args_raft)
    test_core.test_all_datasets(args, model_inference)
   

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_datasets', type=str, nargs='+', default=['sintel','kaleidoscope', 'repeated_frames',
        'hd1k','kitti','monkaa'])
    parser.add_argument('--results_file_pth', type=str, nargs='+')
    parser.add_argument('--thresholds', type=int, nargs='+',default=[0,5,20,10000])
    parser.add_argument('--sintel_pth', type=str, nargs='+')
    parser.add_argument('--matlab_pth', type=str, nargs='+')
    parser.add_argument('--matlab_equivariance_pth', type=str, nargs='+')
    parser.add_argument('--monkaa_pth', type=str, nargs='+')
    parser.add_argument('--hd1k_pth', type=str, nargs='+')
    parser.add_argument('--kitti_pth', type=str, nargs='+')

    parser.add_argument('--train_pth', type=str, nargs='+')
    parser.add_argument('--results_pth', type=str, nargs='+')
    parser.add_argument('--model_name', type=str, nargs='+')
    parser.add_argument('--note', type=str, nargs='+')
    parser.add_argument('--mode', type=str, nargs='+')
    parser.add_argument('--save_per_frame_stats', action='store_true')
    parser.add_argument('--test_mean', action='store_true')
    # new_stefano
    args = parser.parse_args()

    main(args)


