import inference_scripts.inference_DDFlow_v2 as inference_DDflow
import argparse
import dataset_iterators.sintel_iterator as sintel_iterator
import dataset_iterators.matlab_dataset_iterator as matlab_iterator
import dataframe_operations.save_dataframe as save_dataframes
import numpy as np
import tensorflow as tf
import test_all_nets.test_core as test_core

def main(args):

    #fake_input_1 = np.zeros([436,1024,3])
    fake_input_1 = np.zeros(args.fake_input_size)
    model_inference  = inference_DDflow.inference([fake_input_1,fake_input_1],\
                    model_checkpoint_path = '/home/ssavian/optical_flow_networks/DDFlow/models/FlyingChairs/data_distillation',rotate_sintel = False)

    test_core.test_all_datasets(args, model_inference)



    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_datasets', type=str, nargs='+', default=['sintel','kaleidoscope', 'repeated_frames'\
        'hdk1','kitti','monkaa'])
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
    parser.add_argument('--save_per_frame_stats', action='store_true')#new_stefano
    parser.add_argument('--test_mean', action='store_true')
    parser.add_argument('--fake_input_size', type=int, nargs='+', default=[436,1024,3])

    args = parser.parse_args()

    main(args)


