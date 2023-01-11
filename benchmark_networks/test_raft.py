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
    # ##########matlab
    # print('eval equivariance mat')
    # mat_dataframe_all, mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180, dataframe_quarters = matlab_iterator.generate_dataframe(
    #     args.model_name[0], \
    #     model_inference, args.matlab_equivariance_pth[0], args.thresholds, include_lower_bound=False, \
    #     include_upper_bound=True, test_Tlr_Tud=True, rotate_90_degrees=False)
    #
    # save_dataframes.save_dataset_dataframes(mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180, \
    #                                         args.model_name[0], args.results_pth[0], args.results_file_pth[0],\
    #                                         'mat_equivariance_dataset')
    # ###save quarters
    # save_dataframes.save_quarters_dataset_dataframes(dataframe_quarters,args.model_name[0], args.results_pth[0],args.results_file_pth[0],'mat_equivariance_dataset')
    #
    # print('eval mat')
    # mat_dataframe_all, mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180,dataframe_quarters = matlab_iterator.generate_dataframe(args.model_name[0],\
    #                                                 model_inference,args.matlab_pth[0],args.thresholds,include_lower_bound = False,\
    #                    include_upper_bound=True,test_Tlr_Tud=True,rotate_90_degrees=False)
    #
    # save_dataframes.save_dataset_dataframes(mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180,\
    #                                          args.model_name[0], args.results_pth[0],args.results_file_pth[0],'mat_dataset')
    # ###save quarters
    # save_dataframes.save_quarters_dataset_dataframes(dataframe_quarters,args.model_name[0], args.results_pth[0],args.results_file_pth[0],'mat_dataset')
    # print('eval sintel clean')
    # #########sintel clean
    # sintel_dataframe_all,sintel_dataframe_full_frame,sintel_dataframe_masked,sintel_tud_tlr_t180 = \
    #                 sintel_iterator.generate_dataframe(args.model_name[0],model_inference,args.sintel_pth[0],mode='clean',\
    #                    thresholds = args.thresholds,include_lower_bound = True,\
    #                    include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=True)
    #
    # save_dataframes.save_dataset_dataframes(sintel_dataframe_full_frame, sintel_dataframe_masked, sintel_tud_tlr_t180, args.model_name[0], args.results_pth[0],
    #  args.results_file_pth[0],'sintel_clean')
    # #########sintel final
    # print('eval sintel final')
    # sintel_dataframe_all,sintel_dataframe_full_frame,sintel_dataframe_masked,sintel_tud_tlr_t180 = \
    #                 sintel_iterator.generate_dataframe(args.model_name[0],model_inference,args.sintel_pth[0],mode='final',\
    #                    thresholds = args.thresholds,include_lower_bound = True,\
    #                    include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=True)
    #
    # save_dataframes.save_dataset_dataframes(sintel_dataframe_full_frame, sintel_dataframe_masked, sintel_tud_tlr_t180, args.model_name[0], args.results_pth[0],
    #  args.results_file_pth[0],'sintel_final')

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


