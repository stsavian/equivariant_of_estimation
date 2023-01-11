#import inference_scripts.inference_FNC as inference_FNC
import argparse
import dataset_iterators.monkaa_iterator as monkaa_iterator
import dataset_iterators.hd1k_iterator as hd1k_iterator
import dataset_iterators.KITTI_iterator as KITTI_iterator
import dataset_iterators.matlab_dataset_iterator as matlab_iterator
import dataset_iterators.sintel_iterator as sintel_iterator
import dataset_iterators.matlab_dataset_iterator as matlab_iterator
import dataframe_operations.save_dataframe as save_dataframes


def test_all_datasets(args,model_inference):
    if 'kitti' in args.testing_datasets:
        print('eval kitti')
        #########sintel clean
        kitti_dataframe_all,kitti_dataframe_full_frame,kitti_dataframe_masked,kitti_tud_tlr_t180 = \
                        KITTI_iterator.generate_dataframe(args.model_name[0],model_inference,args.kitti_pth[0],mode='training',\
                           thresholds = args.thresholds,include_lower_bound = True,\
                           include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=False)

        save_dataframes.save_dataset_dataframes(kitti_dataframe_full_frame, kitti_dataframe_masked, kitti_tud_tlr_t180, args.model_name[0], args.results_pth[0],
         args.results_file_pth[0],'kitti',save_per_frame_stats=args.save_per_frame_stats)#new_stefano
    #
    if 'hd1k' in args.testing_datasets:
        print('eval hd1k ')

        hd1k_dataframe_all,hd1k_dataframe_full_frame,hd1k_dataframe_masked,hd1k_tud_tlr_t180 = \
                        hd1k_iterator.generate_dataframe(args.model_name[0],model_inference,args.hd1k_pth[0],mode='clean',\
                           thresholds = args.thresholds,include_lower_bound = True,\
                           include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=False)

        save_dataframes.save_dataset_dataframes(hd1k_dataframe_full_frame, hd1k_dataframe_masked, hd1k_tud_tlr_t180, args.model_name[0], args.results_pth[0],
         args.results_file_pth[0],'hd1k_clean',save_per_frame_stats=args.save_per_frame_stats)

    if 'monkaa' in args.testing_datasets:

        print('eval monkaa final')
        #########sintel clean
        monkaa_dataframe_all,monkaa_dataframe_full_frame,monkaa_dataframe_masked,monkaa_tud_tlr_t180 = \
                        monkaa_iterator.generate_dataframe(args.model_name[0],model_inference,args.monkaa_pth[0],mode='final',\
                           thresholds = args.thresholds,include_lower_bound = True,\
                           include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=False)

        save_dataframes.save_dataset_dataframes(monkaa_dataframe_full_frame, monkaa_dataframe_masked, monkaa_tud_tlr_t180, args.model_name[0], args.results_pth[0],
         args.results_file_pth[0],'monkaa_final',save_per_frame_stats=args.save_per_frame_stats[0])
        print('eval monkaa clean')
        #########sintel clean
        monkaa_dataframe_all,monkaa_dataframe_full_frame,monkaa_dataframe_masked,monkaa_tud_tlr_t180 = \
                        monkaa_iterator.generate_dataframe(args.model_name[0],model_inference,args.monkaa_pth[0],mode='clean',\
                           thresholds = args.thresholds,include_lower_bound = True,\
                           include_upper_bound=False,rotate_90_degrees=False, test_Tlr_Tud=False)

        save_dataframes.save_dataset_dataframes(monkaa_dataframe_full_frame, monkaa_dataframe_masked, monkaa_tud_tlr_t180, args.model_name[0], args.results_pth[0],
         args.results_file_pth[0],'monkaa_clean',save_per_frame_stats=args.save_per_frame_stats[0])


    if 'repeated_frames' in args.testing_datasets:
        ##########matlab
        print('eval equivariance mat')
        mat_dataframe_all, mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180, dataframe_quarters = matlab_iterator.generate_dataframe(
            args.model_name[0], \
            model_inference, args.matlab_equivariance_pth[0], args.thresholds, include_lower_bound=False, \
            include_upper_bound=True, test_Tlr_Tud=False, rotate_90_degrees=False)

        save_dataframes.save_dataset_dataframes(mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180, \
                                                args.model_name[0], args.results_pth[0], args.results_file_pth[0],
                                                'mat_equivariance_dataset',save_per_frame_stats=args.save_per_frame_stats)

    if 'kaleidoscope' in args.testing_datasets:
        ##########matlab
        print('eval mat')
        mat_dataframe_all, mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180 ,dataframe_quarters = matlab_iterator.generate_dataframe \
            (args.model_name[0], model_inference ,args.matlab_pth[0] ,args.thresholds ,include_lower_bound = False, \
                            include_upper_bound=True ,test_Tlr_Tud=False ,rotate_90_degrees=False)#,save_per_frame_stats=args.save_per_frame_stats)

        save_dataframes.save_dataset_dataframes(mat_dataframe_full_frame, mat_dataframe_masked, mat_tud_tlr_t180, \
                                                args.model_name[0], args.results_pth[0] ,args.results_file_pth[0]
                                                ,'mat_dataset')
        ###save quarters
        save_dataframes.save_quarters_dataset_dataframes(dataframe_quarters ,args.model_name[0], args.results_pth[0]
                                                         ,args.results_file_pth[0] ,'mat_dataset')
    if 'sintel' in args.testing_datasets:
        print('eval sintel clean')
        #########sintel clean
        # print(args.model_name[0])
        # print(model_inference)
        # print(args.sintel_pth[0])
        sintel_dataframe_all ,sintel_dataframe_full_frame ,sintel_dataframe_masked ,sintel_tud_tlr_t180 = \
            sintel_iterator.generate_dataframe(args.model_name[0] ,model_inference ,args.sintel_pth[0] ,mode='clean', \
                                               thresholds = args.thresholds ,include_lower_bound = True, \
                                               include_upper_bound=False ,rotate_90_degrees=False, test_Tlr_Tud=False,test_mean=args.test_mean)

        save_dataframes.save_dataset_dataframes(sintel_dataframe_full_frame, sintel_dataframe_masked, sintel_tud_tlr_t180, args.model_name[0], args.results_pth[0],\
                                                args.results_file_pth[0] ,'sintel_clean',save_per_frame_stats=args.save_per_frame_stats)

        #########sintel final
        print('eval sintel final')
        sintel_dataframe_all ,sintel_dataframe_full_frame ,sintel_dataframe_masked ,sintel_tud_tlr_t180 = \
            sintel_iterator.generate_dataframe(args.model_name[0] ,model_inference ,args.sintel_pth[0] ,mode='final', \
                                               thresholds = args.thresholds ,include_lower_bound = True, \
                                               include_upper_bound=False ,rotate_90_degrees=False, test_Tlr_Tud=False,test_mean=args.test_mean)

        save_dataframes.save_dataset_dataframes(sintel_dataframe_full_frame, sintel_dataframe_masked, sintel_tud_tlr_t180, args.model_name[0], args.results_pth[0],
                                                args.results_file_pth[0] ,'sintel_final',save_per_frame_stats=args.save_per_frame_stats)


    return