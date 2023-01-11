import time
from glob import glob
import os
import pandas as pd
import re
#import imbalance_csv_plots.inference_FNC as inference
import generate_matrix_and_stats.matrix_from_file_path_and_model_instance as matrix_from_file_path_and_model_instance
import generate_matrix_and_stats.metrics_from_matrix as metrics_from_matrix
import numpy as np

def generate_dataframe(model_name,model_inference,dataset_pth,mode='training',\
                       thresholds = [0,2,5,10,20,30,50,1000],include_lower_bound = False,\
                       include_upper_bound=True,rotate_90_degrees=False, test_Tlr_Tud=False,remove_augmented_seq=True):



    images1 = sorted(glob(os.path.join(dataset_pth,mode, 'image_2/*_10.png')))
    images2 = sorted(glob(os.path.join(dataset_pth,mode, 'image_2/*_11.png')))
    flow_list = sorted(glob(os.path.join(dataset_pth,mode, 'flow_occ/*_10.png')))
    print('number of frame pairs',len(flow_list))
    list_all = []
    list_full_frame = []
    list_masked = []
    list_tud_tlr_t180 = []



    for img1, img2,flow in zip(images1, images2,flow_list):
        row={}
        frL_pth = img1
        frR_pth = img2
        flo_pth = flow

        out,out_star,GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth, model_inference,rotate_90_degrees=rotate_90_degrees)


        row_full_frame = metrics_from_matrix.full_frame_metrics(out,out_star,GND)

        #here you can set the Nan values to zero
        out[out==np.nan]=0
        out_star[out_star==np.nan]=0
        GND[GND == np.nan] = 0

        #print(row_full_frame)
        row_masked = metrics_from_matrix.masked_metrics(out,out_star,GND, thresholds,include_lower_bound = include_lower_bound,include_upper_bound=include_upper_bound)

        if test_Tlr_Tud:
            out, Tout_lr, Tout_ud, Tout_180 = matrix_from_file_path_and_model_instance.generate_O_Tud_Tlr_T180(frL_pth, \
                                                    frR_pth, flo_pth, model_inference,rotate_90_degrees=rotate_90_degrees)

            row_verify_180_Tlr_Tud = metrics_from_matrix.Tud_Tlr_T180_metrics(out, Tout_lr, Tout_ud, Tout_180)
            row.update(row_verify_180_Tlr_Tud)
            list_tud_tlr_t180.append(row_verify_180_Tlr_Tud)

        row_files_pth = {'seq.':  frL_pth.split('/')[-2], 'frame_L': frL_pth.split('/')[-1],'model_name': model_name}

        #print('TEST pair',i)
        row.update(row_files_pth)
        row.update(row_full_frame)
        row.update(row_masked)

        row_full_frame['frame_L']=frL_pth.split('/')[-1]#added on 10/02/2022 check for issues here
        row_full_frame['seq.']  =  frL_pth.split('/')[-2]#added on 10/02/2022 check for issues here

        list_all.append(row)
        list_full_frame.append(row_full_frame)
        #list_full_frame.append(row_files_pth)#added on 10/02/2022 check for issues here

        list_masked.append(row_masked)


    dataframe_all = pd.DataFrame(list_all)
    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
    dataframe_tud_tlr_t180 = pd.DataFrame(list_tud_tlr_t180)
            # if i ==4:
            #     return dataframe_all,dataframe_full_frame,dataframe_masked,dataframe_tud_tlr_t180
            ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_all,dataframe_full_frame,dataframe_masked,dataframe_tud_tlr_t180
