
import time
import os
import pandas as pd
import re
#import imbalance_csv_plots.inference_FNC as inference
import generate_matrix_and_stats.matrix_from_file_path_and_model_instance as matrix_from_file_path_and_model_instance
import generate_matrix_and_stats.metrics_from_matrix as metrics_from_matrix

def generate_dataframe(model_name,model_inference,dataset_pth,mode='clean',\
                       thresholds = [0,2,5,10,20,30,50,1000],include_lower_bound = False,\
                       include_upper_bound=True,rotate_90_degrees=False, test_Tlr_Tud=False,remove_augmented_seq=True):
    #path for clean or final
    if mode =='clean':

        frames_pth = os.path.join(dataset_pth,'frames_cleanpass')
    elif mode=='final':
        frames_pth = os.path.join(dataset_pth,'frames_finalpass')
    flow_pth   = os.path.join(dataset_pth,'optical_flow')
    # %%

    # takes out first sequence for testing
    fr_seq = sorted(os.listdir(frames_pth))
    flo_seq = sorted(os.listdir(flow_pth))

    if remove_augmented_seq:
        fr_seq = list(filter(lambda k: 'augmented'  not in k, fr_seq))
        fr_seq = list(filter(lambda k: 'difftex' not in k, fr_seq))
        flo_seq =  list(filter(lambda k: 'augmented'  not in k, flo_seq))
        flo_seq = list(filter(lambda k: 'difftex' not in k, flo_seq))

    rgt_lft = 'left/'  # 'right/'
    ftr_pst = 'into_future/'  # 'into_past/'

    n_frames = 0
    for item in fr_seq:
        fr_names = sorted(os.listdir(os.path.join(frames_pth, item,rgt_lft)))
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item,ftr_pst,rgt_lft)))
        n_frames += len(fr_names)
        print(item, len(fr_names))

    print('TOT ' + str(n_frames) + 'frames')
    #all_rows = []
    list_all = []
    list_full_frame = []
    list_masked = []
    list_tud_tlr_t180 = []
    for item in fr_seq:
        row = {}
        fr_names = sorted(os.listdir(os.path.join(frames_pth, item, rgt_lft)))
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item, ftr_pst, rgt_lft)))
        for i in range(0, len(fr_names) - 1):  # starts from 1 ends at -1 CHANGED flow_names -> item
            fr_L = fr_names[i]
            fr_R = fr_names[i + 1]
            flo_i = flo_names[i]
            frL_pth = os.path.join(frames_pth, item, rgt_lft, fr_L)
            frR_pth = os.path.join(frames_pth, item, rgt_lft, fr_R)
            flo_pth = os.path.join(flow_pth, item, ftr_pst, rgt_lft, flo_i)

            out,out_star,GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth, model_inference,rotate_90_degrees=rotate_90_degrees)
            row_full_frame = metrics_from_matrix.full_frame_metrics(out,out_star,GND)
            row_masked = metrics_from_matrix.masked_metrics(out,out_star,GND, thresholds,include_lower_bound = include_lower_bound,include_upper_bound=include_upper_bound)


            if test_Tlr_Tud:
                out, Tout_lr, Tout_ud, Tout_180 = matrix_from_file_path_and_model_instance.generate_O_Tud_Tlr_T180(frL_pth, \
                                                        frR_pth, flo_pth, model_inference,rotate_90_degrees=rotate_90_degrees)
                row_verify_180_Tlr_Tud = metrics_from_matrix.Tud_Tlr_T180_metrics(out, Tout_lr, Tout_ud, Tout_180)
                row.update(row_verify_180_Tlr_Tud)
                list_tud_tlr_t180.append(row_verify_180_Tlr_Tud)

            row_files_pth = {'seq.': item, 'frame_L': fr_L,'model_name': model_name}

            #print('TEST pair',i)
            row.update(row_files_pth)
            row.update(row_full_frame)
            row.update(row_masked)

            row_full_frame['frame_L'] = frL_pth.split('/')[-1]  # added on 10/02/2022 check for issues here
            row_full_frame['seq.'] = frL_pth.split('/')[-2]  # added on 10/02/2022 check for issues here
            list_all.append(row)
            list_full_frame.append(row_full_frame)
            #list_full_frame.append(row_files_pth)  # added on 10/02/2022 check for issues here
            list_masked.append(row_masked)
        print(item)


    dataframe_all = pd.DataFrame(list_all)
    dataframe_full_frame = pd.DataFrame(list_full_frame)

    dataframe_masked = pd.DataFrame(list_masked)
    dataframe_tud_tlr_t180 = pd.DataFrame(list_tud_tlr_t180)
            # if i ==4:
            #     return dataframe_all,dataframe_full_frame,dataframe_masked,dataframe_tud_tlr_t180
            ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_all,dataframe_full_frame,dataframe_masked,dataframe_tud_tlr_t180
