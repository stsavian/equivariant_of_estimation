import time
import os
import pandas as pd
#import imbalance_csv_plots.inference_FNC as inference
import generate_matrix_and_stats.matrix_from_file_path_and_model_instance as matrix_from_file_path_and_model_instance
import generate_matrix_and_stats.metrics_from_matrix as metrics_from_matrix
import os
import pandas as pd
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"




def generate_dataframe(model_name,model_inference,dataset_pth,thresholds,include_lower_bound = False,\
                       include_upper_bound=True,test_Tlr_Tud=False,rotate_90_degrees=False):#changed test_Tlr_Tud=True to default


    data_absolute_pth = os.path.dirname(dataset_pth)
    input_csv_pth = os.path.join(dataset_pth, dataset_pth.split('/')[-1] + '.csv')

    input_csv_df = pd.read_csv(input_csv_pth)
    list_all = []
    list_full_frame = []
    list_masked = []
    list_tud_tlr_t180 = []
    list_quarters = []

    #input_csv_df=input_csv_df.iloc[0:5]#debug

    for index, row in input_csv_df.iterrows():
        frL_pth = os.path.join(data_absolute_pth,row[input_csv_df.columns[0]])
        frR_pth = os.path.join(data_absolute_pth,row[input_csv_df.columns[1]])
        flo_pth = os.path.join(data_absolute_pth,row[input_csv_df.columns[2]])
        #print(frameL_pth.split('/')[-1],frameR_pth.split('/')[-1],flow_pth.split('/')[-1])

        out, out_star, GND = matrix_from_file_path_and_model_instance.generate_o_star(frL_pth, frR_pth, flo_pth,
                                                                                      model_inference,
                                                                                      rotate_90_degrees=rotate_90_degrees)
        row_full_frame = metrics_from_matrix.full_frame_metrics(out, out_star, GND,test_on_symmetric_data=True)
        row_masked = metrics_from_matrix.masked_metrics(out, out_star, GND, thresholds,
                                                        include_lower_bound=include_lower_bound,
                                                        include_upper_bound=include_upper_bound)
        row_quarters,row_col_quarters = metrics_from_matrix.quarter_metrics(out, out_star, GND)

        if test_Tlr_Tud:
            out, Tout_lr, Tout_ud, Tout_180 = matrix_from_file_path_and_model_instance.generate_O_Tud_Tlr_T180(frL_pth, \
                                                                                                               frR_pth,
                                                                                                               flo_pth,
                                                                                                               model_inference,
                                                                                                               rotate_90_degrees=rotate_90_degrees)
            row_verify_180_Tlr_Tud = metrics_from_matrix.Tud_Tlr_T180_metrics(out, Tout_lr, Tout_ud, Tout_180)
            #row.update(row_verify_180_Tlr_Tud)
            list_tud_tlr_t180.append(row_verify_180_Tlr_Tud)

        row_files_pth = { 'frame_L': frL_pth.split('/')[-1], 'model_name': model_name}


        # row.update(row_files_pth)
        # row.update(row_full_frame)
        # row.update(row_masked)
        # row.update(row_col_quarters)

        # row.update(pd.Series(row_files_pth))#changed to this because FNC had problems, revert back if it doesn't work
        # row.update(pd.Series(row_full_frame))
        # row.update(pd.Series(row_masked))
        # row.update(pd.Series(row_col_quarters))


        row_full_frame['frame_L']= frL_pth.split('/')[-1]#added on 10/02/2022 check for issues here
        row_full_frame['seq.']  =  frL_pth.split('/')[-2]#added on 10/02/2022 check for issues here
        list_all.append(row)
        list_full_frame.append(row_full_frame)
        list_full_frame.append(row_files_pth)  # added on 10/02/2022 check for issues here
        list_masked.append(row_masked)
        list_quarters.extend(row_col_quarters)
        if index %100 == 0:
            print('computed pairs',index)


    dataframe_all = pd.DataFrame(list_all)
    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
    dataframe_tud_tlr_t180 = pd.DataFrame(list_tud_tlr_t180)
    dataframe_quarters = pd.DataFrame(list_quarters)



        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_all, dataframe_full_frame, dataframe_masked, dataframe_tud_tlr_t180,dataframe_quarters