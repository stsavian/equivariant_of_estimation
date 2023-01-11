import os
import pandas as pd
import dataframe_operations.cell_list_to_single_cell as masked_to_dataframe

def save_to(results_pth,model_name,testing_dataset,dataframe,stats_name):
    if not os.path.exists(results_pth):
        os.makedirs(results_pth)
    model_name_pth = os.path.join(results_pth, model_name)
    if not os.path.exists(model_name_pth):
        os.makedirs(model_name_pth)
    testing_dataset_pth = os.path.join(model_name_pth, testing_dataset)
    if not os.path.exists(testing_dataset_pth):
        os.makedirs(testing_dataset_pth)

    csv_pth = os.path.join(testing_dataset_pth, model_name + stats_name + '_'+ testing_dataset + '.csv')
    dataframe.to_csv(csv_pth, index=True)
    return csv_pth


def save_dataset_dataframes(sintel_dataframe_full_frame,sintel_dataframe_masked,sintel_tud_tlr_t180,\
                           model_name,results_pth,results_file_pth,testing_dataset,save_per_frame_stats=False):
    #save_per_frame_stats = True
    if save_per_frame_stats:
        save_to(results_pth, model_name, testing_dataset, sintel_dataframe_full_frame, 'full_frame_per_frame')

    full_frame_df =  pd.DataFrame(sintel_dataframe_full_frame.mean()).T
    masked_data_df =  masked_to_dataframe.masked_stats_to_dataframe(sintel_dataframe_masked,\
                                    sintel_dataframe_masked.columns,list_counts = 'G_mag_L2_PIXEL_COUNT',mean_type='m1')
    transforms_df = pd.DataFrame(sintel_tud_tlr_t180.mean()).T

    full_frame_df['model'] = model_name
    masked_data_df['model'] = model_name
    transforms_df['model'] = model_name

    full_frame_df.set_index('model',inplace=True)
    transforms_df.set_index('model', inplace=True)
    #masked_data_df.set_index('model', inplace=True)

    #save dataframe
    save_to(results_pth, model_name, testing_dataset,full_frame_df, 'full_frame')
    save_to(results_pth, model_name, testing_dataset, masked_data_df,'masked')
    test_transforms=False
    if test_transforms==True:
        try:
            save_to(results_pth, model_name, testing_dataset,transforms_df,'test_transforms')
            with open(os.path.join(results_file_pth, testing_dataset + '_transforms.csv'), 'a') as f:
                transforms_df.to_csv(f, index_label='model', header=f.tell() == 0)
        except TypeError:
            print('no transforms')
    with open(os.path.join(results_file_pth,testing_dataset+'_full.csv'), 'a') as f:
        full_frame_df.to_csv(f,index_label='model', header=f.tell() == 0)


    return True

def save_quarters_dataset_dataframes(dataframe_quarters,\
                           model_name,results_pth,results_file_pth,testing_dataset):

    dataframe_quarters_df =  dataframe_quarters#pd.DataFrame(dataframe_quarters.mean()).T

    dataframe_quarters_df['model'] = model_name
    dataframe_quarters_df = dataframe_quarters.groupby('quadrant').mean()
    #dataframe_quarters_df.set_index('model',inplace=True)
    #masked_data_df.set_index('model', inplace=True)

    #save dataframe
    save_to(results_pth, model_name, testing_dataset,dataframe_quarters_df, 'quarters_frame')

    with open(os.path.join(results_file_pth,testing_dataset+'quarters_frame.csv'), 'a') as f:
        dataframe_quarters_df.to_csv(f,index_label='model', header=f.tell() == 0)


    return True