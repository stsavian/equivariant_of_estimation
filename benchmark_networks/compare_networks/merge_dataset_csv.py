
# import inference_scripts.inference_RAFT as inference_RAFT
import argparse
import os
import re
import pandas as pd
# import dataset_iterators.sintel_iterator as sintel_iterator
# import dataset_iterators.matlab_dataset_iterator as matlab_iterator
# import dataframe_operations.save_dataframe as save_dataframes


def main(args):
    summary_pth = args.summary_pth[0]
    dest_path = args.merged_summary_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)

    root_path = args.summary_pth[0]
    csv_names_list = os.listdir(root_path)
    csv_names_list = list(filter(lambda x: x != dest_path.split('/')[-1], csv_names_list))

    #filter here the different datasets
    sintel_csv = list(filter(lambda x: re.search('sintel', x) != None, csv_names_list))
    mat_csv  = list(filter(lambda x: re.search('_mat_dataset', x) != None, csv_names_list))

    #filter here masked
    sintel_full_csv = list(filter(lambda x: re.search('full_frame', x) != None, sintel_csv))
    sintel_masked_csv = list(filter(lambda x: re.search('masked', x) != None, sintel_csv))

    # filter here transforms
    sintel_transforms_csv = list(filter(lambda x: re.search('transforms', x) != None, sintel_csv))

    full_dataframes = []
    for item in sintel_full_csv:
        full_path = os.path.join(root_path,item)
        try:
            data = pd.read_csv(full_path)
        except:
            print('you should remove ',full_path, 'and run this script again')


        data= data.set_index('model')
        data.drop_duplicates(subset=None, keep='first', inplace=True)
        if re.search('final',item) != None:
            data.columns = [ str(col) + '_final' if col!='model' else str(col) for col in data.columns ]
        elif re.search('clean',item) != None:
            data.columns = [str(col) + '_clean'if col!='model' else str(col)  for col in data.columns]
        full_dataframes.append(data)
    merged_full_sintel = pd.concat(full_dataframes,axis=1)
    ##concat_transforms
    # transforms_dataframes = []
    # for item in sintel_transforms_csv:
    #     full_path = os.path.join(root_path,item)
    #     data = pd.read_csv(full_path)
    #     data= data.set_index('model')
    #     data.drop_duplicates(subset=None, keep='first', inplace=True)
    #     if re.search('final',item) != None:
    #         data.columns = [ str(col) + '_final' if col!='model' else str(col) for col in data.columns ]
    #     elif re.search('clean',item) != None:
    #         data.columns = [str(col) + '_clean'if col!='model' else str(col)  for col in data.columns]
    #     transforms_dataframes.append(data)
    # merged_transforms_sintel = pd.concat(transforms_dataframes,axis=1)
    ## concat masked
    sintel_masked_dataframes = []
    for item in sintel_masked_csv:
        full_path = os.path.join(root_path,item)
        data = pd.read_csv(full_path)
        data= data.set_index('model')
        data.drop_duplicates(subset=None, keep='first', inplace=True)
        if re.search('final',item) != None:
            data.columns = [ str(col) + '_final' if col!='model' else str(col) for col in data.columns ]
        elif re.search('clean',item) != None:
            data.columns = [str(col) + '_clean'if col!='model' else str(col)  for col in data.columns]
        sintel_masked_dataframes.append(data)
    merged_masked_sintel = pd.concat(sintel_masked_dataframes,axis=1)
    #masked_sintel_summary = merged_masked_sintel.groupby('model').mean()

    ##mat_dataset

    #filter here mat_csv masked
    mat_full_csv = list(filter(lambda x: re.search('full_frame', x) != None, mat_csv))
    mat_quarters_csv = list(filter(lambda x: re.search('quarters_frame', x) != None, mat_csv))
    mat_masked_csv = list(filter(lambda x: re.search('masked', x) != None, mat_csv))
    mat_transforms_csv = list(filter(lambda x: re.search('transforms', x) != None, mat_csv))

    # filter here transforms
    # full_path = os.path.join(root_path,mat_full_csv[0])#[0]
    # data_mat_full = pd.read_csv(full_path)
    # data_mat_full = data_mat_full.set_index('model')
    #data_mat_full.drop_duplicates(subset=None, keep='first', inplace=True)

    #quarters_mat_summary.drop_duplicates(subset=None, keep='first', inplace=True)

    # masked_mat_path = os.path.join(root_path,mat_masked_csv)#[0]
    # data_mat_masked = pd.read_csv(masked_mat_path)
    # data_mat_masked = data_mat_masked.set_index('model').groupby('model').mean()
    #data_mat_masked.drop_duplicates(subset=None, keep='first', inplace=True)

    #not implemented yet
    # transforms_mat_path = os.path.join(root_path, mat_transforms_csv[0])
    # data_mat_transforms = pd.read_csv(masked_mat_path)
    # data_mat_transforms = data_mat_masked.set_index('model')

    ##collect all dataframes

    merged_full_sintel.to_csv(os.path.join(dest_path,'sintel_clean_final_full.csv'))  # test
   # merged_transforms_sintel.to_csv(os.path.join(dest_path,'sintel_clean_final_transforms.csv'))
    merged_masked_sintel.to_csv(os.path.join(dest_path,'sintel_clean_final_masked.csv'))

    # data_mat_full.to_csv(os.path.join(dest_path,'matlab_full.csv'))
    # quarters_mat_summary.to_csv(os.path.join(dest_path,'matlab_quarters.csv'))
    # data_mat_masked.to_csv(os.path.join(dest_path,'matlab_masked.csv'))


    sintel_data = pd.concat([merged_full_sintel],axis=1)#, masked_sintel_summary],axis=1)
    sintel_data.columns = [str(col) + '_sintel'if col!='model' else str(col)  for col in sintel_data.columns]

    # mat_data = pd.concat([data_mat_full,quarters_mat_summary,data_mat_masked],axis=1)
    # mat_data.columns = [str(col) + '_mat'if col!='model' else str(col)  for col in mat_data.columns]
    #
    # data_all = pd.concat([sintel_data,mat_data],axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merged_summary_pth', type=str, nargs='+')
    parser.add_argument('--summary_pth', type=str, nargs='+')
    # parser.add_argument('--thresholds', type=int, nargs='+',default=[0,5,20,10000])
    # parser.add_argument('--sintel_pth', type=str, nargs='+')
    # parser.add_argument('--matlab_pth', type=str, nargs='+')

    # parser.add_argument('--train_pth', type=str, nargs='+')
    # parser.add_argument('--results_pth', type=str, nargs='+')
    # parser.add_argument('--model_name', type=str, nargs='+')
    # parser.add_argument('--note', type=str, nargs='+')
    # parser.add_argument('--mode', type=str, nargs='+')

    args = parser.parse_args()

    main(args)