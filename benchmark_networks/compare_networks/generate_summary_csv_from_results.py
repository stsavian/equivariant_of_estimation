
# import inference_scripts.inference_RAFT as inference_RAFT
import argparse
import os
import pandas as pd
# import dataset_iterators.sintel_iterator as sintel_iterator
# import dataset_iterators.matlab_dataset_iterator as matlab_iterator
# import dataframe_operations.save_dataframe as save_dataframes

def summary_from_folder_root(args):
    '''
    generates per dataset summary from models rooted in  in desired folder --folder_pth
    the summary is saved in  --folder_pth
    :param args:
    :return:
    '''
    dest_path = args.result_summary_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)

    root_path = args.folder_pth[0]
    model_names_list = os.listdir(root_path)
    model_names_list = list(filter(lambda x: x[-4:] != '.csv' , model_names_list))
    model_names_list = list(filter(lambda x: x[0] != '.', model_names_list))
    for model_name in model_names_list: #for every model
        path = os.path.join(root_path,model_name)
        datasets = os.listdir(path)
        datasets = list(filter(lambda x: x[0] != '.', datasets))
        for dataset in datasets: #for every testing set
            #create dst csv for
            fldr_pth = os.path.join(path,dataset)
            try:
                csv_list = os.listdir(fldr_pth)
            except NotADirectoryError:
                print('output can not be in the same folder, remove output and re-run')
            for csv_name in csv_list: #for full frame, thresholded areas, quadrants
                csv_path = os.path.join(fldr_pth,csv_name)
                try:
                    new_dataframe_row = pd.read_csv(csv_path)
                except NotADirectoryError:
                    print('testing')
                if 'model' not in new_dataframe_row.columns:
                    new_dataframe_row['model'] = model_name
                ##if model not in new_dataframe_row.columns
                dest_csv_name = csv_name[len(model_name):]#name for destination file to update
                dest_csv_path = os.path.join(dest_path,dest_csv_name)
                if os.path.isfile(dest_csv_path):
                    #dest_df = pd.read_csv(dest_csv_path)
                    #dest_df.append(new_dataframe_row,ignore_index=True)
                    new_dataframe_row.to_csv(dest_csv_path,mode='a',header=None,index=False,columns=sorted(new_dataframe_row.columns))
                    print(dest_csv_name)
                else:
                    new_dataframe_row.to_csv(dest_csv_path,index=False,columns=sorted(new_dataframe_row.columns))#test
                    #print(dest_csv_path)
    return True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_summary_pth', type=str, nargs='+')
    parser.add_argument('--folder_pth', type=str, nargs='+')
    # parser.add_argument('--thresholds', type=int, nargs='+',default=[0,5,20,10000])
    # parser.add_argument('--sintel_pth', type=str, nargs='+')
    # parser.add_argument('--matlab_pth', type=str, nargs='+')

    # parser.add_argument('--train_pth', type=str, nargs='+')
    # parser.add_argument('--results_pth', type=str, nargs='+')
    # parser.add_argument('--model_name', type=str, nargs='+')
    # parser.add_argument('--note', type=str, nargs='+')
    # parser.add_argument('--mode', type=str, nargs='+')

    args = parser.parse_args()

    summary_from_folder_root(args)