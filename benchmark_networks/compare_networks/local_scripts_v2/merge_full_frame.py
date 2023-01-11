import pandas as pd
import argparse
import os
import re


def main(args):

    dest_path = args.full_frame_merged_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)

    root_path = args.full_frame_pth[0]
    csv_names_list = os.listdir(root_path)
    #csv_names_list = list(filter(lambda x: x != dest_path.split('/')[-1], csv_names_list))
    full_frame_csv = list(filter(lambda x: re.search('full_frame', x) != None, csv_names_list))


    #add for loop here
    datasets_csv_list=[]
    for i in range(0,len(full_frame_csv)):
        dataset_full_csv_pth = os.path.join(root_path, full_frame_csv[i])
        csv_list =os.listdir(dataset_full_csv_pth)
        if len(csv_list)==1:
            csv=pd.read_csv(os.path.join(dataset_full_csv_pth,csv_list[0]))

            columns_list = (list(csv.columns) and
                           args.columns)

            csv=csv[columns_list+['model']]
            csv=csv.set_index(csv['model'])
            csv= csv.drop('model',axis=1)
            columns_list=[ x+'_'+full_frame_csv[i]  for x in  columns_list]
            columns_list=[x.replace('full_frame','') for x in columns_list]
            if re.search(columns_list[i],'dataset') !=None:
                columns_list=[x.replace('.dataset', '') for x in columns_list]

            csv.columns=columns_list# +['model']
            #csv.set_index(csv['model'])
            datasets_csv_list.append(csv)


    data_merged= pd.concat(datasets_csv_list, axis=1)

    data_merged.to_csv(os.path.join(dest_path, 'full_frame_all.csv'))
    # data_merged[data_merged['I_L2_m1__sintel_final']==0].to_csv(os.path.join(dest_path, 'full_frame_average_O.csv'))
    # data_merged[data_merged['I_L2_m1__sintel_final'] != 0].to_csv(os.path.join(dest_path, 'full_frame_single_inference.csv'))

    #split_things
    # things_ind=list(filter(lambda x: re.search('things', x) != None, data_merged.index))
    # chairs_ind= set(data_merged.i ndex).symmetric_difference(set(things_ind))

    # average_test_ind= data_merged['I_L2_m1__sintel_final']==0
    # single_inference_ind = data_merged['I_L2_m1__sintel_final'] != 0

    # things_ind= data_merged.index.str.contains('things')
    # chairs_ind = ~things_ind
    # ##save_csv
    # ind= single_inference_ind & chairs_ind
    # data_merged[ind].to_csv(os.path.join(dest_path,'chairs_single_inference.csv'))
    # ind= average_test_ind & chairs_ind
    # data_merged[ind].to_csv(os.path.join(dest_path,'chairs_averaged_O.csv'))

    # ind= single_inference_ind & things_ind
    # data_merged[ind].to_csv(os.path.join(dest_path,'things_single_inference.csv'))
    # ind= average_test_ind & things_ind
    # data_merged[ind].to_csv(os.path.join(dest_path,'things_averaged_O.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_frame_pth', type=str, nargs='+')
    parser.add_argument('--full_frame_merged_pth', type=str, nargs='+')
    parser.add_argument('--columns', type=str, nargs='+',default=['EPE','EPE_180','I_L2_m1','cos_sim'])
    # parser.add_argument('--sintel_pth', type=str, nargs='+')
    # parser.add_argument('--matlab_pth', type=str, nargs='+')

    # parser.add_argument('--train_pth', type=str, nargs='+')
    # parser.add_argument('--results_pth', type=str, nargs='+')
    # parser.add_argument('--model_name', type=str, nargs='+')
    # parser.add_argument('--note', type=str, nargs='+')
    # parser.add_argument('--mode', type=str, nargs='+')

    args = parser.parse_args()
#
    main(args)