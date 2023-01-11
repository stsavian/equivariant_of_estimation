import pandas as pd
import argparse
import os
import re
import matplotlib.pyplot as plt

def save_table_histogram(dataset_full_pth,dest_path):
    data = pd.read_csv(dataset_full_pth)
    data = data.set_index('model')
    data = data.sort_index()
    dest_path = os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4])
    if not os.path.exists(dest_path):  # be careful if dest path match the folder
        os.makedirs(dest_path)

    table_reduced = data
    table_reduced = data[['EPE', 'I_L2_m1','EPE_u',  'EPE_v','Iu_m1',
       'Iv_m1']]
    
    table_reduced = table_reduced.rename(columns={"EPE_u": "$e_u$", "EPE_v": "$e_v$","Iu_m1":"$I_u$","Iv_m1":"$I_v$",\
                                                  'I_L2_m1':"$\overline{||I||}$"})
    table_L1 = table_reduced[["$e_u$","$e_v$","$I_u$","$I_v$"]]
    bar_plot = table_reduced.plot.bar(width=0.9)
    plt.ylabel('value [px]')
    plt.title(dataset_full_pth.split('/')[-1][:-4])
    plt.savefig(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4]+'.png'), bbox_inches='tight')
    # plt.clf()
    # bar_plot = table_L1.plot.bar(width=0.9)
    # plt.ylabel('value [px]')
    # plt.title(dataset_full_pth.split('/')[-1][:-4])
    # plt.savefig(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4]+'_L1.png'), bbox_inches='tight')
    #
    # #plt.show()
    # data.to_csv(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4]+'_all.csv'))
    table_reduced.to_csv(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4] + '_reduced.csv'))

    return table_reduced

def save_transforms_table_histogram(dataset_full_pth,dest_path):
    data = pd.read_csv(dataset_full_pth)
    data = data.set_index('model')
    data = data.sort_index()
    dest_path = os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4])
    if not os.path.exists(dest_path):  # be careful if dest path match the folder
        os.makedirs(dest_path)

    table_reduced = data#debug
    #table_reduced = data[['Iu_T180_m1', 'Iu_Tlr_m1', 'Iv_T180_m1', 'Iv_Tud_m1']]

    # table_reduced = table_reduced.rename(columns={"EPE_u": "$e_u$", "EPE_v": "$e_v$","Iu_m1":"$I_u$","Iv_m1":"$I_v$",\
    #                                               'I_L2_m1':"$\overline{||I||}$"})
    # bar_plot = table_reduced.plot.bar(width=0.9)
    # plt.ylabel('value [px]')
    # plt.title(dataset_full_pth.split('/')[-1][:-4])
    # plt.savefig(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4]+'.png'), bbox_inches='tight')
    # #plt.show()
    # data.to_csv(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4]+'_all.csv'))
    table_reduced.to_csv(os.path.join(dest_path, dataset_full_pth.split('/')[-1][:-4] + '_reduced.csv'))

    return table_reduced



def main(args):

    dest_path = args.plots_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)

    root_path = args.summary_pth[0]
    csv_names_list = os.listdir(root_path)
    csv_names_list = list(filter(lambda x: x != dest_path.split('/')[-1], csv_names_list))

    #filter here the different datasets
    sintel_csv = list(filter(lambda x: re.search('sintel', x) != None, csv_names_list))
    mat_csv  = list(filter(lambda x: re.search('mat_dataset', x) != None, csv_names_list))

    sintel_full_csv = list(filter(lambda x: re.search('full', x) != None, sintel_csv))
    mat_full_csv = list(filter(lambda x: re.search('full', x) != None, mat_csv))

    # sintel_transforms_csv = list(filter(lambda x: re.search('transforms', x) != None, sintel_csv))
    # mat_transforms_csv = list(filter(lambda x: re.search('transforms', x) != None, mat_csv))

    # sintel_transforms_csv_pth = os.path.join(root_path,sintel_transforms_csv[0])
    # sintel_transforms_clean_df = save_transforms_table_histogram(sintel_transforms_csv_pth,dest_path)

    # sintel_transforms_csv_pth = os.path.join(root_path,sintel_transforms_csv[1])
    # sintel_transforms_final_df = save_transforms_table_histogram(sintel_transforms_csv_pth,dest_path)

    #mat_full_csv_pth = os.path.join(root_path,mat_full_csv[0])

    # mat_transforms_csv_pth = os.path.join(root_path, mat_transforms_csv[0])
    # mat_transforms_df = save_transforms_table_histogram(mat_transforms_csv_pth,dest_path)

    ############
    ###### MAT EQUIVARIANCE
    ###########
    #mat_equivariance_csv = list(filter(lambda x: re.search('mat_equivariance', x) != None, csv_names_list))
    #mat_equivariance_full_csv = list(filter(lambda x: re.search('full', x) != None, mat_equivariance_csv))
   # mat_equivariance_transforms_csv = list(filter(lambda x: re.search('transforms', x) != None, mat_equivariance_csv))
    #mat_equivariance_transforms_csv_pth = os.path.join(root_path, mat_equivariance_transforms_csv[0])
    #mat_equivariance_transforms_df = save_transforms_table_histogram(mat_equivariance_transforms_csv_pth,dest_path)

    ############
    ########
    ############

    sintel_full_csv_pth = os.path.join(root_path,sintel_full_csv[0])
    sintel_clean_df = save_table_histogram(sintel_full_csv_pth,dest_path)
    sintel_full_csv_pth = os.path.join(root_path,sintel_full_csv[1])
    sintel_final_df = save_table_histogram(sintel_full_csv_pth,dest_path)

    #mat_equivariance_full_csv_pth = os.path.join(root_path,mat_equivariance_full_csv[0])
    #mat_equivariance_df = save_table_histogram(mat_equivariance_full_csv_pth,dest_path)
    ############
    ###### MAT EQUIVARIANCE
    ###########
    #mat_full_csv_pth = os.path.join(root_path, mat_full_csv[0])
    #mat_df = save_table_histogram(mat_full_csv_pth, dest_path)

    # table_mat_sign_imb_equivariance = pd.concat([mat_df[['EPE','$\overline{||I||}$','$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' sign '),\
    #                                             mat_equivariance_df[['EPE','$\overline{||I||}$','$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' equiv ')],axis=1)
    # table_all = pd.concat([sintel_clean_df[['EPE','$\overline{||I||}$','$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' sint. clean'),\
    #           sintel_final_df[['EPE','$\overline{||I||}$','$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' sint. final'),\
    #     mat_df[['EPE','$\overline{||I||}$','$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' mat ')],axis=1)
    #
    #
    # table_red = pd.concat([sintel_clean_df[['EPE','$\overline{||I||}$']].add_suffix(' sint. clean'),\
    #           sintel_final_df[['EPE','$\overline{||I||}$']].add_suffix(' sint. final'),\
    #     mat_df[['EPE','$\overline{||I||}$']].add_suffix(' mat ')],axis=1)
    #
    # table_L1 = pd.concat([sintel_clean_df[['$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' sint. clean'),\
    #           sintel_final_df[['$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' sint. final'),\
    #     mat_df[['$e_u$', '$e_v$','$I_u$', '$I_v$']].add_suffix(' mat ')],axis=1)
    #
    # table_L1.to_csv(os.path.join(dest_path,'sintel_mat_full_frame_l1.csv'))
    # table_all.to_csv(os.path.join(dest_path,'sintel_mat_full_frame.csv'))
    # table_red.to_csv(os.path.join(dest_path,'sintel_mat_full_frame_reduced.csv'))
    # table_mat_sign_imb_equivariance.to_csv(os.path.join(dest_path,'mat_full_frame_equiv.csv'))
    # bar_plot = table_red.plot.bar(width=0.9)
    # plt.ylabel('value [px]')
    # plt.title('imbalance_sintel and mat')
    # plt.savefig(os.path.join(dest_path,'sintel_matlab_full'+ '.png'), bbox_inches='tight')
    #plt.show()

    #make here table transforms

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_pth', type=str, nargs='+')
    parser.add_argument('--plots_pth', type=str, nargs='+')
    # parser.add_argument('--thresholds', type=int, nargs='+',default=[0,5,20,10000])
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