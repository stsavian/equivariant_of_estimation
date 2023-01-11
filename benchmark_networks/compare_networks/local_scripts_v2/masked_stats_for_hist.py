import pandas as pd
import argparse
import os
import re
import matplotlib.pyplot as plt


def save_masked_tables_and_hist(dataset_masked_pth, dest_path):
    data = pd.read_csv(dataset_masked_pth)
    dest_path = os.path.join(dest_path, dataset_masked_pth.split('/')[-1][:-4])
    if not os.path.exists(dest_path):  # be careful if dest path match the folder
        os.makedirs(dest_path)

    v = {}
    for label, group in data.groupby('Unnamed: 0'):
        v[str(label)] = group

    # v is a dictionary containing the metrics
    # merge for table
    # make histogram with v
    # 'EPEu_sum_msk', 'EPEv_sum_msk',
    all_metrics = v.keys()
    I_L2_df = v['I_L2_sum_m1_msk'].groupby('model').mean()
    EPE_df = v['EPE_sum_msk'].groupby('model').mean()
    eu_df = v['EPEu_sum_msk'].groupby('model').mean()
    ev_df = v['EPEv_sum_msk'].groupby('model').mean()
    iu_df = v['Iu_sum_m1_msk'].groupby('model').mean()
    iv_df = v['Iv_sum_m1_msk'].groupby('model').mean()

    iu_df = iu_df[['0', '5', '20']]
    iu_df = iu_df.rename(columns={"0": "$0 \leq Gu <5 $", "5": "$5 \leq Gu <20 $", "20": "$Gu \geq 20$"})
    iv_df = iv_df[['0', '5', '20']]
    iv_df = iv_df.rename(columns={"0": "$0 \leq Gv <5 $", "5": "$5 \leq Gv <20 $", "20": "$Gv \geq 20$"})

    eu_df = eu_df[['0', '5', '20']]
    eu_df = eu_df.rename(columns={"0": "$0 \leq Gu <5 $", "5": "$5 \leq Gu <20 $", "20": "$Gu \geq 20$"})
    ev_df = ev_df[['0', '5', '20']]
    ev_df = ev_df.rename(columns={"0": "$0 \leq Gv <5 $", "5": "$5 \leq Gv <20 $", "20": "$Gv \geq 20$"})

    I_L2_df = I_L2_df[['0', '5', '20']]
    I_L2_df = I_L2_df.rename(columns={"0": "$0 \leq G <5 $", "5": "$5 \leq G <20 $", "20": "$G \geq 20$"})

    EPE_df = EPE_df[['0', '5', '20']]
    EPE_df = EPE_df.rename(columns={"0": "$0 \leq G <5 $", "5": "$5 \leq G <20 $", "20": "$G \geq 20$"})

    ############################
    # TABLES
    #########################
    iu_df_lab = iu_df.add_suffix(' iu')
    iv_df_lab = iv_df.add_suffix(' iv')
    eu_df_lab = eu_df.add_suffix(' eu')
    ev_df_lab = ev_df.add_suffix(' ev')
    I_L2_df_lab = I_L2_df.add_suffix(' I_l2')
    EPE_df_lab = EPE_df.add_suffix(' EPE')

    tab_masked = pd.concat([iu_df_lab, iv_df_lab, eu_df_lab, ev_df_lab, I_L2_df_lab, EPE_df_lab], axis=1)
    tab_masked.to_csv(os.path.join(dest_path, 'table_masked_all.csv'))
    tab_masked = pd.concat([I_L2_df_lab, EPE_df_lab], axis=1)
    tab_masked.to_csv(os.path.join(dest_path, 'table_masked_I_EPE.csv'))

    ############################
    # HISTOGRAMS
    #########################

    # for I_L2
    # plot of ||I|| for different motion range
    plt.clf()
    labels = I_L2_df.index
    data_labels = I_L2_df.columns

    bar_plot = I_L2_df.plot.bar()
    plt.ylabel('$\overline{||I||}$')
    plt.title('Imbalance masked on different motion magnitudes')
    plt.savefig(os.path.join(dest_path,'masked_I.png'), bbox_inches='tight')  # ,pad_inches = 0)

    # for Iu,v
    # plot of ||I|| for different motion range
    plt.clf()

    iu_v = pd.concat([iu_df, iv_df], axis=1)
    iu_v = iu_v[["$0 \leq Gu <5 $", "$0 \leq Gv <5 $", "$5 \leq Gu <20 $", "$5 \leq Gv <20 $", \
                 "$Gu \geq 20$", "$Gv \geq 20$"]]

    colors = {"$0 \leq Gu <5 $": 'navy', "$0 \leq Gv <5 $": 'darkred', "$5 \leq Gu <20 $": 'royalblue',
              "$5 \leq Gv <20 $": \
                  'red', "$Gu \geq 20$": 'skyblue', "$Gv \geq 20$": 'tomato'}
    bar_plot = iu_v.plot.bar(color=colors, width=0.8)
    plt.ylabel('$\overline{|I|}$ [px]')
    plt.title('Imbalance masked on different motion magnitudes')
    plt.savefig(os.path.join(dest_path, 'masked_iu_v.png'), bbox_inches='tight')  # ,pad_inches = 0)
    # for Iu,v
    # plot of ||I|| for different motion range
    plt.clf()
    eu_v = pd.concat([eu_df, ev_df], axis=1)
    eu_v = eu_v[["$0 \leq Gu <5 $", "$0 \leq Gv <5 $", "$5 \leq Gu <20 $", "$5 \leq Gv <20 $", \
                 "$Gu \geq 20$", "$Gv \geq 20$"]]

    colors = {"$0 \leq Gu <5 $": 'navy', "$0 \leq Gv <5 $": 'darkred', "$5 \leq Gu <20 $": 'royalblue',
              "$5 \leq Gv <20 $": \
                  'red', "$Gu \geq 20$": 'skyblue', "$Gv \geq 20$": 'tomato'}
    bar_plot = eu_v.plot.bar(color=colors, width=0.8)
    plt.ylabel('eu,ev [px]')
    plt.title('EPE masked on different motion magnitudes')
    plt.savefig(os.path.join(dest_path, 'masked_eu_v.png'), bbox_inches='tight')  # ,pad_inches = 0)
    plt.clf()

    bar_plot = EPE_df.plot.bar()
    plt.ylabel('EPE')
    plt.title('EPE masked on different motion magnitudes')
    plt.savefig(os.path.join(dest_path, 'masked_EPE.png'), bbox_inches='tight')
    return True

def save_quadrant_tables_and_hist(dataset_masked_pth, dest_path):
    data = pd.read_csv(dataset_masked_pth)
    dest_path = os.path.join(dest_path, dataset_masked_pth.split('/')[-1][:-4])
    if not os.path.exists(dest_path):  # be careful if dest path match the folder
        os.makedirs(dest_path)

    data1 = data[['EPE', 'model','quadrant']]
    data = data.set_index('model')

    v = {}
    for label, group in data.groupby('quadrant'):
        v[str(label)] = group['EPE']#.columns.add_suffix(str(label))
    EPE_df = pd.DataFrame(v)

    v = {}
    for label, group in data.groupby('quadrant'):
        v[str(label)] = group['I_L2_m1']#.columns.add_suffix(str(label))
    I_L2_df = pd.DataFrame(v)

    v = {}
    for label, group in data.groupby('quadrant'):
        v[str(label)] = group['cos_sim']#.columns.add_suffix(str(label))
    cos_sim_df = pd.DataFrame(v)

    v = {}
    for label, group in data.groupby('quadrant'):
        v[str(label)] = group['spatium']#.columns.add_suffix(str(label))
    spatium_df = pd.DataFrame(v)


    bar_plot = EPE_df.plot.bar(width=0.8)
    plt.ylabel('EPE [px]')
    plt.title('EPE masked on quarters')
    plt.savefig(os.path.join(dest_path, 'EPE_quarters.png'), bbox_inches='tight')  # ,pad_inches = 0)
    plt.clf()

    bar_plot = I_L2_df.plot.bar(width=0.8)
    plt.ylabel('$\overline{||I||}$ [px]')
    plt.title('$\overline{||I||}$ masked on quarters')
    plt.savefig(os.path.join(dest_path, 'I_quarters.png'), bbox_inches='tight')  # ,pad_inches = 0)
    plt.clf()


    bar_plot = cos_sim_df.plot.bar(width=0.8)
    plt.ylabel('$\theta$ [$\degree$]')
    plt.title('$\theta$ masked on quarters')
    plt.savefig(os.path.join(dest_path, 'theta_quarters.png'), bbox_inches='tight')  # ,pad_inches = 0)
    plt.clf()

    bar_plot = spatium_df.plot.bar(width=0.8)
    plt.ylabel('$spatium$ [px]')
    plt.title('$spatium$ masked on quarters')
    plt.savefig(os.path.join(dest_path, 'spatium_quarters.png'), bbox_inches='tight')  # ,pad_inches = 0)
    plt.clf()

    #plt.show()
    return True

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
    csv_names_list = list(filter(lambda x: x != dest_path.split('/')[-1], csv_names_list))

    sintel_masked_csv = list(filter(lambda x: re.search('masked', x) != None, sintel_csv))

    mat_quarters_csv = list(filter(lambda x: re.search('quarters_frame', x) != None, mat_csv))
    mat_masked_csv = list(filter(lambda x: re.search('masked', x) != None, mat_csv))

    ############
    ###### MAT EQUIVARIANCE
    ###########
    mat_equivariance_csv = list(filter(lambda x: re.search('mat_equivariance', x) != None, csv_names_list))
    mat_quarters_equivariance_csv = list(filter(lambda x: re.search('quarters_frame', x) != None, mat_equivariance_csv))
    mat_masked_equivariance_csv = list(filter(lambda x: re.search('masked', x) != None, mat_equivariance_csv))

    mat_masked_equivariance = os.path.join(root_path,mat_masked_equivariance_csv[0])
    save_masked_tables_and_hist(mat_masked_equivariance, dest_path)
    if len(mat_quarters_equivariance_csv) !=0:
        mat_quarters_equivariance = os.path.join(root_path, mat_quarters_equivariance_csv[0])
        save_quadrant_tables_and_hist(mat_quarters_equivariance, dest_path)
    ############

    sintel_masked_csv_pth = os.path.join(root_path,sintel_masked_csv[0])
    save_masked_tables_and_hist(sintel_masked_csv_pth, dest_path)
    sintel_masked_csv_pth = os.path.join(root_path,sintel_masked_csv[1])
    save_masked_tables_and_hist(sintel_masked_csv_pth, dest_path)

    matlab_masked = os.path.join(root_path,mat_masked_csv[0])
    save_masked_tables_and_hist(matlab_masked, dest_path)

    matlab_quadrants = os.path.join(root_path, mat_quarters_csv[0])
    save_quadrant_tables_and_hist(matlab_quadrants, dest_path)

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