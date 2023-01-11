import pandas as pd
import argparse
import os
import re
import matplotlib.pyplot as plt

def read_csv_and_plot(data_pth,dest_path,plot_name):


    ####manipulating dataframe

    data = pd.read_csv(data_pth)
    data = data.set_index(['model'])
    assert data.columns.to_list() == ['$0 \\leq G <5 $ I_l2','$5 \\leq G <20 $ I_l2','$G \\geq 20$ I_l2',\
                                      '$0 \\leq G <5 $ EPE', '$5 \\leq G <20 $ EPE','$G \\geq 20$ EPE']

    #data.columns = ['Iu(T180)', 'Iu(TLR)', 'Iv(T180)', 'Iv(TUD)']
    # model_name = ['DDFlow', 'FlowNetC', 'FlowNetC-M', 'GMA_things', 'IRR-PWC-chairs',
    #               'IRR-PWC-kitti', 'IRR-PWC-sintel', 'IRR-PWC-things', 'RAFT-chairs',
    #               'RAFT-kitti', 'RAFT-sintel', 'RAFT-small-things', 'RAFT-things',
    #               'RAFT_chairs_0501', 'RAFT_chairs_mir', 'RAFT_chairs_no_mir',
    #               'RAFT_things_0501_trained', 'RAFT_things_mir_pth',
    #               'RAFT_things_no_mir_pth']

    model_name = ['DDFlow',
     'FlowNetC',
     'FlowNetC-M',
     'GMA_things',
     'IRR-PWC-chairs',
     'IRR-PWC-kitti',
     'IRR-PWC-sintel',
     'IRR-PWC-things',
     'RAFT-chairs',
     'RAFT-kitti',
     'RAFT-sintel',
     'RAFT-small-things',
     'RAFT-things',
     'RAFT_chairs_0501',
     'RAFT_chairs_mir',
     'RAFT_chairs_no_mir',
     'RAFT_things_0501_trained',
     'RAFT_things_mir_pth',
     'RAFT_things_no_mir_pth',
     'kitti_fine_tune_sintel_baseline',
     'kitti_no_mirror_fine_tune_sintel_no_mirror',
     'raft-chairs2-fwd_mir',
     'raft-chairs2_mir',
     'sintel_baseline_fine_tune_raft_things_mir_batch5',
     'sintel_no_mir_fine_tune_RAFT_chairs_no_mirror',
     'things_fine_tune_raft-chairs_mir_only_FWD_FLOW',
     'things_mir_fwd_bck_fine_tune_raft-chairs2_mir']
    # if plot_name=='test_transforms_sintel_final_reduced':
        # print(plot_name)
        # a = set(model_name)
        # b = set(data.index.to_list())
        # a-b
        # b-a
    assert model_name == data.index.to_list()

    dataset_labels = ['AC', 'AC', 'AC','BT', 'ACo', 'DK', 'CS', 'BT', 'AC', 'DK', 'CS', 'BT', 'BT', 'AC', 'AC', 'AC', 'BT', 'BT','BT', 'DK','DK','AC2-FWD','AC2-ALL','CS','CS','BT-FWD','BT2']
    models_name_no_data = ['DDFlow', 'FlowNetC', 'FlowNetC-M', 'GMA', 'IRR-PWC',
                           'IRR-PWC', 'IRR-PWC', 'IRR-PWC', 'RAFT',
                           'RAFT', 'RAFT', 'RAFT-small', 'RAFT',
                           'RAFT', 'RAFT', 'RAFT','RAFT', 'RAFT',
                           'RAFT', 'RAFT','RAFT','RAFT','RAFT','RAFT','RAFT','RAFT']


    labels = ['DF\n(C)', 'FC\'\n(C)', 'FC-M\'\n(C)', 'G\n(T)', 'IP\n(Co)', 'IP\n(K)', 'IP\n(S)', 'IP\n(T)', 'R\n(C)',
              'R\n(K)',
              'R\n(S)', 'Rs\n(T)', 'Ro\n(T)', 'Ro\'\n(C)', 'R-M\'\n(C)', 'R\'\n(C)', 'Ro\'\n(T)', 'R-M\'\n(T)', 'R\'\n(T)',
              'R-M\'\n(K)','R\'\n(K)','R-M\'\n(C2f)','R-M\'\n(C2)','R-M\'\n(S)','R\'\n(S)','R-M\'\n(Tf)','R-M\'\n(C2-T)']

    data['dataset'] = dataset_labels
    data['labels'] = labels
    ##remove some models
    # models_to_remove=['RAFT-chairs',
    #  'RAFT-kitti',
    #  'RAFT-sintel',
    #  'RAFT-small-things',
    #  'RAFT-things',]

    data = data.drop(args.models_to_remove, axis=0)

    # data['dataset'] = dataset_labels
    # data['labels'] = labels
    data = data.set_index('labels')
    data = data.sort_values(['dataset', 'labels'])
    data=data.drop(['dataset'], axis=1)
    ######
    # data = data.drop('IP\n(K)')
    if args.metric =='IL2':
        data = data.drop(['$0 \leq G <5 $ EPE', '$5 \leq G <20 $ EPE', '$G \geq 20$ EPE'], axis=1)
    elif args.metric =='EPE':
        data = data.drop(['$0 \\leq G <5 $ I_l2','$5 \\leq G <20 $ I_l2','$G \\geq 20$ I_l2'], axis=1)

    data.columns = ['$0 \leq ||G|| <5 $', '$5 \leq ||G|| <20 $', '$||G|| \geq 20$']
    patterns = ['','/','///']#, '\\\\', '\\\\\\']
    colors=['#001eff','#808eff','#b8c0ff']

    if args.gnd_norm:
        if re.search('sintel', data_pth.split('/')[-2]) != None:
            gnd_mag_sintel = pd.Series({'$0 \leq ||G|| <5 $':1.82,'$5 \leq ||G|| <20 $':10.45,'$||G|| \geq 20$':54.51})
            gnd_mag = gnd_mag_sintel
        elif re.search('mat', data_pth.split('/')[-2]) != None:
            gnd_mag_mat    = pd.Series({'$0 \leq ||G|| <5 $':2.94,'$5 \leq ||G|| <20 $':12.7,'$||G|| \geq 20$':58.3})
            gnd_mag = gnd_mag_mat

        data_normed = data.div(gnd_mag)
        data = data_normed



    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 36

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)

    ####SETTING GLOBAL PARAM
    # https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Helvetica'

    # plt.rcParams['axes.edgecolor'] = '#333F4B'
    # plt.rcParams['axes.linewidth'] = 0.8
    # plt.rcParams['xtick.color'] = '#333F4B'
    # plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['hatch.linewidth'] = 1.5
    plt.rcParams['hatch.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['hatch.linewidth'] = 1.5
    plt.rcParams['hatch.color'] = 'white'

    fig, ax = plt.subplots(figsize=(10, 7))
    data.plot.bar(ax=ax, width=0.9, alpha=0.6, legend=True, color=colors)
    ##SORTING PATHCES AND STUFF FOR PANDAS MATPLOTLIB COMPATIBILITY
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    #patterns = ['//', '///', '\\\\', '\\\\\\']
    for item, pattern in zip(handles, patterns):
        print(item.patches)
        for bar_ in item:
            bar_.set_hatch(pattern)
    ax.legend()
    # data.plot(ax=ax, kind='bar', legend=False,color=['blue','skyblue','r','coral'])
    # change the style of the axis spines
    ax.set_title('')
    ax.set_xlabel('Network', color='#333F4B')

    if args.gnd_norm:
        if args.metric == 'IL2':
            ax.set_ylabel('Imbalance over groundtruth', color='#333F4B')
        elif args.metric == 'EPE':
            ax.set_ylabel('EPE over groundtruth', color='#333F4B')
    else:
        if args.metric == 'IL2':
            ax.set_ylabel('Imbalance [px]', color='#333F4B')
        elif args.metric == 'EPE':
            ax.set_ylabel('EPE [px]', color='#333F4B')

    plt.setp(ax.get_xticklabels(), rotation='horizontal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_position(('outward', 8))
    # ax.spines['bottom'].set_position(('outward', 5))
    ax.tick_params(labelrotation=0, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(dest_path, plot_name+'.pdf'), format='pdf', pad_inches=0)

    #plt.show()

    return True



def main(args):

    dest_path = args.paper_plots_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)
    dest_path=os.path.join(dest_path,'masked_hist')
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)
    if args.gnd_norm:
        dest_path = os.path.join(dest_path, args.metric + 'gnd_normed_hist')
    else:
        dest_path = os.path.join(dest_path, args.metric + '_hist')
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)
    # else:
    #     dest_path = os.path.join(dest_path, 'per_axis_hist')
    #
    # if not os.path.exists(dest_path): #be careful if dest path match the folder
    #     os.makedirs(dest_path)


    root_path = args.input_plots_pth[0]
    print('path ', root_path)
    folder_list = os.listdir(root_path)
    csv_names_list = list(filter(lambda x: x[-4:] == ".csv", folder_list))
    # csv_names_list = list(filter(lambda x: x == "", folder_list))
    # csv_names_list = list(filter(lambda x: x[0] == ".", folder_list))

    folders_names_list = list(set(folder_list) - set(csv_names_list))

    #######
    full_frame_dataset_list = list(filter(lambda x: re.search('masked', x) != None, folders_names_list))
    for item in full_frame_dataset_list:
        full_frame_csv_pth = os.path.join(root_path, item)
        print(full_frame_csv_pth)
        full_frame_csv_list = os.listdir(full_frame_csv_pth)
        masked_reduced_csv = list(filter(lambda x: re.search('table_masked_I_EPE', x) != None, full_frame_csv_list))

        masked_csv_pth = os.path.join(full_frame_csv_pth, masked_reduced_csv[0])
        read_csv_and_plot(masked_csv_pth, dest_path, item + '_' + args.metric + '_'+masked_reduced_csv[0][:-4])

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper_plots_pth', type=str, nargs='+')
    parser.add_argument('--input_plots_pth', type=str, nargs='+')
    parser.add_argument('--gnd_norm', action='store_true')
    parser.add_argument('--metric', type=str, nargs='+',default='IL2')
    parser.add_argument('--models_to_remove', type=str, nargs='+', default=[''])
    #parser.add_argument('--fig_size', type=int, nargs='+',default=[0,5,20,10000])
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