import pandas as pd
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def read_csv_and_plot(data_pth,dest_path,plot_name):


    ####manipulating dataframe

    data = pd.read_csv(data_pth)
    data = data.set_index(['model'])
    assert data.columns.to_list() == ['EPE', '$\\overline{||I||}$', '$e_u$', '$e_v$', '$I_u$', '$I_v$']

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

    ##drop certain labels!
    #sort by C-T-S-K for all
    data = data.set_index('labels')
    data = data.sort_values(['dataset', 'labels'])
    data.drop(['dataset'], axis=1)


    ######
    # data = data.drop('IP\n(K)')

    ######
    #REMOVE UNECESSARY COLUMNS
    if args.L2_eval:
        data = data.drop(['$e_u$', '$e_v$', '$I_u$', '$I_v$'], axis=1)
        patterns = ['//', '///']#, '\\\\', '\\\\\\']
        if len(data)<10:
            patterns = ['/', '//']
        colors=['blue','coral']
        #colors=['#5B84B1FF','#FC766AFF']
        #https://www.designwizard.com/blog/design-trends/colour-combination
        #colors=['#00203FFF', '#ADEFD1FF']
        plt.rcParams['hatch.linewidth'] = 5
        plt.rcParams['hatch.color'] = 'white'

    else:
        data = data.drop(['EPE', '$\\overline{||I||}$'], axis=1)
        #data=data[['$e_u$', '$e_v$', '$I_u$', '$I_v$']]
        data = data[['$I_u$', '$I_v$']]
        colors = ['blue', 'coral']
        #data.columns = ['$\overline{e_u}$','$\overline{e_v}$','$\overline{I_u}$', '$\overline{I_v}$']
        data.columns = ['$\overline{I_u}$', '$\overline{I_v}$']
        patterns = ['//', '///', '\\\\', '\\\\\\']
        colors =['blue', 'r']#'skyblue', 'r', 'coral']
        plt.rcParams['hatch.linewidth'] = 1.5
        plt.rcParams['hatch.color'] = 'white'
    return
    ######
    # PLOTTING
    ######

def plot_data(data,dest_path,plot_name):
    patterns = ['/', '//']
    colors=['blue','coral']

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

    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

    plt.rcParams['hatch.linewidth'] = 1.5
    plt.rcParams['hatch.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333F4B'
    plt.rcParams['ytick.color'] = '#333F4B'

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

    ax.set_ylabel('Value [px]', color='#333F4B')

    plt.setp(ax.get_xticklabels(), rotation='horizontal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_position(('outward', 8))
    # ax.spines['bottom'].set_position(('outward', 5))
    ax.tick_params(labelrotation=0, axis='x')
    fig.tight_layout()
    if len(data) < 15:
        fig.savefig(os.path.join(dest_path, plot_name +  '_raft_trainings' + '.pdf'), format='pdf', pad_inches=0)
    else:
        fig.savefig(os.path.join(dest_path, plot_name+'.pdf'), format='pdf', pad_inches=0)

    #plt.show()

    return True



def main(args):

    dest_path = args.plots_pth[0]
    if not os.path.exists(dest_path): #be careful if dest path match the folder
        os.makedirs(dest_path)

    random=False
    if random:
        rng = np.random.default_rng()
        values = rng.integers(0, 10, size=(4, 2))
    values=[[2.83,1.47],[2.60,1.02],[2.66,0.77],[2.78,0.54]]
    data = pd.DataFrame(values, columns=['EPE', '$\overline{||I||}$'],index=['BASELINE', '$0.3$', '$0.6$','$1.0$'])

    
    plot_data(data,args.plots_pth[0],'TEST')

    

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots_pth', type=str, nargs='+')
    # parser.add_argument('--input_plots_pth', type=str, nargs='+')
    # parser.add_argument('--L2_eval', action='store_true')
    # parser.add_argument('--models_to_remove', type=str, nargs='+',default=[''])

    #parser.add_argument('--fig_size', type=int, nargs='+',default=[0,5,20,10000])
    # parser.add_argument('--sintel_pth', type=str, nargs='+')
    # parser.add_argument('--matlab_pth', type=str, nargs='+')


    # parser.add_argument('--results_pth', type=str, nargs='+')
    # parser.add_argument('--model_name', type=str, nargs='+')
    # parser.add_argument('--note', type=str, nargs='+')
    # parser.add_argument('--mode', type=str, nargs='+')

    args = parser.parse_args()
#
    main(args)