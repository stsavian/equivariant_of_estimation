import time
import os
import pandas as pd
import test_groundtruth_motion.stats_from_gnd_matrix as stats_from_gnd_matrix
import argparse
from dataframe_operations.cell_list_to_single_cell import masked_stats_to_dataframe
from glob import glob
import os.path as osp
import networks.RAFT.core.utils.frame_utils as frame_utils
def chairsOcc_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000],flow_direction='all'):

    # image1_filenames = sorted(glob(os.path.join(dataset_pth, "*_img1.png")))
    # image2_filenames = sorted(glob(os.path.join(dataset_pth, "*_img2.png")))
    # occ1_filenames = sorted(glob(os.path.join(dataset_pth, "*_occ1.png")))
    # occ2_filenames = sorted(glob(os.path.join(dataset_pth, "*_occ2.png")))
    # flow_f_filenames = sorted(glob(os.path.join(dataset_pth, "*_flow.flo")))
    # flow_b_filenames = sorted(glob(os.path.join(dataset_pth, "*_flow_b.flo")))
    if flow_direction == 'fwd':
        directions = ['fwd']
    elif flow_direction == 'bck':
        directions = ['bck']
    else:
        directions = ['bck','fwd']
    list_full_frame = []
    list_masked = []

    for flo in directions:
        labels = {'fwd': '*_flow.flo', 'bck': '*_flow_b.flo'}
        # images_0 = sorted(glob(osp.join(dataset_pth, item, '*img_0.png')))
        # images_1 = sorted(glob(osp.join(dataset_pth, item, '*img_1.png')))
        flows = sorted(glob(osp.join(dataset_pth,'data', labels[flo])))
        # assert (len(images)//2 == len(flows))
        #print(flows)
        for i in range(len(flows)):
            flo_pth= flows[i]
#data_absolute_pth = os.path.dirname(dataset_pth)
            row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
            print(i,flo_pth)
            row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth, thresholds=thresholds)
            list_full_frame.append(row_full_frame)
            list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked

def chairs2_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000],chairs2_flow_direction='all'):

    if chairs2_flow_direction == 'fwd':
        directions = ['fwd']
    elif chairs2_flow_direction == 'bck':
        directions = ['bck']
    else:
        directions = ['fwd', 'bck']
    list_full_frame = []
    list_masked = []
    for item in ['train', 'val']:
        for flo in directions:
            labels = {'fwd': '*flow_01.flo', 'bck': '*flow_10.flo'}
            # images_0 = sorted(glob(osp.join(dataset_pth, item, '*img_0.png')))
            # images_1 = sorted(glob(osp.join(dataset_pth, item, '*img_1.png')))
            flows = sorted(glob(osp.join(dataset_pth, item, labels[flo])))
            # assert (len(images)//2 == len(flows))

            for i in range(len(flows)):
                flo_pth= flows[i]
    #data_absolute_pth = os.path.dirname(dataset_pth)
                row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
                row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth, thresholds=thresholds)
                list_full_frame.append(row_full_frame)
                list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked

def hd1k_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000]):
    #data_absolute_pth = os.path.dirname(dataset_pth)
    flows = sorted(glob(os.path.join(dataset_pth, 'hd1k_flow_gt', 'flow_occ/*.png')))
    list_full_frame = []
    list_masked = []
    for flo_pth in flows:

        row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
        row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth, thresholds=thresholds)
        list_full_frame.append(row_full_frame)
        list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked


def kitti_generate_gnd_dataframe_full_masked(dataset_pth, thresholds=[0, 2, 5, 10, 20, 30, 50, 1000]):
    # data_absolute_pth = os.path.dirname(dataset_pth)
    mode='training'
    flow_list = sorted(glob(os.path.join(dataset_pth,mode, 'flow_occ/*_10.png')))
    list_full_frame = []
    list_masked = []
    for flo_pth in flow_list:
        row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
        row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth, thresholds=thresholds)
        list_full_frame.append(row_full_frame)
        list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
    ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked

def monkaa_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000],evaluate_past = True, evaluate_future = True,remove_augmented_seq=True):
    flow_pth = os.path.join(dataset_pth, 'optical_flow')
    #flow_pth = os.path.join(dataset_pth, flow_pth)
    flo_seq = sorted(os.listdir(flow_pth))
    rgt_lft = 'left/'  # 'right/'
    #ftr_pst = 'into_future/'  # 'into_past/'
    #
    #     fr_seq = list(filter(lambda k: 'augmented'  not in k, fr_seq))
    #     fr_seq = list(filter(lambda k: 'difftex' not in k, fr_seq))
    if remove_augmented_seq:
        flo_seq =  list(filter(lambda k: 'augmented'  not in k, flo_seq))
        flo_seq = list(filter(lambda k: 'difftex' not in k, flo_seq))
    directions = []
    if evaluate_future:
        directions.append('into_future/')
    if evaluate_past:
        directions.append('into_past/')

    for ftr_pst in directions:
        n_frames = 0
        for item in flo_seq:
            flo_names = sorted(os.listdir(os.path.join(flow_pth, item,ftr_pst,rgt_lft)))
            n_frames += len(flo_names)
            print(item, len(flo_names))

        print('TOT ' + str(n_frames) + ' flow fields')
        #all_rows = []

        list_full_frame = []
        list_masked = []
        for ftr_pst in directions:
            for item in flo_seq:
                print(item)
                flo_names = sorted(os.listdir(os.path.join(flow_pth, item,ftr_pst,rgt_lft)))
                start = time.time()
                for i in range(0,len(flo_names) ):
                    flo_i = flo_names[i]
                    flo_pth = os.path.join(flow_pth, item,ftr_pst,rgt_lft,flo_i)
                    row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
                    row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth,thresholds=thresholds)
                    list_full_frame.append(row_full_frame)
                    list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)

    return dataframe_full_frame,dataframe_masked


def sintel_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000]):

    flow_pth   = 'training/flow/'
    flow_pth = os.path.join(dataset_pth, flow_pth)
    flo_seq = sorted(os.listdir(flow_pth))

    n_frames = 0
    for item in flo_seq:
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item)))
        n_frames += len(flo_names)
        print(item, len(flo_names))

    print('TOT ' + str(n_frames) + ' flow fields')
    #all_rows = []

    list_full_frame = []
    list_masked = []
    for item in flo_seq:
        print(item)
        flo_names = sorted(os.listdir(os.path.join(flow_pth, item)))
        start = time.time()
        for i in range(0,len(flo_names)):
            flo_i = flo_names[i]
            flo_pth = os.path.join(flow_pth, item, flo_i)
            row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
            row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth,thresholds=thresholds)
            list_full_frame.append(row_full_frame)
            list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)

    return dataframe_full_frame,dataframe_masked

def matlab_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000]):
    data_absolute_pth = os.path.dirname(dataset_pth)
    input_csv_pth = os.path.join(dataset_pth, dataset_pth.split('/')[-1] + '.csv')
    input_csv_df = pd.read_csv(input_csv_pth)
    list_full_frame = []
    list_masked = []
    for index, row in input_csv_df.iterrows():
        flo_pth = os.path.join(data_absolute_pth,row[input_csv_df.columns[2]])
        row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flo_pth)
        row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flo_pth, thresholds=thresholds)
        list_full_frame.append(row_full_frame)
        list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked


def chairs_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000]):

    flows = sorted(glob(osp.join(dataset_pth, '*.flo')))
    list_full_frame = []
    list_masked = []
    for i in range(len(flows)):
        row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flows[i])
        row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flows[i], thresholds=thresholds)
        list_full_frame.append(row_full_frame)
        list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked

def things_generate_gnd_dataframe_full_masked(dataset_pth, thresholds = [0,2,5,10,20,30,50,1000],\
                                              evaluate_past=True,evaluate_future=False):
    root = dataset_pth
    list_full_frame = []
    list_masked = []
    for cam in ['left']:
        for direction in ['into_future', 'into_past']:

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
            flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

            for  fdir in  flow_dirs:
                print(fdir)
                flows = sorted(glob(osp.join(fdir, '*.pfm')))
                for i in range(len(flows)):
                    if direction == 'into_future':
                        if evaluate_future == True:
                            row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flows[i])
                            row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flows[i],
                                                                                              thresholds=thresholds)
                            list_full_frame.append(row_full_frame)
                            list_masked.append(row_masked)
                    elif direction == 'into_past':
                        if evaluate_past == True:
                            row_full_frame = stats_from_gnd_matrix.full_frame_row_from_file_pth(flows[i])
                            row_masked = stats_from_gnd_matrix.masked_frame_row_from_file_pth(flows[i],
                                                                                          thresholds=thresholds)
                            list_full_frame.append(row_full_frame)
                            list_masked.append(row_masked)

    dataframe_full_frame = pd.DataFrame(list_full_frame)
    dataframe_masked = pd.DataFrame(list_masked)
        ##add stats for tlr, tud vs T180, m1, m2 for paper
    return dataframe_full_frame, dataframe_masked


def save_dataframe(data,name,results_pth):
    if not os.path.exists(results_pth):
        os.makedirs(results_pth)
    csv_pth = os.path.join(results_pth,name + '.csv')
    data.to_csv(csv_pth, index=True)

    return True
def weighted_mean_by_column(data,column):
    b = data[column+'_counts'] * data[column]
    c = b/data[column+'_counts'].sum()
    return c.sum()
def full_frame_means(data,weighted_columns,columns):

    means ={}
    for item in weighted_columns:
        means[item]= weighted_mean_by_column(data, item)
    means.update(data[columns].mean().to_dict())
    return pd.Series(means)
def save_dataframes(name,dest_pth,full_df,masked_df):
    columns = ['G_L2', 'G_L1_u', 'G_L1_v']
    weighted_columns =['Gu_maj_0','Gu_min_0','Gv_maj_0','Gv_min_0']
    means_full = full_frame_means(full_df, weighted_columns, columns)
    means_masked = masked_stats_to_dataframe(masked_df,masked_df.columns,list_counts = 'target_pixel_counts',mean_type='m1')
    save_dataframe(means_full,name+'_full',dest_pth)
    save_dataframe(means_masked,name+'_masked',dest_pth)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--chairs_pth', type=str, nargs='+')
    parser.add_argument('--chairs2_pth', type=str, nargs='+')
    parser.add_argument('--chairsOcc_pth', type=str, nargs='+')
    parser.add_argument('--things_pth', type=str, nargs='+')
    parser.add_argument('--monkaa_pth', type=str, nargs='+')
    parser.add_argument('--hd1k_pth', type=str, nargs='+')
    parser.add_argument('--kitti_pth', type=str, nargs='+')
    parser.add_argument('--matlab_pth', type=str, nargs='+')
    parser.add_argument('--matlab_equiv_pth', type=str, nargs='+')
    parser.add_argument('--sintel_pth', type=str, nargs='+')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[0,5,20, 10000])
    parser.add_argument('--results_pth', type=str, nargs='+')

    args = parser.parse_args()
    thresholds = args.thresholds

    # full_df, masked_df =chairsOcc_generate_gnd_dataframe_full_masked(args.chairsOcc_pth[0], thresholds=args.thresholds,
    #                                            flow_direction='all')
    dest_pth= args.results_pth[0]
    # save_dataframes('chairsOcc', dest_pth, full_df, masked_df)
    # full_df, masked_df =chairsOcc_generate_gnd_dataframe_full_masked(args.chairsOcc_pth[0], thresholds=args.thresholds,
    #                                            flow_direction='fwd')
    # dest_pth= args.results_pth[0]
    # save_dataframes('chairsOcc_fwd', dest_pth, full_df, masked_df)
    # full_df, masked_df =chairsOcc_generate_gnd_dataframe_full_masked(args.chairsOcc_pth[0], thresholds=args.thresholds,
    #                                            flow_direction='bck')
    # save_dataframes('chairsOcc_bck', dest_pth, full_df, masked_df)
    # full_df, masked_df =chairs2_generate_gnd_dataframe_full_masked(args.chairs2_pth[0], args.thresholds,
    #                                            chairs2_flow_direction='all')
    # dest_pth= args.results_pth[0]
    # save_dataframes('chairs2', dest_pth, full_df, masked_df)
    # full_df, masked_df =chairs2_generate_gnd_dataframe_full_masked(args.chairs2_pth[0], args.thresholds,
    #                                            chairs2_flow_direction='fwd')
    # dest_pth= args.results_pth[0]
    # save_dataframes('chairs2_fwd', dest_pth, full_df, masked_df)
    # full_df, masked_df =chairs2_generate_gnd_dataframe_full_masked(args.chairs2_pth[0], args.thresholds,
    #                                            chairs2_flow_direction='bck')
    # dest_pth= args.results_pth[0]
    # save_dataframes('chairs2_bck', dest_pth, full_df, masked_df)
    full_df, masked_df = kitti_generate_gnd_dataframe_full_masked(args.kitti_pth[0],thresholds = args.thresholds)
    dest_pth= args.results_pth[0]
    save_dataframes('kitti', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df = hd1k_generate_gnd_dataframe_full_masked(args.hd1k_pth[0],thresholds = args.thresholds)
    # dest_pth= args.results_pth[0]
    # save_dataframes('hd1k', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =monkaa_generate_gnd_dataframe_full_masked(args.monkaa_pth[0], thresholds = args.thresholds,
    #                                                               evaluate_past=True, evaluate_future=True)
    # dest_pth= args.results_pth[0]
    # save_dataframes('monkaa', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =monkaa_generate_gnd_dataframe_full_masked(args.monkaa_pth[0], thresholds = args.thresholds, \
    #                                                               evaluate_past=True, evaluate_future=False)
    # dest_pth= args.results_pth[0]
    # save_dataframes('monkaa_past', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =monkaa_generate_gnd_dataframe_full_masked(args.monkaa_pth[0], thresholds = args.thresholds,
    #                                                               evaluate_past=False, evaluate_future=True)
    # dest_pth= args.results_pth[0]
    # save_dataframes('monkaa_future', dest_pth, full_df, masked_df)

    #
    #
    # full_df, masked_df =things_generate_gnd_dataframe_full_masked(args.things_pth[0], thresholds = args.thresholds, \
    #                                                               evaluate_past=True, evaluate_future=False)
    #
    # dest_pth= args.results_pth[0]
    # save_dataframes('things_no_future', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =things_generate_gnd_dataframe_full_masked(args.things_pth[0], thresholds = args.thresholds,
    #                                                               evaluate_past=False, evaluate_future=True)
    # dest_pth= args.results_pth[0]
    # save_dataframes('things_no_past', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =things_generate_gnd_dataframe_full_masked(args.things_pth[0], thresholds = args.thresholds,
    #                                                               evaluate_past=True, evaluate_future=True)
    # dest_pth= args.results_pth[0]
    # save_dataframes('things', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df =chairs_generate_gnd_dataframe_full_masked(args.chairs_pth[0],thresholds = args.thresholds)
    # dest_pth= args.results_pth[0]
    # save_dataframes('chairs', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df = matlab_generate_gnd_dataframe_full_masked(args.matlab_pth[0],thresholds = args.thresholds)
    # dest_pth= args.results_pth[0]
    # save_dataframes('kaleidoscope', dest_pth, full_df, masked_df)
    #
    # full_df, masked_df = matlab_generate_gnd_dataframe_full_masked(args.matlab_pth[0],thresholds = args.thresholds)
    # dest_pth= args.results_pth[0]
    # save_dataframes('mat_equiv', dest_pth, full_df, masked_df)
    #
    # full_df,masked_df = sintel_generate_gnd_dataframe_full_masked(args.sintel_pth[0],thresholds = args.thresholds)
    # dest_pth= args.results_pth[0]
    # save_dataframes('sintel', dest_pth, full_df, masked_df)


