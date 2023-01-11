import argparse
import cv2
import numpy as np
import numpy.ma as ma
import utils.mask_utils as mask_utils
from PIL import Image
import networks.RAFT.core.utils.frame_utils as frame_utils
import utils.flow
def read_gnd_from_path(flow_pth):
    target = cv2.readOpticalFlow(flow_pth)
    return target


def stats_from_matrix(target):
    row={}
    target_magL2_sq = np.square(target.copy())  # elementwise square
    target_magL2_sq_summed = target_magL2_sq[:, :, 0] + target_magL2_sq[:, :, 1]
    target_magL2 = np.sqrt(target_magL2_sq_summed)

    target_u = target.copy()[:, :, 0]
    target_v = target.copy()[:, :, 1]
    row['Gu_maj_0'] = np.mean(target_u[target_u > 0])
    row['Gu_maj_0_counts'] = len(target_u[target_u > 0])
    row['Gu_min_0'] = np.mean(target_u[target_u < 0])
    row['Gu_min_0_counts'] = len(target_u[target_u < 0])
    row['Gv_maj_0'] = np.mean(target_v[target_v > 0])
    row['Gv_maj_0_counts'] = len(target_v[target_v > 0])
    row['Gv_min_0'] = np.mean(target_v[target_v < 0])
    row['Gv_min_0_counts'] =len(target_v[target_v < 0])

    row['G_L2'] = np.mean(target_magL2)
    row['G_L1_u'] = np.mean(abs(target_u))
    row['G_L1_v'] = np.mean(abs(target_v))

    return row

def stats_from_matrix_masked(target,thresholds,include_lower_bound = True,include_upper_bound=False):
    row={}
    target_magL2_sq = np.square(target.copy())  # elementwise square
    target_magL2_sq_summed = target_magL2_sq[:, :, 0] + target_magL2_sq[:, :, 1]
    target_magL2 = np.sqrt(target_magL2_sq_summed)

    target_u = target.copy()[:, :, 0]
    target_v = target.copy()[:, :, 1]

    G_L2_sum, target_mag_sums,target_pixel_counts = mask_utils.mask_matrix_over_target(target_magL2,target_magL2,thresholds=thresholds,include_lower_bound = include_lower_bound,include_upper_bound=include_upper_bound)
    row['thresholds']=thresholds
    row['G_L2_sum'] = G_L2_sum#you also need to return the number of items
    row['target_mag_sums'] =target_mag_sums
    row['target_pixel_counts'] = target_pixel_counts

    return row

def full_frame_row_from_file_pth(flow_pth):
    if flow_pth[-4:] =='.flo':
        target = read_gnd_from_path(flow_pth)
    elif flow_pth[-4:] =='.pfm':
        target = frame_utils.readPFM(flow_pth)[...,:2]
    elif flow_pth[-4:]=='.png':

        # target =  Image.open(flow1_pth)
        # target = np.array(target).astype(np.float32)[:,:,:2]#check@!!!!!!!!
        target,valid = utils.flow.read_png_flow(flow_pth)
        invalid_areas = np.concatenate((valid.copy(), valid.copy()), axis=2) == 0
        target[invalid_areas]=np.NaN
    target_stats = stats_from_matrix(target)
    return target_stats

def masked_frame_row_from_file_pth(flow_pth,thresholds):
    if flow_pth[-4:] == '.flo':
        target = read_gnd_from_path(flow_pth)
    elif flow_pth[-4:] == '.pfm':
        target = frame_utils.readPFM(flow_pth)
    elif flow_pth[-4:] == '.png':
        # target =  Image.open(flow1_pth)
        # target = np.array(target).astype(np.float32)[:,:,:2]#check@!!!!!!!!
        target = utils.flow.read_png_flow(flow_pth)[0]
    target_masked_stats = stats_from_matrix_masked(target, thresholds)
    return target_masked_stats

# check groundtruth > 0, <0
# histogram of motion magnitude masked
# mean gnd_mag l2, L1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_pth', type=str, nargs='+')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[0, 5, 20, 10000])

    args = parser.parse_args()
    thresholds = args.thresholds
    target = read_gnd_from_path(args.flow_pth[0])
    target_stats = stats_from_matrix(target)
    target_masked_stats = stats_from_matrix_masked(target, thresholds)

    print(target_stats)